/* Copyright (c) 1998-2002 Ben Jackson (ben@ben.com).  All rights reserved.
 *
 * Use and copying of this software and preparation of derivative works based
 * upon this software are permitted provided this copyright notice remains
 * intact.
 * 
 * THIS SOFTWARE IS PROVIDED BY BEN J JACKSON ``AS IS'' AND ANY
 * EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT BEN J JACKSON BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "my-ctype.h"
#include "my-stdio.h"

#include "ast.h"
#include "config.h"
#include "decompile.h"
#include "exceptions.h"
#include "functions.h"
#include "keywords.h"
#include "list.h"
#include "log.h"
#include "opcode.h"
#include "program.h"
#include "unparse.h"
#include "storage.h"
#include "streams.h"
#include "utils.h"
#include "optimize.h"

void markup_stmts(Stmt * top);

struct Var_Attr {
    /*
     * Attributes of a variable inherited from the rhs of assignments
     * to that variable.
     */
    unsigned short typemask;
    unsigned value_serial;
    unsigned maybe_put:1;
    unsigned definitely_put:1;
    /* XXX should move in known_true etc */
};

#define MAX_VARATTR	64

struct Forw_Attr {
    struct Var_Attr rt_env[MAX_VARATTR];
};

static void markup_stmts_internal(Stmt * top, struct Forw_Attr *f);

struct Forw_Expr_Stack {
    struct Forw_Expr_Stack *parent, *saved_parent;
    struct Forw_Attr f;
    unsigned is_f_initialized:1;
    unsigned typemask;
    unsigned bits_clrd;
};

static int bf_length_index, bf_typeof_index;

/* XXX waifs -- can't use this on a waif */
static struct {
	unsigned namelen;
	const char *name;
	unsigned typemask;
} bi_prop_types[] = {
	{ 4,  "name",		TYPEMASK(TYPE_STR) },
	{ 5,  "owner",		TYPEMASK(TYPE_OBJ) },
	{ 10, "programmer",	TYPEMASK(TYPE_INT) },
	{ 6, "wizard",		TYPEMASK(TYPE_INT) },
	{ 1, "r",		TYPEMASK(TYPE_INT) },
	{ 1, "w",		TYPEMASK(TYPE_INT) },
	{ 1, "f",		TYPEMASK(TYPE_INT) },
	{ 8, "location",	TYPEMASK(TYPE_OBJ) },
	{ 8, "contents",	TYPEMASK(TYPE_LIST) },
	{ 0, NULL }
};

struct Optimizer;

struct Loop_Context {
    struct Loop_Context *next;
    Stmt *root;
    unsigned id;
    struct Forw_Attr at_end;
    struct Forw_Attr after_end;

    /* crufty old back attr */
    unsigned putmap_after_end;
};

struct Optimizer {
    /*
     * Layers of error handling enclosing the current AST node.
     * A block within an error handling context can jump to the
     * end (by unwinding the error) at any point, so these blocks
     * cannot generate any promises.
     */
    unsigned try_depth;

    /*
     * Crufty old attributes for backwards pass:
    *
     * While flowing backwards the putmap has 1 bits set whenever
     * there is an uninterrupted path from the current location forward
     * to a PUT of a variable.  That means that when we find a PUSH
     * (still moving backwards) it can PUSH_CLEAR.
     */
    unsigned putmap;

    /*
     * While flowing backwards the pushmap gets 1 bits set whenever
     * a PUSH is encountered.  Because the PUSH semantics are 'a push
     * on ANY path' they are much simpler.  This is used to probe
     * inside a loop and determine what variables are *not* referenced
     * by it and allow their putmap bits to pass the loop.
     *
     * You get the same answer going forward so it isn't even computed.
     *
     * The pushmap is saved and restored to learn about individual
     * blocks of code.  When that happens pushmap_global will be
     * have pushmap or'd into it so that the union of both reflects
     * all subsequent code.
     */
    unsigned pushmap;
    unsigned pushmap_global;

    /*
     * Linked list of loop contests, innermost first
     */
    struct Loop_Context *loops;

    /*
     * The [] or [..] expr that contains us (for $ to refer to)
     */
    Expr *index_context;

    /*
     * Attributes for forward analysis.
     */
    struct Forw_Attr f;

    int allocated;
};

static void
free_optimizer(struct Optimizer *opt)
{
    if (opt->allocated && opt->loops)
	myfree(opt->loops, M_CODE_GEN);
}

static struct Optimizer *
deep_copy_optimizer(struct Optimizer *opt)
{
    struct Optimizer *ret;
    struct Loop_Context *loops;
    int i;

    ret = mymalloc(sizeof(struct Optimizer), M_CODE_GEN);
    memcpy(ret, opt, sizeof(struct Optimizer));
    ret->allocated = 1;

    for (i = 0, loops = opt->loops; loops; loops = loops->next)
	++i;
    if (i) {
	struct Loop_Context *t;

	ret->loops = t = mymalloc(sizeof(struct Loop_Context) * i, M_CODE_GEN);
	for (loops = opt->loops; loops; loops = loops->next, ++t) {
	    *t = *loops;
	    t->next = t + 1;
	}
	--t;
	t->next = NULL;
    }
    if (opt->index_context)
	panic("deep_copy_optimizer not expected mid-expression");
    return ret;
}

static void foreopt_attr_identity(struct Forw_Attr *fp);

static void
enter_loop_context(struct Optimizer *opt, struct Stmt *stmt, int id)
{
    struct Loop_Context *loop;

    loop = mymalloc(sizeof(struct Loop_Context), M_CODE_GEN);

    loop->root = stmt;
    loop->id = id;
    loop->next = opt->loops;
    opt->loops = loop;
    foreopt_attr_identity(&loop->at_end);
    foreopt_attr_identity(&loop->after_end);
    /*
     * XXX make some mark (save opt?) so break/continue can know if this is
     * XXX even an "important" context or a throwaway.
     */
}

static void
exit_loop_context(struct Optimizer *opt, struct Loop_Context *loop)
{
    opt->loops = loop->next;
    myfree(loop, M_CODE_GEN);
}

static inline unsigned
next_value_serial()
{
    static unsigned serial;

    return serial++;
}

#define VR_VAR_MASK    ((1 << SLOT_ARGSTR) | \
			(1 << SLOT_DOBJ) | \
			(1 << SLOT_DOBJSTR) | \
			(1 << SLOT_PREPSTR) | \
			(1 << SLOT_IOBJ) | \
			(1 << SLOT_IOBJSTR) | \
			(1 << SLOT_PLAYER))

/*   ______    _____ ___  ______        ___    ____  ____   ______  
 *   \ \ \ \  |  ___/ _ \|  _ \ \      / / \  |  _ \|  _ \  \ \ \ \ 
 *    \ \ \ \ | |_ | | | | |_) \ \ /\ / / _ \ | |_) | | | |  \ \ \ \
 *    / / / / |  _|| |_| |  _ < \ V  V / ___ \|  _ <| |_| |  / / / /
 *   /_/_/_/  |_|   \___/|_| \_\ \_/\_/_/   \_\_| \_\____/  /_/_/_/ 
 */

static int foreopt_stmt(struct Optimizer *opt, Stmt * stmt);
static int foreopt_stmt_cond(struct Optimizer *opt, struct Stmt_Cond cond);
static void foreopt_stmt_catch(struct Optimizer *opt, struct Stmt_Catch catch);
static void foreopt_scatter(struct Optimizer *opt, Scatter * sc);
static void foreopt_arglist(struct Optimizer *opt, Arg_List * args, struct Forw_Expr_Stack *, int, var_type *);
static void foreopt_expr(struct Optimizer *opt, Expr * expr, struct Forw_Expr_Stack *);
static void foreopt_expr_typemask(struct Optimizer *opt, Expr *expr, unsigned typemask);
static void foreopt_expr_true_cond(struct Optimizer *opt, Expr *cond);

unsigned
prototype_to_typemask(var_type t)
{
    switch(t) {
    case TYPE_ANY:
	return TYPEMASK_ANY;
    case TYPE_NUMERIC:
	return TYPEMASK(TYPE_INT) | TYPEMASK(TYPE_FLOAT);
    case TYPE_LISTORSTR:
	return TYPEMASK(TYPE_LIST) | TYPEMASK(TYPE_STR);
    default:
	return TYPEMASK(t);
    }
}

/*
 * Initialize a Forw_Attr that is an identity when combined with
 # the foreopt_attr_join operation
 */
static void
foreopt_attr_identity(struct Forw_Attr *fp)
{
    int i;
    for (i = 0; i < MAX_VARATTR; ++i) {
	fp->rt_env[i].typemask = 0;
	fp->rt_env[i].maybe_put = 0;
	fp->rt_env[i].definitely_put = 1;
    }
}

/*
 * Record a PUT
 *
 * I originally through that PUTs in error handling had to be ignored,
 * but in fact they CAN'T be ignored, but anything in error handling
 * must pass through a _might_not_happen().
 */
static inline void
foreopt_attr_do_put(struct Forw_Attr *fp, int id, unsigned typemask, int definite)
{
    if (id < 0)
	return;
    if (id < MAX_VARATTR) {
	fp->rt_env[id].typemask = typemask;	/* mask of possible types */
	fp->rt_env[id].value_serial = next_value_serial();
	fp->rt_env[id].maybe_put = 1;		    /* flag of maybe put */
	fp->rt_env[id].definitely_put = definite ||
			fp->rt_env[id].definitely_put;  /* flag of no kidding */
    }
}

static inline void
foreopt_attr_set_typemask(struct Forw_Attr *fp, int id, unsigned typemask)
{
    if (id < 0)
	return;
    if (id < MAX_VARATTR) {
	fp->rt_env[id].typemask = typemask;
    }
}


/*
 * Record a PUSH_CLEAR -- a push stealing the ref, leaving only scorched
 * earth.  The type inference should never "discover" this value again,
 * but it can be used to know that no free_var() is needed on this slot
 * on the next push.
 */
static inline void
foreopt_attr_do_last_use(struct Forw_Attr *fp, int id)
{
    if (id < 0)
	return;
    if (id < MAX_VARATTR) {
	fp->rt_env[id].typemask = TYPEMASK(TYPE_NONE);
	fp->rt_env[id].maybe_put = 0;
	fp->rt_env[id].definitely_put = 0;
    }
}

/*
 * Test the putmap -- the "MUST have put by this point" bit.  When you
 * absolutely positively must have a value of some kind already in the
 * variable slot.
 */
static inline int
foreopt_attr_test_definitely_put(struct Forw_Attr *fp, int id)
{
    if (id < 0 || id >= MAX_VARATTR)
	return 0;
    return fp->rt_env[id].definitely_put;
}

/*
 * Test the putmap -- the "MAY have put by this point" bit.  When you
 * wonder if a variable has had any chance to be set by this point.
 */
static inline int
foreopt_attr_test_maybe_put(struct Forw_Attr *fp, int id)
{
    if (id < 0 || id >= MAX_VARATTR)
	return 1;		/* note well, it might be! */
    return fp->rt_env[id].maybe_put;
}

/*
 * Retrieve the saved type mask
 */
static inline unsigned
foreopt_attr_get_typemask(struct Forw_Attr *fp, int id)
{
    if (id < 0 || id >= MAX_VARATTR)
	return TYPEMASK_ANY;		/* note well, it might be! */
    return fp->rt_env[id].typemask;
}

/*
 * Retrieve the saved value_serial.  It's particularly important to use
 * the accessor here because of the unusual behavior for unmapped slots.
 */
static inline unsigned
foreopt_attr_get_value_serial(struct Forw_Attr *fp, int id)
{
    if (id < 0 || id >= MAX_VARATTR)
	return next_value_serial();  /* act like it always changes */
    return fp->rt_env[id].value_serial;
}

/*
 * When two potential threads of execution merge, this function combines
 * their states into the new set of possibilities going forward.
 */
static inline void
foreopt_attr_join(struct Forw_Attr *LHS, struct Forw_Attr *RHS)
{
    int i;
    struct Var_Attr *lv = LHS->rt_env, *rv = RHS->rt_env;

    for (i = 0; i < MAX_VARATTR; ++i, ++lv, ++rv) {
	lv->typemask |= rv->typemask;
	lv->maybe_put = lv->maybe_put || rv->maybe_put ||
			lv->definitely_put || rv->definitely_put;
	lv->definitely_put = lv->definitely_put && rv->definitely_put;
	if (lv->value_serial != rv->value_serial)
	    lv->value_serial = next_value_serial();
    }
}

/*
 * Copy
 */
static inline void
foreopt_attr_copy(struct Forw_Attr *LHS, struct Forw_Attr *RHS)
{
    memcpy(LHS, RHS, sizeof(*LHS));
}

/*
 * Test Equality.  Ignore value_serial -- I'm not smart enough to
 * figure out what equality would mean for that!
 */
static inline int
foreopt_attr_equal(struct Forw_Attr *LHS, struct Forw_Attr *RHS)
{
    /* some undef mem return !memcmp(LHS, RHS, sizeof(*LHS)); */
    int i;
    struct Var_Attr *lv = LHS->rt_env, *rv = RHS->rt_env;

    for (i = 0; i < MAX_VARATTR; ++i, ++rv, ++lv) {
	if (lv->typemask != rv->typemask || lv->maybe_put != rv->maybe_put ||
	    lv->definitely_put != rv->definitely_put)
	    return 0;
    }
    return 1;
}

/*
 * Forget what we know about types.  Optionally pass in a mask attribute
 * and only forget about types with the maybe_put attribute.
 */
static inline void
foreopt_attr_oubliette(struct Forw_Attr *fp, struct Forw_Attr *mask)
{
    int i;
    for (i = 0; i < MAX_VARATTR; ++i) {
	if (mask && foreopt_attr_test_maybe_put(mask, i))
	    continue;
	fp->rt_env[i].typemask = TYPEMASK_ANY;
	fp->rt_env[i].value_serial = next_value_serial();
    }
}

/*
 * Wherever rhs.put, it was filled in by inherited type information.
 * Unlike the common attr_join case, this is not a time when two
 * threads recombine (allowing any of the possible types from the
 * threads) but rather a time when two pieces of INFORMATION join,
 * both of which are true, so the resulting mask is the intersection.
 */
static inline void
foreopt_attr_intersect_put(struct Forw_Attr *lhs, struct Forw_Attr *rhs)
{
    int i;
    for (i = 0; i < MAX_VARATTR; ++i)
	if (foreopt_attr_test_definitely_put(rhs, i))
	    lhs->rt_env[i].typemask &= rhs->rt_env[i].typemask;
}

/*
 * A variable has inherited type information.  It is being pushed as
 * part of an expression whose eventual type we know must be in a
 * certain set for the expression to avoid raising an error.
 *
 *                  .--this function moves the info----.
 *                  |                                  V
 * (i + 1) => push i, push 1, add <-- but here we know it was int i
 *                           |
 *                           `--here we still need typechecking
 *
 * We search up the expression stack looking for the closest parent
 * which will make the type promise that we need.  We might be nested
 * ala str_or_int_or_float(int_or_float(int(us))) in which case we will
 * spread our variable info over all of the levels necessary until we
 * have accounted for all of the type promises.
 *
 * Implementation note:  If we find a level that is not initialized
 * we initialize it here.  Filling it isn't cheap, and most expressions
 * are not going to ever need to make good on their promises.
 */
static inline void
foreopt_attr_inherit_typemask(struct Forw_Expr_Stack *estk, int id)
{
    unsigned oldmask = TYPEMASK_ANY;
    unsigned removed_bits;

    if (id < 0 || id >= MAX_VARATTR)
	return;
    if (!estk)
	return;

    removed_bits = oldmask & ~estk->typemask;
    while (estk && removed_bits) {
	unsigned found = estk->bits_clrd & removed_bits;
	if (found) {
	    if (!estk->is_f_initialized) {
		int i;

		for (i = 0; i < MAX_VARATTR; ++i) {
			estk->f.rt_env[i].typemask = TYPEMASK_ANY;
			estk->f.rt_env[i].maybe_put = 0;
			estk->f.rt_env[i].definitely_put = 0;
		}
		estk->is_f_initialized = 1;
	    }
	    oldmask &= ~found;
	    removed_bits &= ~found;
	    /*
	     * If we are not the first, we add our information.  Unlike
	     * other cases where two typemasks converge, the result must
	     * be the instersection of two types, such as in:
	     *
	     * (<<j si> + i si>) + (<<k fi> + i fi>)
	     *                 ^ here <i si>       ^ here <i fi>
	     *                                                  ^here <i i>
	     */
	    if (estk->f.rt_env[id].definitely_put)
		estk->f.rt_env[id].typemask &= oldmask;
	    else
		foreopt_attr_do_put(&estk->f, id, oldmask, 1);
	}
	estk = estk->parent;
    }
    if (removed_bits)
	fprintf(stderr, "did not expect leftover bits!\n");
}

/*
 * Execute one pass through a loop, optionally preserving the attributes
 * of the incoming tree (by copying it before evaluating it) and optionally
 * stopping at the point just inside the end of the loop (before the next
 * iteration) rather than falling clear off the end of the loop.
 *
 * If loop_cond is passed in, this is a while loop.  Its condition testing
 * is part of the loop.
 */
static int
foreopt_stmt_loop_help(struct Optimizer *opt, Stmt * stmt, Expr * loop_cond, int loop_id, Stmt * loop_stmt, int preserve_attrs, int stop_within_loop)
{
    struct Loop_Context *loop;
    struct Forw_Attr loop_id_save;
    int completes;

    enter_loop_context(opt, loop_stmt, loop_id);
    loop = opt->loops;

    /*
     * If we got a cond we're in a while loop. The while cond happens
     * first (possibly setting an id var in the process).  At that point
     * if it's false, it's like a BREAK.  So the state after the cond
     * goes into the _after_end state.
     */
    if (loop_cond) {
	if (preserve_attrs)
	    loop_cond = copy_expr(loop_cond);
	foreopt_expr(opt, loop_cond, NULL);
	foreopt_attr_do_put(&opt->f, loop_id, loop_cond->a.typemask, 1);
	/*
	 * If the condition is known to be true it will always fall
	 * through, never branch straight to the end (carrying this
	 * state forward).
	 */
	if (!loop_cond->a.known_true)
	    foreopt_attr_copy(&loop->after_end, &opt->f);
	/*
	 * Of course, if we DO fall through, we know this cond is
	 * true:
	 */
	foreopt_expr_true_cond(opt, loop_cond);

	if (preserve_attrs)
	    free_expr(loop_cond);
    }
    /*
     * This is the point where RANGE/LIST loops may branch to the end
     * ala BREAK.  They don't set their name variable on the last loop.
     */
    if (loop_stmt && (loop_stmt->kind == STMT_LIST ||
		      loop_stmt->kind == STMT_RANGE)) {
	foreopt_attr_copy(&loop->after_end, &opt->f);
    }
    /*
     * This is the moment where the loop_id variable gets set, so
     * drag in the type inferences!  These won't get set after the
     * last pass, or at all if the range/list is null, so we have
     * suck them back out after the stmts run.
     */
    switch (loop_stmt->kind) {
    case STMT_WHILE:
	/* done above with cond */
	break;
    case STMT_RANGE:
	/*
	 * Inside the range the loop variable has the type of the from/to
	 * range.  Can be int or obj, but they must be equal, so take the
	 * intersection as the possibility
	 */
	if (loop_id < MAX_VARATTR) {
	    foreopt_attr_identity(&loop_id_save);
	    loop_id_save.rt_env[loop_id] = opt->f.rt_env[loop_id];
	    foreopt_attr_join(&loop->after_end, &loop_id_save);
	}
	/*
	 * Memorize the typemask of the loop variable at this moment so
	 * that the generated code can avoid freeing it if we know it to
	 * be a simple type (99.999% of the time it's still going to be
	 * an int because it's the loop var!)
	 */
	loop_stmt->a.u.range.loopvar_typemask =
		foreopt_attr_get_typemask(&opt->f, loop_id);
	foreopt_attr_do_put(&opt->f, loop_id,
			    loop_stmt->s.range.from->a.typemask &
			    loop_stmt->s.range.to->a.typemask, 1);
	break;
    case STMT_LIST:
	/*
	 * Don't know!
	 */
	if (loop_id < MAX_VARATTR) {
	    foreopt_attr_identity(&loop_id_save);
	    loop_id_save.rt_env[loop_id] = opt->f.rt_env[loop_id];
	    foreopt_attr_join(&loop->after_end, &loop_id_save);
	}
	foreopt_attr_do_put(&opt->f, loop_id, TYPEMASK_ANY, 1);
	break;
    default:
	errlog("FOREOPT_loopmumble..: Unknown loop Stmt_Kind: %d\n", loop_stmt->kind);
	break;
    }

    /*
     * Now the body, possibly in a copy.
     */
    if (preserve_attrs)
	stmt = copy_stmt(stmt);
    completes = foreopt_stmt(opt, stmt);
    if (preserve_attrs)
	free_stmt(stmt);

    /*
     * We're just inside the ENDFOO now and it's time to merge the
     * paths within the loop that branched directly to this point
     * by CONTINUE.  The continues all merged into the _at_end data.
     *
     * If by some chance the satements can't complete (eg ifelse where
     * every arm and the else all break/continue/return) then our
     * only state is that which came to us via CONTINUE.  But IF we
     * never continued (and have no _at_end thereby) then we know
     * something wonderful:  this loop never iterates!  That means
     * we can stuff the state with the most permissive values so it
     * doesn't count as a path when we rejoin below.
     */
    if (completes)
	foreopt_attr_join(&opt->f, &loop->at_end);
    else
	foreopt_attr_copy(&opt->f, &loop->at_end);

    /*
     * If the caller wants to propagate information back to the top of
     * the loop this is where we finish.  The current state represents
     * the point just before the next iteration would begin.
     */
    if (stop_within_loop) {
	exit_loop_context(opt, loop);
	return completes;
    }
    /*
     * Now we have reached the bottom by falling off (or for a WHILE,
     * by going around by the cond again and failing).  Merge the paths
     * which reached this point by BREAK, in the _after_end data.
     */
    foreopt_attr_join(&opt->f, &loop->after_end);

    exit_loop_context(opt, loop);

    /*
     * The current state now reflects the executed-loop state and is
     * ready to be merged with a saved copy of the pre-loop state.
     */
    return completes;
}

#define FANCYPANTS_LOOP_LVL 2
static void
foreopt_stmt_loop_body(struct Optimizer *opt, Stmt * stmt, Expr * loop_cond, int loop_id, Stmt * loop_stmt)
{
#if (FANCYPANTS_LOOP_LVL == 0)
    /*
     * Since a loop starts from either the top (the state we have now)
     * or repeats from the bottom (state we don't yet know) we can't
     * simply take the union of those states.  Instead we take a far
     * more pessimistic starting point -- knowing nothing at all!
     * This is worse off than we are at the start of the verb.
     * XXX we could preserve untouched vars around it
     */
    foreopt_attr_oubliette(&opt->f, NULL);
    /* actual pass at end */
#elif (FANCYPANTS_LOOP_LVL == 1)
    /*
     * Our alternate loop strategy is to create a snapshot of our
     * current point of execution, run the loop once from zero knowledge,
     * and see where we * end up.  Think of it like unrolling the first
     * iteration.  This should be the most pessimistic possible outcome.
     * Knowing we get "at least" that much information from an iteration,
     * haul the information we got from that back up and combine it with
     * the information coming from above and do the real flow from that
     * point.
     * XXX still need to portage unmolested vars around rapids
     */
    /*
     * NB:  This is legitimate as long as any type analysis
     * starting with NARROWER type information does not generate
     * BROADER type information than an analysis *starting* with
     * broader information.  This is basically a prohibition of
     * using type information to avoid evaluating dead code *for
     * the purpose of generating type information*, since the a
     * narrower type spec which eliminates a narrowing subexpression
     * could result in a broader result.  This would invalidate our
     * assumption that starting from no knowledge gives the most
     # pessimistic result.
     */
    struct Optimizer *dum = deep_copy_optimizer(opt);

    /* IIIII know NNUUUTHINK!! */
    foreopt_attr_oubliette(&dum->f, NULL);	/* XXX set mask */
    /*
     * Execute the body, preserving the attributes and stopping just
     * inside the end of the loop (before the next iteration):
     */
    foreopt_stmt_loop_help(dum, stmt, loop_cond, loop_id, loop_stmt, 1, 1);
    /*
     * Now we know how we would end up after having no knowledge at
     * the start.  Union that with the current state of the "real"
     * optimizer and then run the loop for real (building attributes
     * and falling all the way off the end this time):
     */
    foreopt_attr_join(&opt->f, &dum->f);
    free_optimizer(dum);
    /* actual pass at end */
#elif (FANCYPANTS_LOOP_LVL == 2)
    /*
     * Our best loop type inference ever!  Start with the original input
     * state and unroll the loop until it begs for mercy.  Or until it
     * produces consistent output.
     */
    struct Optimizer *dum = deep_copy_optimizer(opt);
    struct Forw_Attr last;
    static int worst;
    int count = 0;

    do {
	++count;
	if (worst < count) {
	    worst = count;
	    fprintf(stderr, "worst unroll = %d\n", worst);
	    if (worst > 1000)
		panic("eeee");
	}
	foreopt_attr_copy(&last, &dum->f);
	foreopt_stmt_loop_help(dum, stmt, loop_cond, loop_id, loop_stmt, 1, 1);
	/*
	 * Each pass through the loop produces a set of possible outputs.
	 * We can never narrow it, so join inside the loop, and carry that
	 * information back to the top.
	 */
	foreopt_attr_join(&opt->f, &dum->f);
	foreopt_attr_copy(&dum->f, &opt->f);
    } while (!foreopt_attr_equal(&last, &dum->f));
    free_optimizer(dum);
#endif				/* FANCYPANTS_LOOP_LVL */
    foreopt_stmt_loop_help(opt, stmt, loop_cond, loop_id, loop_stmt, 0, 0);
}

/*
 * This might not happen, so it can use putmap bits but not set them.
 *
 * Possible typemasks after this list of statements are the union of
 * all possible types at the beginning, and after every exit point
 * of the statement block.  
 */
static void
foreopt_stmt_might_not_happen(struct Optimizer *opt, Stmt * stmt)
{
    struct Forw_Attr saved;

    foreopt_attr_copy(&saved, &opt->f);
    foreopt_stmt(opt, stmt);
    foreopt_attr_join(&opt->f, &saved);		/* merge with the not-us */
}

/*
 * Infer any type information available given that you know the cond
 * is true.  This should not modify the expression tree.  It's not
 * meant to be exhaustive, only to pick off some low-hanging fruit
 * associated with common expressions.
 *
 * Of course this should only be called in a context where after evaluating
 * further statements inside the conditional, the state will be rejoined
 * with the previous state.
 *
 * Since we only scratch the surface of an arbitrarily complex condition,
 * we rely on the value_serials of any EXPR_IDs we find to ensure that they
 * still have the same value they did for the parts we're peeking at.
 */
#define CURRENT(expr) ((expr)->a.value_serial == \
			foreopt_attr_get_value_serial(&opt->f, (expr)->e.id))
static void
foreopt_expr_true_cond(struct Optimizer *opt, Expr *cond)
{
    switch (cond->kind) {
    case EXPR_EQ: {
	Expr *a = cond->e.bin.lhs, *b = cond->e.bin.rhs;
	unsigned typemask = a->a.typemask & b->a.typemask;
	/*
	 * First move any constant to the b
	 */
	if (a->a.constant && !b->a.constant) {
	    Expr *tmp = a;
	    a = b;
	    b = tmp;
	}
	/*
	 * (typeof(x) == const)			XXX fix for type consts
	 * (typeof(x = expr) == const)
	 *
	 * We know the type of x because this expression was true.
	 */
	if (a->kind == EXPR_CALL && a->e.call.func == bf_typeof_index &&
	    a->e.call.args && a->e.call.args->kind == ARG_NORMAL &&
	    b->a.constant && b->kind == EXPR_VAR && b->e.var.type == TYPE_INT) {
	    Expr *id_expr = a->e.call.args->expr;
	    int type = b->e.var.v.num;

	    if (id_expr->kind == EXPR_ASGN)		/* allow form 2 */
		id_expr = id_expr->e.bin.lhs;
	    if (id_expr->kind == EXPR_ID && CURRENT(id_expr) &&
		type >= TYPE_INT && type <= _TYPE_FLOAT) { /* XXX WAIFS */
		foreopt_attr_set_typemask(&opt->f, id_expr->e.id, TYPEMASK(type));
	    }
	}
	/*
	 * x == expr
	 * expr == y
	 * (x = expr) == y
	 *
	 * We know the types of both sides were identical, so either side
	 * can only be the intersection of the two sets of possible types
	 * (computed above in typemask)
	 */
	if (a->kind == EXPR_ID && CURRENT(a) && !a->a.last_use)
	    foreopt_attr_set_typemask(&opt->f, a->e.id, typemask);
	if (b->kind == EXPR_ID && CURRENT(b) && !b->a.last_use)
	    foreopt_attr_set_typemask(&opt->f, b->e.id, typemask);
	if (a->kind == EXPR_ASGN
	    && a->e.bin.lhs->kind == EXPR_ID && CURRENT(a->e.bin.lhs))
	    foreopt_attr_set_typemask(&opt->f, a->e.bin.lhs->e.id, typemask);
	break;
    }

    case EXPR_ID:
	/*
	 * Oddly, in MOO values of some types just can't be true.
	 */
	if (!cond->a.last_use && CURRENT(cond)) {
	    unsigned typemask = foreopt_attr_get_typemask(&opt->f, cond->e.id)
					& TYPEMASK_OKTRUE;
	    foreopt_attr_set_typemask(&opt->f, cond->e.id, typemask);
	}
	break;

    case EXPR_AND:
	foreopt_expr_true_cond(opt, cond->e.bin.lhs);
	foreopt_expr_true_cond(opt, cond->e.bin.rhs);
	break;

    case EXPR_ASGN:
	foreopt_expr_true_cond(opt, cond->e.bin.rhs);
	/* if the lhs is a simple ID then recurse to pick up ID rule */
	if (cond->e.bin.lhs->kind == EXPR_ID)
	    foreopt_expr_true_cond(opt, cond->e.bin.lhs);
	break;

    default:
	break;
    }
}

/*
 * if/elseif/else/endif
 */
static int
foreopt_stmt_cond(struct Optimizer *opt, struct Stmt_Cond cond)
{
    Cond_Arm *elseifs;
    struct Forw_Attr running_cond, aggregate_stmt;
    int can_continue = 0, this_continue;

    foreopt_attr_identity(&aggregate_stmt);
    for (elseifs = cond.arms; elseifs; elseifs = elseifs->next) {
	if (elseifs != cond.arms)	/* not first */
	    foreopt_attr_copy(&opt->f, &running_cond);
	foreopt_expr(opt, elseifs->condition, NULL);
	foreopt_attr_copy(&running_cond, &opt->f);
	foreopt_expr_true_cond(opt, elseifs->condition);

	this_continue = foreopt_stmt(opt, elseifs->stmt);
	can_continue = can_continue || this_continue;
	if (this_continue)
	    foreopt_attr_join(&aggregate_stmt, &opt->f);
    }

    /*
     * At this point there are two possibilities:
     *
     * WITH an else clause, we handle it as above, and the aggregated output
     * is *the* result.  There is no way around this if/else.
     *
     * WITHOUT an else clause there is an arc that goes through all of
     * the conditionals but does nothing else.
     */
    foreopt_attr_copy(&opt->f, &running_cond);

    if (cond.otherwise) {
	this_continue = foreopt_stmt(opt, cond.otherwise);
	can_continue = can_continue || this_continue;
	if (this_continue)
	    foreopt_attr_join(&aggregate_stmt, &opt->f);
    } else {
	can_continue = 1;
	foreopt_attr_join(&aggregate_stmt, &opt->f);
    }

    /*
     * The aggregate state is THE state.  If nothing reached this point
     * it's nonsensical, but nothing is going to use it!
     */
    foreopt_attr_copy(&opt->f, &aggregate_stmt);
    return can_continue;
}

static void
foreopt_stmt_list(struct Optimizer *opt, struct Stmt *stmt)
{
    foreopt_expr_typemask(opt, stmt->s.list.expr, TYPEMASK(TYPE_LIST));
    foreopt_stmt_loop_body(opt, stmt->s.list.body, NULL, stmt->s.range.id, stmt);
}

static void
foreopt_stmt_range(struct Optimizer *opt, struct Stmt *stmt)
{
    /* can range over 1..n or #1..#n */
    /* XXX this promise of type inheritance doesn't become true
     * until the loop begins to execute, AFTER the `to' range is
     * put.  I should wrap both of these in a Forw_Expr_Stack, but I'm
     * lazy right now
     */
    // foreopt_expr_typemask(opt, stmt->s.range.from,
    //			TYPEMASK(TYPE_INT)|TYPEMASK(TYPE_OBJ));
    foreopt_expr(opt, stmt->s.range.from, NULL);
    /* ...but they must both have the same type */
    foreopt_expr_typemask(opt, stmt->s.range.to,
				(stmt->s.range.from->a.typemask &
				 (TYPEMASK(TYPE_INT)|TYPEMASK(TYPE_OBJ))));
    foreopt_stmt_loop_body(opt, stmt->s.range.body, NULL, stmt->s.range.id, stmt);
}

static void
foreopt_stmt_fork(struct Optimizer *opt, struct Stmt_Fork fork_stmt)
{
    foreopt_expr_typemask(opt, fork_stmt.time, TYPEMASK_TIMER);
    /* XXX remember former typemask of id */
    foreopt_attr_do_put(&opt->f, fork_stmt.id, fork_stmt.time->a.typemask, 1);
    if (!opt->allocated) {	/* only if we're the real deal */
	markup_stmts_internal(fork_stmt.body, &opt->f);
    }
}

static void
foreopt_stmt_catch(struct Optimizer *opt, struct Stmt_Catch catch)
{
    Except_Arm *ex;
    struct Forw_Attr base, aggregate_stmt;

    ++opt->try_depth;
    foreopt_stmt_might_not_happen(opt, catch.body);
    --opt->try_depth;

    if (!catch.excepts)
	return;

    /* XXX
     * We'd like to carry in information from the above statements, but
     * they could stop and jump out to the exception handlers at any point.
     * To live and reach the bottom we must complete either the statements
     * (current state) or one of the exception handlers.
     */
    foreopt_attr_copy(&aggregate_stmt, &opt->f);
    /* XXX foreopt_attr_copy(&base, &opt->f); not correct */
    foreopt_attr_oubliette(&base, NULL);

    for (ex = catch.excepts; ex; ex = ex->next) {
	if (ex != catch.excepts)	/* not first */
	    foreopt_attr_copy(&opt->f, &base);
	if (ex->id >= 0)
	    foreopt_attr_do_put(&opt->f, ex->id, TYPEMASK(TYPE_LIST), 1);
	if (ex->codes)
	    foreopt_arglist(opt, ex->codes, NULL, 0, NULL);
	else
	    /* ANY */ ;
	foreopt_stmt(opt, ex->stmt);
	foreopt_attr_join(&aggregate_stmt, &opt->f);
    }
    /* this covers the no-errors case and each of the handlers */
    foreopt_attr_copy(&opt->f, &aggregate_stmt);
}

static int
foreopt_stmt(struct Optimizer *opt, Stmt * stmt)
{
    int final = 0;

    while (stmt) {
	switch (stmt->kind) {
	case STMT_COND:
	    /* returns whether it can be passed */
	    final = !foreopt_stmt_cond(opt, stmt->s.cond);
	    break;
	case STMT_LIST:
	    foreopt_stmt_list(opt, stmt);
	    break;
	case STMT_RANGE:
	    foreopt_stmt_range(opt, stmt);
	    break;
	case STMT_FORK:
	    foreopt_stmt_fork(opt, stmt->s.fork);
	    break;
	case STMT_EXPR:
	    foreopt_expr(opt, stmt->s.expr, NULL);
	    break;
	case STMT_WHILE:
	    foreopt_stmt_loop_body(opt, stmt->s.loop.body,
			  stmt->s.loop.condition, stmt->s.loop.id, stmt);
	    break;
	case STMT_RETURN:
	    if (stmt->s.expr)
		foreopt_expr(opt, stmt->s.expr, NULL);
	    final = 1;
	    break;
	case STMT_TRY_EXCEPT:
	    foreopt_stmt_catch(opt, stmt->s.catch);
	    break;
	case STMT_TRY_FINALLY:
	    ++opt->try_depth;
	    foreopt_stmt_might_not_happen(opt, stmt->s.finally.body);
	    --opt->try_depth;
	    foreopt_stmt(opt, stmt->s.finally.handler);
	    break;
	case STMT_CONTINUE:
	case STMT_BREAK:
	    {
		struct Loop_Context *lc;
		/*
		 * This jumps to the end (just inside or just outside)
		 * of one of the loops we're nested * in.  That means
		 * that when this arc and the loop rejoin it will have
		 * to take the union of what got here and what got to
		 * the end of the loop without no steekin' gotos.
		 */
		lc = opt->loops;	/* inner loop */
		if (stmt->s.exit >= 0) {
		    for (; lc->id != stmt->s.exit; lc = lc->next)
			/* find named loop */ ;
		}
		/* XXX identify if we're in a throwaway context */
		if (stmt->kind == STMT_BREAK) {
		    /* send our state on a vacation to the end of the loop */
		    foreopt_attr_join(&lc->after_end, &opt->f);
		} else {
		    /* ...or to just INSIDE the end of the loop */
		    foreopt_attr_join(&lc->at_end, &opt->f);
		}
		final = 1;
	    }
	    break;
	default:
	    errlog("DOWNOPT_STMT: Unknown Stmt_Kind: %d\n", stmt->kind);
	    break;
	}
	/* this stmt block terminates in a nonlocal goto */
	if (final)
	    return 0;
	stmt = stmt->next;
    }
    /* this stmt block falls off the end */
    return 1;
}

static inline void
foreopt_expr_stack_initialize(struct Forw_Expr_Stack *estk, struct Forw_Expr_Stack *parent)
{
    estk->parent = estk->saved_parent = parent;
    estk->is_f_initialized = 0;
}

static inline struct Forw_Expr_Stack *
foreopt_expr_stack_set_typemask(struct Forw_Expr_Stack *estk, unsigned typemask, int passes_inheritance)
{
    unsigned parent_typemask;

    if (!estk)
	return NULL;

    if (passes_inheritance)
	estk->parent = estk->saved_parent;
    else
	estk->parent = NULL;

    parent_typemask = estk->parent ? estk->parent->typemask : TYPEMASK_ANY;
    estk->typemask = typemask & parent_typemask;
    estk->bits_clrd = parent_typemask & ~typemask;
    return estk;
}

#undef F
#undef FI
#define F(T)	foreopt_expr_stack_set_typemask(&estk, (T), 0)
#define FI(T)	foreopt_expr_stack_set_typemask(&estk, (T), 1)

static inline void
foreopt_expr_stack_finish(struct Forw_Attr *f, struct Forw_Expr_Stack *estk)
{
    if (estk->is_f_initialized)
	foreopt_attr_intersect_put(f, &estk->f);
}

/*
 * For function which are evaluating single expressions with some type
 * information for the expression to inherit.
 */
static void
foreopt_expr_typemask(struct Optimizer *opt, Expr *expr, unsigned typemask)
{
    struct Forw_Expr_Stack estk;

    foreopt_expr_stack_initialize(&estk, NULL);
    foreopt_expr_stack_set_typemask(&estk, typemask, 0);
    foreopt_expr(opt, expr, &estk);
    foreopt_expr_stack_finish(&opt->f, &estk);
}

static void
foreopt_expr_prop(struct Optimizer *opt, Expr *expr, struct Forw_Expr_Stack *estk)
{
    Expr *lhs = expr->e.bin.lhs;
    Expr *rhs = expr->e.bin.rhs;

    foreopt_expr(opt, lhs,
		foreopt_expr_stack_set_typemask(estk, TYPEMASK_HASPROPS, 0));
    foreopt_expr(opt, rhs, 
		foreopt_expr_stack_set_typemask(estk, TYPEMASK(TYPE_STR), 0));

    /* not just any HASPROPS will do -- only OBJ */
    if (lhs->a.typemask == TYPEMASK(TYPE_OBJ) && rhs->kind == EXPR_VAR &&
	    rhs->e.var.type == TYPE_STR) {
	const char *pname = rhs->e.var.v.str;
	int len = memo_strlen(pname);
	int i;

	for (i = 0; bi_prop_types[i].name; ++i) {
	    if (len == bi_prop_types[i].namelen && !mystrcasecmp(bi_prop_types[i].name, pname)) {
		expr->a.typemask = bi_prop_types[i].typemask;
		break;
	    }
	}
    }
}

static void
foreopt_lvalue_before(struct Optimizer *opt, Expr * expr, struct Forw_Expr_Stack *parent, int indexed_above)
{
    struct Forw_Expr_Stack estk;
    unsigned parent_typemask = parent ? parent->typemask : TYPEMASK_ANY;

    foreopt_expr_stack_initialize(&estk, parent);

    switch (expr->kind) {
    case EXPR_RANGE: {
	Expr *saved_index_context = opt->index_context;
	/*
	 * Inherit type here because our parent is an assignment which
	 * may have inherited a type, and a the result of a range operation
	 * is always the same as the type of the range.
	 */
	foreopt_lvalue_before(opt, expr->e.range.base, FI(TYPEMASK_INDEXABLE), 1);
	opt->index_context = expr;
	foreopt_expr(opt, expr->e.range.from, F(TYPEMASK(TYPE_INT)));
	foreopt_expr(opt, expr->e.range.to, F(TYPEMASK(TYPE_INT)));
	opt->index_context = saved_index_context;
	expr->a.typemask = expr->e.range.base->a.typemask & TYPEMASK_INDEXABLE;
	break;
	}
    case EXPR_INDEX: {/* XXX UNIFY indexed_above with normal expr :( */
	Expr *lhs = expr->e.bin.lhs;
	Expr *saved_index_context = opt->index_context;

	expr->a.direct_var_rd = 0; /* XXX come back and pick up later */

	if (indexed_above)
	    foreopt_lvalue_before(opt, lhs, F(TYPEMASK(TYPE_LIST)), 1);
	else
	    foreopt_lvalue_before(opt, lhs, F(TYPEMASK_INDEXABLE), 1);
	opt->index_context = expr;
	foreopt_expr(opt, expr->e.bin.rhs, F(TYPEMASK(TYPE_INT)));
	opt->index_context = saved_index_context;
	/*
	 * If we're not indexed above, we don't have a proper type.
	 * In that case we pushed 2 values, the original list and the
	 * index to modify and the _after case will indexset with those
	 * args.
	 */
	if (indexed_above) {
	    if (expr->e.bin.lhs->a.typemask == TYPEMASK(TYPE_STR))
		expr->a.typemask = expr->e.bin.lhs->a.typemask;
	}
	break;
	}
    case EXPR_ID:		/* maybe PUSH */
	/*
	 * Note the type of the data flowing IN (typemask)
	 * which can be used if indexed_above to elide typechecks.
	 */
	if (indexed_above) {
	    expr->a.typemask = foreopt_attr_get_typemask(&opt->f, expr->e.id);
	    foreopt_attr_inherit_typemask(parent, expr->e.id);
	    if (foreopt_attr_test_definitely_put(&opt->f, expr->e.id))
		expr->a.guaranteed = 1;
	    if (expr->a.last_use && (expr->a.typemask & TYPEMASK_COMPLEX)) {
		/* technically we don't know the type anymore */
		foreopt_attr_do_last_use(&opt->f, expr->e.id);
	    }
	}
	break;
    case EXPR_PROP:
	/* XXX can this be right? */
	foreopt_expr_prop(opt, expr, &estk);
	break;
    default:
	errlog("DOWNOPT_LVALUE_BEFORE: Unknown Expr_Kind: %d\n", expr->kind);
    }
    foreopt_expr_stack_finish(&opt->f, &estk);
}

static void
foreopt_lvalue_after(struct Optimizer *opt, Expr * expr, Expr * rhs)
{
    int is_indexed = 0;

    while (1) {
	switch (expr->kind) {
	case EXPR_RANGE:
	    expr = expr->e.range.base;
	    is_indexed = 1;
	    continue;
	case EXPR_INDEX:
	    expr = expr->e.bin.lhs;
	    is_indexed = 1;
	    continue;
	case EXPR_ID:{		/* PUT */
		/*
		 * If is_indexed (stringorlist[i] = rhs)
		 * then the type of this slot *does not change*
		 * over the whole assignment expression.  Even
		 * if you have x[2] = (x = 7), the x=7 happens
		 * after the original value of x was pushed on
		 * the stack.  The 7 is indexset in that list
		 * and it gets put back into x.  Since the
		 * type evaluation of the rhs may have changed
		 * x's type information, we get it back where
		 * it was saved.
		 *
		 * However, we do know that if the rhs is not a
		 * string then the lhs must be a list.
		 *
		 * Otherwise it's a plain assignment, and the
		 * attributes are passed from the rhs.
		 *
		 * Capture typemask_put here -- the type the slot
		 * will have right before it is overwritten.
		 */
		expr->a.typemask_put = foreopt_attr_get_typemask(&opt->f, expr->e.id);
		if (is_indexed) {
		    unsigned mask = expr->a.typemask;
		    if ((rhs->a.typemask & TYPEMASK(TYPE_STR)) == 0)
			mask &= TYPEMASK(TYPE_LIST);
		    foreopt_attr_do_put(&opt->f, expr->e.id, mask, 1);
		} else
		    foreopt_attr_do_put(&opt->f, expr->e.id, rhs->a.typemask, 1);
		/*
		 * Note that the SERIAL is the NEW serial
		 */
		expr->a.value_serial = foreopt_attr_get_value_serial(&opt->f, expr->e.id);
		break;
	    }
	case EXPR_PROP:
	    break;
	default:
	    errlog("DOWNOPT_LVALUE_AFTER: Unknown Expr_Kind: %d\n", expr->kind);
	}
	break;
    }
}

/*
 * The expression may or may not happen, so the result afterwards is
 * the union of the happen|not-happen cases.  Note that we cannot inherit
 * type information in the subexpression because the inheritance would
 * do an end-run around our attribute joining (the promises of higher
 * layers won't be applied until they complete, and we have no way to
 * say might-happen in the inheritance code).
 */
static void
foreopt_expr_might_not_happen(struct Optimizer *opt, Expr * expr)
{
    struct Forw_Attr saved;

    foreopt_attr_copy(&saved, &opt->f);
    foreopt_expr(opt, expr, NULL);
    foreopt_attr_join(&opt->f, &saved);
}


static void
foreopt_expr(struct Optimizer *opt, Expr * expr, struct Forw_Expr_Stack *parent)
{
    struct Forw_Expr_Stack estk;
    unsigned parent_typemask = parent ? parent->typemask : TYPEMASK_ANY;
    Expr *direct_var;
    unsigned direct_var_saved_value_serial;

    /*
     * If the first subexpression of this node is an EXPR_ID we might
     * be able to operate on it directly (never pushing it onto the
     * stack).  This only works if the subsequent expressions don't
     * modify it.  Remember the initial value_serial so we can detect
     * any changes.
     * XXX include the ASGN case!
     */
    switch (expr->kind) {
    case EXPR_INDEX:
    case EXPR_PLUS:
    case EXPR_MINUS:
    case EXPR_TIMES:
    case EXPR_DIVIDE:
    case EXPR_MOD:
    case EXPR_EQ:
    case EXPR_NE:
    case EXPR_LT:
    case EXPR_GT:
    case EXPR_LE:
    case EXPR_GE:
    case EXPR_IN:
    case EXPR_AND:
    case EXPR_OR:
    case EXPR_EXP:
	direct_var = expr->e.bin.lhs;
	break;

    case EXPR_COND:
	direct_var = expr->e.cond.condition;
	break;

    case EXPR_RANGE:
	direct_var = expr->e.range.base;
	break;

    case EXPR_NEGATE:
    case EXPR_NOT:
	direct_var = expr->e.expr;
	break;

    default:
	direct_var = NULL;
	break;
    }
    if (direct_var && direct_var->kind == EXPR_ID) {
	direct_var_saved_value_serial =
		    foreopt_attr_get_value_serial(&opt->f, direct_var->e.id);
	direct_var->a.direct_var_rd = 1;
    }

    foreopt_expr_stack_initialize(&estk, parent);
    switch (expr->kind) {
    case EXPR_PROP:
	foreopt_expr_prop(opt, expr, &estk);
	break;

    case EXPR_VERB:
	foreopt_expr(opt, expr->e.verb.obj, F(TYPEMASK_HASVERBS));
	foreopt_expr(opt, expr->e.verb.verb, F(TYPEMASK(TYPE_STR)));
	foreopt_arglist(opt, expr->e.verb.args, NULL, 0, NULL);
	/* actual verbcall here */
	break;

    case EXPR_INDEX: {
	Expr *lhs = expr->e.bin.lhs;
	Expr *saved_index_context = opt->index_context;

	/*
	 * If this is var[] then we might be able to get it directly out
	 * of the var without ever pushing the var.  However, due XXX
	 * to EXPR_LENGTH's code generation currently relying on a stack
	 * based list, we don't do this if we see [$].
	 */
	if (lhs->kind == EXPR_ID)
	    expr->a.direct_var_rd = 1;
	/*
	 * If we know that the result of this indexing is NOT a string
	 * then we know that the lhs IS a list.  If the result is a
	 * string it could be either ("abc"[1] or {"a","b","c"}[1])
	 */
	if ((parent_typemask & TYPEMASK(TYPE_STR)) == 0)
	    foreopt_expr(opt, lhs, F(TYPEMASK(TYPE_LIST)));
	else
	    foreopt_expr(opt, lhs, F(TYPEMASK_INDEXABLE));
	opt->index_context = expr;
	foreopt_expr(opt, expr->e.bin.rhs, F(TYPEMASK(TYPE_INT)));
	opt->index_context = saved_index_context;
	/* the value of base[idx] is a string if base is a string */
	if (lhs->a.typemask == TYPEMASK(TYPE_STR))
	    expr->a.typemask = lhs->a.typemask;
	break;
    }

    case EXPR_RANGE: {
	Expr *saved_index_context = opt->index_context;

	foreopt_expr(opt, expr->e.range.base, FI(TYPEMASK_INDEXABLE));
	opt->index_context = expr;
	foreopt_expr(opt, expr->e.range.from, F(TYPEMASK(TYPE_INT)));
	foreopt_expr(opt, expr->e.range.to, F(TYPEMASK(TYPE_INT)));
	opt->index_context = saved_index_context;
	/* value of base[ran..ge] is value of base. */
	expr->a.typemask = expr->e.range.base->a.typemask & TYPEMASK_INDEXABLE;
	break;
	}

	/* left-associative binary operators */
	{ unsigned mask;
    case EXPR_PLUS:
	mask = TYPEMASK_ADDABLE;
	goto ahead1;
    case EXPR_MINUS:
    case EXPR_TIMES:
    case EXPR_DIVIDE:
    case EXPR_MOD:
	mask = TYPEMASK_MATHABLE;
	ahead1:
	/*
	 * You'd think it was pointless to use the synthesized typemask
	 * of the rhs here, before we even evaluate it.  But we might have
	 * snuck some type information in (notably for constants) in the
	 * backopt phase which already happened.
	 */
	foreopt_expr(opt, expr->e.bin.lhs, FI(expr->e.bin.rhs->a.typemask & mask));
	/* both sides must be the same */
	foreopt_expr(opt, expr->e.bin.rhs, FI(expr->e.bin.lhs->a.typemask & mask));
	/* these arithmetic (and one string) operator an only combine like
	 * types, so the result is the subset of the possible types on each
	 * side.  If that's not true an error will be raised.
	 */
	expr->a.typemask = expr->e.bin.lhs->a.typemask &
			    expr->e.bin.rhs->a.typemask;
	break;
	}

    case EXPR_EQ:
    case EXPR_NE:
	foreopt_expr(opt, expr->e.bin.lhs, NULL);
	foreopt_expr(opt, expr->e.bin.rhs, NULL);
	/* these comparison operators all return int */
	expr->a.typemask = TYPEMASK(TYPE_INT);
	break;

    case EXPR_LT:
    case EXPR_GT:
    case EXPR_LE:
    case EXPR_GE:
	foreopt_expr(opt, expr->e.bin.lhs, F(TYPEMASK_GTLTABLE &
						expr->e.bin.rhs->a.typemask));
	foreopt_expr(opt, expr->e.bin.rhs, F(TYPEMASK_GTLTABLE &
						expr->e.bin.lhs->a.typemask));
	/* these comparison operators all return int */
	expr->a.typemask = TYPEMASK(TYPE_INT);
	break;

    case EXPR_IN:
	foreopt_expr(opt, expr->e.bin.lhs, NULL);
	foreopt_expr(opt, expr->e.bin.rhs, F(TYPEMASK(TYPE_LIST)));
	expr->a.typemask = TYPEMASK(TYPE_INT);
	break;

    case EXPR_AND:
    case EXPR_OR:
	foreopt_expr(opt, expr->e.bin.lhs, NULL);
	foreopt_expr_might_not_happen(opt, expr->e.bin.rhs);
	/* one or the other could happen, so the result is the union */
	expr->a.typemask = expr->e.bin.lhs->a.typemask |
			    expr->e.bin.rhs->a.typemask;
	break;

	/* right-associative binary operators */
    case EXPR_EXP:
	/* this could really be more specific because the rules about
	 * how the types mix is complicated, but it's not a common op
	 */
	foreopt_expr(opt, expr->e.bin.lhs, FI(TYPEMASK_MATHABLE));
	foreopt_expr(opt, expr->e.bin.rhs, F(TYPEMASK_MATHABLE));
	/* this op takes a mix of floats and ints, but the result
	 * is always of the type of the lhs.
	 */
	expr->a.typemask = expr->e.bin.lhs->a.typemask & TYPEMASK_MATHABLE;
	break;

    case EXPR_COND:{
	    struct Forw_Attr root, agg;
	    foreopt_expr(opt, expr->e.cond.condition, NULL);
	    foreopt_attr_copy(&root, &opt->f);
	    /*
	     * It is tempting here to pass FI(TYPEMASK_ANY) because
	     * one side or the other of this cond should inherit the
	     * types the expression does.  However, the way that the
	     * inheritance information is carried forward will pass
	     * AROUND our attribute copying to a higher level, where
	     * it will appear as if these both happened in sequence,
	     * which is never the case.
	     */
	    foreopt_expr(opt, expr->e.cond.consequent, NULL);
	    foreopt_attr_copy(&agg, &opt->f);
	    foreopt_attr_copy(&opt->f, &root);
	    foreopt_expr(opt, expr->e.cond.alternate, NULL);
	    foreopt_attr_join(&agg, &opt->f);
	    foreopt_attr_copy(&opt->f, &agg);

	    /* one or the other could happen, so the result is the union */
	    expr->a.typemask = expr->e.cond.consequent->a.typemask |
				expr->e.cond.alternate->a.typemask;
	    break;
	}

    case EXPR_NEGATE:
	foreopt_expr(opt, expr->e.expr, FI(TYPEMASK_MATHABLE));
	expr->a.typemask = expr->e.expr->a.typemask & TYPEMASK_MATHABLE;
	break;

    case EXPR_NOT:
	foreopt_expr(opt, expr->e.expr, NULL);
	/* logical not => bool */
	expr->a.typemask = TYPEMASK(TYPE_INT);
	break;

    case EXPR_VAR:
	/* literal expr->e.var */
	expr->a.typemask = TYPEMASK(expr->e.var.type);
	break;

    case EXPR_ASGN:{
	    Expr *e = expr->e.bin.lhs;

	    if (e->kind == EXPR_SCATTER) {
		/*
		 * Yes, the rhs really does inherit our type, since the
		 * type of the ASGN is type type of the rhs.  If you do
		 * 1+({a,b,c}=l) it's going to know that l can't be any
		 * type and live.
		 */
		foreopt_expr(opt, expr->e.bin.rhs, FI(TYPEMASK(TYPE_LIST)));
		foreopt_scatter(opt, e->e.scatter);
	    } else {
		foreopt_lvalue_before(opt, expr->e.bin.lhs, NULL, 0);
		foreopt_expr(opt, expr->e.bin.rhs, FI(TYPEMASK_ANY));
		foreopt_lvalue_after(opt, expr->e.bin.lhs, expr->e.bin.rhs);
	    }
	    /* value of an assignment is the value of the rval */
	    expr->a.typemask = expr->e.bin.rhs->a.typemask;
	    break;
	}

    /* XXX call raise() or (hehe) shutdown() final = 1 */
    case EXPR_CALL: {
	int nargs = 0;
	var_type *prototype;
	var_type rt;
	static var_type bf_length_prototype[] = { TYPE_LISTORSTR };

	/*
	 * Infer types from the prototype of the builtin function by
	 * passing it down to foreopt_arglist, which can pass each
	 * individual one down to the subexpressions.
	 */
	if (expr->e.call.func == bf_length_index) {
	    nargs = 1;
	    prototype = bf_length_prototype;
	    rt = TYPE_INT;
	} else
	    nargs = info_func_by_num(expr->e.call.func, &prototype, &rt);
	foreopt_arglist(opt, expr->e.call.args, &estk, nargs, prototype);
	/* actual func call here */
	expr->a.typemask = prototype_to_typemask(rt);
	break;
	}

    case EXPR_ID:		/* PUSH */
	expr->a.typemask = foreopt_attr_get_typemask(&opt->f, expr->e.id);
	expr->a.value_serial = foreopt_attr_get_value_serial(&opt->f, expr->e.id);
	if (foreopt_attr_test_definitely_put(&opt->f, expr->e.id))
	    expr->a.guaranteed = 1;
	if (expr->a.last_use && (expr->a.typemask & TYPEMASK_COMPLEX))
	    foreopt_attr_do_last_use(&opt->f, expr->e.id);
	else
	    foreopt_attr_inherit_typemask(parent, expr->e.id);
	break;

    case EXPR_LIST:
	foreopt_arglist(opt, expr->e.list, NULL, 0, NULL);
	/* if it succeeds, it will build a list */
	expr->a.typemask = TYPEMASK(TYPE_LIST);
	break;

    case EXPR_CATCH:
	++opt->try_depth;
	foreopt_expr_might_not_happen(opt, expr->e.catch.try);
	--opt->try_depth;
	if (expr->e.catch.codes)
	    foreopt_arglist(opt, expr->e.catch.codes, NULL, 0, NULL);
	else
	    /* ANY */ ;
	if (expr->e.catch.except)
	    foreopt_expr_might_not_happen(opt, expr->e.catch.except);
	/*
	 * ideally it will have the value of the original expr, but
	 * it might have the value of the except (and no except returns
	 * the error itself, which in this brave new world can be of any
	 * type).  if the verb author has any sense, they're
	 * the same anyway!
	 */
	expr->a.typemask = expr->e.catch.try->a.typemask |
	    (expr->e.catch.except ?
	     expr->e.catch.except->a.typemask : TYPEMASK_ANY);
	break;

    case EXPR_LENGTH:
	if (! opt->index_context)
	    panic("no index_context for $ in foreopt_expr");
	opt->index_context->a.direct_var_rd = 0;
	expr->a.typemask = TYPEMASK(TYPE_INT);
	break;

    default:
	errlog("DOWNOPT_EXPR: Unknown Expr_Kind: %d\n", expr->kind);
	break;
    }

    /*
     * If this was an expression that could operate directly on the
     * lhs (if it is a var) check to see that its value didn't change
     * while evaluating arms to the right of the EXPR_ID.
     */
    if (direct_var && direct_var_saved_value_serial !=
		    foreopt_attr_get_value_serial(&opt->f, direct_var->e.id)) {
	direct_var->a.direct_var_rd = 0;
    }

    /*
     * Determine whether this expression has a constant value.
     */
    switch (expr->kind) {
    case EXPR_VAR:
	expr->a.constant = 1;	/* the origin of all constantness! */
	break;

    case EXPR_ID:		/* could know from varattr XXX */
    case EXPR_PROP:
    case EXPR_VERB:
    case EXPR_CALL:		/* some could be if the args were constant */
	expr->a.constant = 0;
	break;

    case EXPR_RANGE:
	expr->a.constant = expr->e.range.base->a.constant &&
				expr->e.range.from->a.constant &&
				expr->e.range.to->a.constant;
	break;

    case EXPR_INDEX:
    case EXPR_PLUS:
    case EXPR_MINUS:
    case EXPR_TIMES:
    case EXPR_DIVIDE:
    case EXPR_MOD:
    case EXPR_EQ:
    case EXPR_NE:
    case EXPR_LT:
    case EXPR_GT:
    case EXPR_LE:
    case EXPR_GE:
    case EXPR_IN:
    case EXPR_AND:
    case EXPR_OR:
    case EXPR_EXP:
	expr->a.constant = expr->e.bin.lhs->a.constant &&
				expr->e.bin.rhs->a.constant;
	break;

    case EXPR_COND:
	/* really the answer is based on the TRUTH of the lhs if it's
	 * constant, so when we compute that, use it here XXX
	 */
	if (expr->e.cond.condition->a.known_true)
	    expr->a.constant = expr->e.cond.consequent->a.constant;
	else
	    expr->a.constant = expr->e.cond.condition->a.constant &&
				    expr->e.cond.consequent->a.constant &&
				    expr->e.cond.alternate->a.constant;
	break;

    case EXPR_NEGATE:
    case EXPR_NOT:
	expr->a.constant = expr->e.expr->a.constant;
	break;

    case EXPR_ASGN:
	expr->a.constant = expr->e.bin.rhs->a.constant;
	break;

    case EXPR_LIST: {
	Arg_List *args = expr->e.list;

	expr->a.constant = 1;
	for (args = expr->e.list; args; args = args->next) {
	    if (!args->expr->a.constant) {
		expr->a.constant = 0;
		break;
	    }
	}
	break;
	}

    case EXPR_CATCH:
	/* even if the try expr is constant, it might be an error */
	/* ...but if it is an error, it's always the same error */
	/* so if the try expr is constant, AND the set of errors
	 * being handled is constant, AND the except expr is constant,
	 * then so are we.  but honestly, when will that ever happen?
	 */
	expr->a.constant = 0;
	break;

    case EXPR_LENGTH:
	if (! opt->index_context)
	    panic("no index_context for $ in foreopt_expr");
	switch (opt->index_context->kind) {
	case EXPR_RANGE:
	    expr->a.constant = opt->index_context->e.range.base->a.constant;
	    break;
	case EXPR_INDEX:
	    expr->a.constant = opt->index_context->e.bin.lhs->a.constant;
	    break;
	default:
	    panic("unknown index_context->kind");
	}
	break;

    default:
	errlog("DOWNOPT_EXPR: Unknown Expr_Kind: %d\n", expr->kind);
	break;
    }


    /*
     * Determine whether this expression is known to be true.  This isn't
     * exhaustive, it's just what I found quick and easy.  It's mainly for
     * while (1) (and known_false could be done for if (0))
     *
     * Due to the fact that non-constant lists can be known_true this switch
     * happens even for nonconstant expressions.
     */
    switch (expr->kind) {
    case EXPR_VAR:
	expr->a.known_true = is_true(expr->e.var);
	break;

    case EXPR_RANGE:
    case EXPR_INDEX:	/* hmm string[any] is true or error... */
    case EXPR_PLUS:
    case EXPR_MINUS:
    case EXPR_TIMES:	/* 2^31 * 2 :( */
    case EXPR_EXP:	/* 2^32 */
    case EXPR_DIVIDE:
    case EXPR_MOD:
    case EXPR_EQ:
    case EXPR_NE:
    case EXPR_LT:
    case EXPR_GT:
    case EXPR_LE:
    case EXPR_GE:
    case EXPR_IN:
    case EXPR_NOT:
    case EXPR_LENGTH:
	/* could detect by knowing values */
	expr->a.known_true = 0;
	break;

    case EXPR_ID:
    case EXPR_CALL:
    case EXPR_VERB:
    case EXPR_PROP:
	expr->a.known_true = 0;
	break;

    case EXPR_AND:
	expr->a.known_true = expr->e.bin.lhs->a.known_true &&
				expr->e.bin.lhs->a.known_true;
	break;

    case EXPR_OR:
	expr->a.known_true = expr->e.bin.lhs->a.known_true ||
				expr->e.bin.lhs->a.known_true;
	break;

    case EXPR_COND:
	if (expr->e.cond.condition->a.known_true)
	    expr->a.known_true = expr->e.cond.consequent->a.known_true;
	else
	    expr->a.known_true = expr->e.cond.alternate->a.known_true &&
				expr->e.cond.consequent->a.known_true;
	break;

    case EXPR_NEGATE:
	expr->a.known_true = expr->e.expr->a.known_true;
	break;

    case EXPR_ASGN:
	expr->a.known_true = expr->e.bin.rhs->a.known_true;
	break;

    case EXPR_LIST: {
	Arg_List *args;

	expr->a.known_true = 0;
	for (args = expr->e.list; args; args = args->next) {
	    if (args->kind != ARG_SPLICE || args->expr->a.known_true) {
		expr->a.known_true = 1;
		break;
	    }
	}
	break;
	}

    case EXPR_CATCH:
	/* never constant from above, ignore for now */
	break;

    default:
	errlog("DOWNOPT_EXPR: Unknown Expr_Kind: %d\n", expr->kind);
	break;
    }

    if ((parent_typemask & expr->a.typemask) == 0) {
	//XXX fprintf(stderr, "found impossible type conflict: %04x & %04x\n", parent_typemask, expr->a.typemask);
	expr->a.typemask = TYPEMASK(TYPE_FINALLY); /* XXX */
    }
    foreopt_expr_stack_finish(&opt->f, &estk);
}

static void
foreopt_arglist(struct Optimizer *opt, Arg_List * args, struct Forw_Expr_Stack *estk, int nargs, var_type *prototype)
{
    int i = 0;
    while (args) {
	unsigned mask;

	if (i < nargs)
	    mask = prototype_to_typemask(prototype[i]);
	else
	    mask = TYPEMASK_ANY;
	if (args->kind == ARG_SPLICE) {
	    /*
	     * The slice gets the rest of the args.  If this were
	     * to drill back down to this function (only in the
	     * case of literal @{list}) then we could continue
	     * inheriting types.  But since that's not likely, and
	     * a pain to do, skip it.
	     *
	     * And since we don't know how long this list is, we
	     * don't know how many prototype elements it should consume
	     * so we quit doing the prototypes after this.
	     *
	     * Use foreopt_expr_typemask rather than F() because the
	     * promise of listness comes true NOW (so we need a new
	     * estk), not when the builtin whose prototype we are
	     * schlepping for completes.
	     */
	    i = nargs + 1;
	    foreopt_expr_typemask(opt, args->expr, TYPEMASK(TYPE_LIST));
	} else {
	    foreopt_expr(opt, args->expr,
			foreopt_expr_stack_set_typemask(estk, mask, 0));
	}
	args = args->next;
	++i;
    }
}

static void
foreopt_scatter(struct Optimizer *opt, Scatter *root)
{
    Scatter *sc;

    for (sc = root; sc; sc = sc->next) {
	switch (sc->kind) {
	case SCAT_REST:
	    /* @var */
	    foreopt_attr_do_put(&opt->f, sc->id, TYPEMASK(TYPE_LIST), 1);
	    break;
	case SCAT_REQUIRED:
	    /* var_names[sc->id] */
	    foreopt_attr_do_put(&opt->f, sc->id, TYPEMASK_ANY, 1);
	    break;
	case SCAT_OPTIONAL:
	    /* defaults happen after the main assignment pass, so
	     * at this point it may be assigned, type unknown
	     */
	    foreopt_attr_do_put(&opt->f, sc->id, TYPEMASK_ANY, 0);
	    break;
	}
    }

    for (sc = root; sc; sc = sc->next) {
	if (sc->kind == SCAT_OPTIONAL && sc->expr) { /* with default */
	    foreopt_expr(opt, sc->expr, NULL);
	    /* it's definitely assigned (either by the scatter or
	     * now, in the default), but if it was by the scatter
	     * we don't know the type, so sc->expr->a.typemask
	     * doesn't help us.
	     */
	    foreopt_attr_do_put(&opt->f, sc->id, TYPEMASK_ANY, 1);
	}
    }
}

 /*    ______  ____    _    ____ _  ____        ___    ____  ____     ______
  *   / / / / | __ )  / \  / ___| |/ /\ \      / / \  |  _ \|  _ \   / / / /
  *  / / / /  |  _ \ / _ \| |   | ' /  \ \ /\ / / _ \ | |_) | | | | / / / / 
  *  \ \ \ \  | |_) / ___ \ |___| . \   \ V  V / ___ \|  _ <| |_| | \ \ \ \ 
  *   \_\_\_\ |____/_/   \_\____|_|\_\   \_/\_/_/   \_\_| \_\____/   \_\_\_\
  */

/*
 * A bit in the putmap can be set if it's outside any error handling
 * context (since it must succeed or cause the verb to abort) OR even
 * INSIDE an error handling context as long as nothing in the future
 * (earlier in our backwards analysis) PUSHs that variable again.
 * Unfortunately the pushmap only answers that question for instructions
 * topologically later -- in a loop we could jump backwards into (as yet
 * unseen in a backwards pass) code that would refer to the variable.
 *
 * if not catching
 *   set anything
 * else
 *   if looping
 *     set nothing
 *   else
 *     set anything that is never used again
 */
#define can_set_putmap_mask(OPT) \
	(((OPT)->try_depth == 0) ? TYPEMASK_ANY : \
		((OPT)->loops ? 0 : ~((OPT)->pushmap | (OPT)->pushmap_global)))

#define set_putmap(OPT, ID) \
	((ID < 32) ? ((OPT)->putmap |= (1 << ID) & can_set_putmap_mask(OPT)) \
		   : 0)
#define test_putmap(OPT, ID) \
	((ID < 32) ? ((OPT)->putmap & (1 << ID)) : 0)
#define reset_putmap(OPT, ID) \
	((ID < 32) ? ((OPT)->putmap &= ~(1 << ID)) : 0)

#define set_putmap_all(OPT)	((OPT)->putmap = can_set_putmap_mask(OPT))
#define reset_putmap_all(OPT)	((OPT)->putmap = 0)
#define reset_putmap_vr_vars(OPT) ((OPT)->putmap &= ~VR_VAR_MASK)

#define can_set_pushmap(OPT)	(1)
#define set_pushmap(OPT, ID) \
	((ID < 32 && can_set_pushmap(OPT)) ? ((OPT)->pushmap |= (1 << ID)) : 0)
#define reset_pushmap_all(OPT)		((OPT)->pushmap = 0)
#define reset_pushmap_vr_vars(OPT)	((OPT)->pushmap |= VR_VAR_MASK)

static void backopt_stmt(struct Optimizer *opt, Stmt * stmt);
static void backopt_stmt_cond(struct Optimizer *opt, struct Stmt_Cond cond);
static void backopt_stmt_catch(struct Optimizer *opt, struct Stmt_Catch catch);
static void backopt_scatter(struct Optimizer *opt, Scatter * sc);
static void backopt_arglist(struct Optimizer *opt, Arg_List * args);
static void backopt_expr(struct Optimizer *opt, Expr * expr);
static void backopt_expr_might_not_happen(struct Optimizer *opt, Expr * expr, int is_loop);

static void
backopt_stmt_might_not_happen(struct Optimizer *opt, Stmt * stmt, int is_loop)
{
    /* this might not happen, so it can use bits but not set them */
    unsigned saved_putmap = opt->putmap;
    unsigned saved_pushmap = (opt->pushmap_global |= opt->pushmap);

    if (is_loop) {
	/* don't let state pass into the loop body */
	reset_putmap_all(opt);
    }
    /* clear the pushmap to see what gets ref'd inside */
    reset_pushmap_all(opt);

    backopt_stmt(opt, stmt);

    /* anything untouched by PUSHes can pass. */
    opt->putmap = saved_putmap & ~opt->pushmap;
    /* accumulate push information for outer loops */
    opt->pushmap |= saved_pushmap;
}

static void
backopt_stmt_cond(struct Optimizer *opt, struct Stmt_Cond cond)
{
    Cond_Arm *elseifs;
    int i;
    unsigned root = opt->putmap, shrinking;

    if (cond.otherwise)
	backopt_stmt(opt, cond.otherwise);

    for (i = 0, elseifs = cond.arms; elseifs; elseifs = elseifs->next)
	++i;
    for (; i > 0; --i) {
	int j;
	for (j = 1, elseifs = cond.arms; j < i; elseifs = elseifs->next)
	    ++j;
	shrinking = opt->putmap;
	opt->putmap = root;
	backopt_stmt(opt, elseifs->stmt);
	opt->putmap &= shrinking;
	backopt_expr(opt, elseifs->condition);
    }
}

static void
backopt_stmt_list(struct Optimizer *opt, struct Stmt *stmt)
{
    struct Loop_Context lc;

    lc.root = stmt;
    lc.putmap_after_end = opt->putmap;
    lc.next = opt->loops;
    lc.id = stmt->s.range.id;
    opt->loops = &lc;
    backopt_stmt_might_not_happen(opt, stmt->s.list.body, 1);
    opt->loops = lc.next;
    /* actually this won't happen unless the loop range is positive
       set_putmap(opt, stmt->s.list.id);
     */
    backopt_expr(opt, stmt->s.list.expr);
}

static void
backopt_stmt_range(struct Optimizer *opt, struct Stmt *stmt)
{
    struct Loop_Context lc;

    lc.root = stmt;
    lc.putmap_after_end = opt->putmap;
    lc.next = opt->loops;
    lc.id = stmt->s.range.id;
    opt->loops = &lc;
    backopt_stmt_might_not_happen(opt, stmt->s.range.body, 1);
    opt->loops = lc.next;
    /* actually this won't happen unless the list is nonempty
       set_putmap(opt, stmt->s.range.id);
     */
    backopt_expr(opt, stmt->s.range.to);
    backopt_expr(opt, stmt->s.range.from);
}

static void
backopt_stmt_fork(struct Optimizer *opt, struct Stmt_Fork fork_stmt)
{
    /*
     * Forward pass will run the works on the inside.  Our job is to
     * make sure that we don't clear bits above it that the fork will
     * need.
     */
    backopt_stmt_might_not_happen(opt, fork_stmt.body, 0);
    if (fork_stmt.id >= 0)
	set_putmap(opt, fork_stmt.id);
    backopt_expr(opt, fork_stmt.time);
}

static void
backopt_stmt_catch(struct Optimizer *opt, struct Stmt_Catch catch)
{
    Except_Arm *ex;
    int i;
    unsigned root = opt->putmap, shrinking = root;

    for (i = 0, ex = catch.excepts; ex; ex = ex->next)
	++i;
    for (; i > 0; --i) {
	int j;
	for (j = 1, ex = catch.excepts; j < i; ex = ex->next)
	    ++j;
	opt->putmap = root;
	backopt_stmt(opt, ex->stmt);
	if (ex->id >= 0)
	    set_putmap(opt, ex->id);
	shrinking &= opt->putmap;

	if (ex->codes)
	    backopt_arglist(opt, ex->codes);
	else
	    /* ANY */ ;
    }
    opt->putmap &= shrinking;
    ++opt->try_depth;
    backopt_stmt(opt, catch.body);
    --opt->try_depth;
}

static void
backopt_stmt(struct Optimizer *opt, Stmt * stmt)
{
    /* XXX Matthew says, "it's like */
    /* recursive call here.
     * It had better operate in O(1) or else we'll be slow. */
    if (!stmt)
	return;
    switch (stmt->kind) {
    case STMT_BREAK:
    case STMT_CONTINUE:
    case STMT_RETURN:
	break;
    default:
	backopt_stmt(opt, stmt->next);
    }

    stmt->a.in_try = opt->try_depth > 0;
    switch (stmt->kind) {
    case STMT_COND:
	backopt_stmt_cond(opt, stmt->s.cond);
	break;
    case STMT_LIST:
	backopt_stmt_list(opt, stmt);
	break;
    case STMT_RANGE:
	backopt_stmt_range(opt, stmt);
	break;
    case STMT_FORK:
	backopt_stmt_fork(opt, stmt->s.fork);
	break;
    case STMT_EXPR:
	backopt_expr(opt, stmt->s.expr);
	break;
    case STMT_WHILE:{
	    struct Loop_Context lc;

	    lc.root = stmt;
	    lc.putmap_after_end = opt->putmap;
	    lc.next = opt->loops;
	    lc.id = stmt->s.loop.id;
	    opt->loops = &lc;
	    backopt_stmt_might_not_happen(opt, stmt->s.loop.body, 1);
	    opt->loops = lc.next;

	    /* each execution of WHILE_ID sets a variable */
	    if (stmt->s.loop.id >= 0)
		set_putmap(opt, stmt->s.loop.id);
	    backopt_expr_might_not_happen(opt, stmt->s.loop.condition, 1);
	    break;
	}
    case STMT_RETURN:
	/* at the moment a program ends, all variables can be trashed */
	set_putmap_all(opt);
	if (stmt->s.expr)
	    backopt_expr(opt, stmt->s.expr);
	break;
    case STMT_TRY_EXCEPT:
	backopt_stmt_catch(opt, stmt->s.catch);
	break;
    case STMT_TRY_FINALLY:
	backopt_stmt_might_not_happen(opt, stmt->s.finally.handler, 0);
	++opt->try_depth;
	backopt_stmt(opt, stmt->s.finally.body);
	--opt->try_depth;
	break;
    case STMT_BREAK:
	{
#if 1
	    struct Loop_Context *lc;
	    /* This jumps to the end of one of the loops we're nested
	     * in.  The loop remembered its putmap for us so we can
	     * revive it here.
	     */
	    lc = opt->loops;	/* inner loop */
	    if (stmt->s.exit >= 0) {
		for (; lc->id != stmt->s.exit; lc = lc->next)
		    /* find named loop */ ;
	    }
	    opt->putmap = lc->putmap_after_end;
#else
	    /* I was wrong about that... */
	    reset_putmap_all(opt);
#endif
	}
    case STMT_CONTINUE:
	{
	    /* This is basically the same case as entering
	     * a loop from the bottom.
	     */
	    reset_putmap_all(opt);
	    /* varslots[stmt->s.exit] */
	}
	break;
    default:
	errlog("DOWNOPT_STMT: Unknown Stmt_Kind: %d\n", stmt->kind);
	break;
    }
}

static void
backopt_name_expr(struct Optimizer *opt, Expr * expr)
{
    backopt_expr(opt, expr);
}

static void
backopt_lvalue_before(struct Optimizer *opt, Expr * expr, int indexed_above)
{
    switch (expr->kind) {
    case EXPR_RANGE:
	backopt_expr(opt, expr->e.range.to);
	backopt_expr(opt, expr->e.range.from);
	backopt_lvalue_before(opt, expr->e.range.base, 1);
	break;
    case EXPR_INDEX:
	backopt_expr(opt, expr->e.bin.rhs);
	backopt_lvalue_before(opt, expr->e.bin.lhs, 1);
	break;
    case EXPR_ID:		/* maybe PUSH */
	if (indexed_above) {
	    set_pushmap(opt, expr->e.id);
	    if (test_putmap(opt, expr->e.id)) {
		reset_putmap(opt, expr->e.id);
		expr->a.last_use = 1;
	    }
	}
	break;
    case EXPR_PROP:
	backopt_expr(opt, expr->e.bin.rhs);
	backopt_expr(opt, expr->e.bin.lhs);
	break;
    default:
	errlog("DOWNOPT_LVALUE_BEFORE: Unknown Expr_Kind: %d\n", expr->kind);
    }
}

static void
backopt_lvalue_after(struct Optimizer *opt, Expr * expr)
{
    int i;
    Expr *t;

    expr->a.in_try = opt->try_depth > 0;
    for (i = 0, t = expr; t; ++i) {
	switch (t->kind) {
	case EXPR_RANGE:
	    t = t->e.range.base;
	    break;
	case EXPR_INDEX:
	    t = t->e.bin.lhs;
	    break;
	default:
	    t = NULL;
	}
    }
    for (; i > 0; --i) {
	int j;
	for (j = 1, t = expr; j < i; ++j) {
	    switch (t->kind) {
	    case EXPR_RANGE:
		t = t->e.range.base;
		break;
	    case EXPR_INDEX:
		t = t->e.bin.lhs;
		break;
	    default:
		t = NULL;
	    }
	}
	switch (t->kind) {
	case EXPR_RANGE:
	    break;
	case EXPR_INDEX:
	    break;
	case EXPR_ID:{		/* PUT */
		/* This is where putting really happens */
		set_putmap(opt, t->e.id);
		break;
	    }
	case EXPR_PROP:
	    break;
	default:
	    panic("bad expr.type in backopt_lvalue_after");
	}
	break;
    }
}

static void
backopt_expr_might_not_happen(struct Optimizer *opt, Expr * expr, int is_loop)
{
    /* this might not happen, so it can use bits but not set them */
    unsigned saved_putmap = opt->putmap;
    unsigned saved_pushmap = (opt->pushmap_global |= opt->pushmap);

    if (is_loop) {
	/* don't let state pass into the loop body */
	reset_putmap_all(opt);
    }
    /* clear the pushmap to see what gets ref'd inside */
    reset_pushmap_all(opt);

    backopt_expr(opt, expr);

    /* anything untouched by PUSHes can pass. */
    opt->putmap = saved_putmap & ~opt->pushmap;
    /* accumulate push information for outer loops */
    opt->pushmap |= saved_pushmap;
}

static void
backopt_expr(struct Optimizer *opt, Expr * expr)
{
    expr->a.in_try = opt->try_depth > 0;
    switch (expr->kind) {
    case EXPR_PROP:
	backopt_name_expr(opt, expr->e.bin.rhs);
	backopt_expr(opt, expr->e.bin.lhs);
	break;

    case EXPR_VERB:
	reset_putmap_vr_vars(opt);
	reset_pushmap_vr_vars(opt);
	/* actual verbcall here */
	backopt_arglist(opt, expr->e.verb.args);
	backopt_name_expr(opt, expr->e.verb.verb);
	backopt_expr(opt, expr->e.verb.obj);
	break;

    case EXPR_INDEX:
	backopt_expr(opt, expr->e.bin.rhs);
	backopt_expr(opt, expr->e.bin.lhs);
	break;

    case EXPR_RANGE:
	backopt_expr(opt, expr->e.range.to);
	backopt_expr(opt, expr->e.range.from);
	backopt_expr(opt, expr->e.range.base);
	break;

	/* left-associative binary operators */
    case EXPR_PLUS:
    case EXPR_MINUS:
    case EXPR_TIMES:
    case EXPR_DIVIDE:
    case EXPR_MOD:
    case EXPR_EQ:
    case EXPR_NE:
    case EXPR_LT:
    case EXPR_GT:
    case EXPR_LE:
    case EXPR_GE:
    case EXPR_IN:
	backopt_expr(opt, expr->e.bin.rhs);
	backopt_expr(opt, expr->e.bin.lhs);
	break;

    case EXPR_AND:
    case EXPR_OR:{
	    backopt_expr_might_not_happen(opt, expr->e.bin.rhs, 0);
	    backopt_expr(opt, expr->e.bin.lhs);
	    break;
	}

	/* right-associative binary operators */
    case EXPR_EXP:
	backopt_expr(opt, expr->e.bin.rhs);
	backopt_expr(opt, expr->e.bin.lhs);
	break;

    case EXPR_COND:{
	    unsigned root = opt->putmap;
	    unsigned shrinking;

	    backopt_expr(opt, expr->e.cond.alternate);
	    shrinking = opt->putmap;
	    opt->putmap = root;
	    backopt_expr(opt, expr->e.cond.consequent);
	    opt->putmap &= shrinking;
	    backopt_expr(opt, expr->e.cond.condition);
	    break;
	}

    case EXPR_NEGATE:
	backopt_expr(opt, expr->e.expr);
	break;

    case EXPR_NOT:
	backopt_expr(opt, expr->e.expr);
	break;

    case EXPR_VAR:
	/* sneak some type information into this node to help out
	 * the forward pass when it happens later
	 */
	expr->a.typemask = TYPEMASK(expr->e.var.type);
	break;

    case EXPR_ASGN:{
	    Expr *e = expr->e.bin.lhs;

	    if (e->kind == EXPR_SCATTER) {
		backopt_scatter(opt, e->e.scatter);
		backopt_expr(opt, expr->e.bin.rhs);
	    } else {
		backopt_lvalue_after(opt, expr->e.bin.lhs);
		backopt_expr(opt, expr->e.bin.rhs);
		backopt_lvalue_before(opt, expr->e.bin.lhs, 0);
	    }
	    break;
	}

    case EXPR_CALL:
	reset_putmap_vr_vars(opt);
	reset_pushmap_vr_vars(opt);
	/* actual func call here */
	backopt_arglist(opt, expr->e.call.args);
	break;

    case EXPR_ID:		/* PUSH */
	set_pushmap(opt, expr->e.id);
	if (test_putmap(opt, expr->e.id)) {
	    reset_putmap(opt, expr->e.id);
	    expr->a.last_use = 1;
	}
	break;

    case EXPR_LIST:
	backopt_arglist(opt, expr->e.list);
	break;

    case EXPR_CATCH:
	if (expr->e.catch.except)
	    backopt_expr_might_not_happen(opt, expr->e.catch.except, 0);

	if (expr->e.catch.codes)
	    backopt_arglist(opt, expr->e.catch.codes);
	else
	    /* ANY */ ;

	++opt->try_depth;
	backopt_expr(opt, expr->e.catch.try);
	--opt->try_depth;

	break;

    case EXPR_LENGTH:
	break;

    default:
	errlog("DOWNOPT_EXPR: Unknown Expr_Kind: %d\n", expr->kind);
	break;
    }
}

static void
backopt_arglist(struct Optimizer *opt, Arg_List * args)
{
    int i;
    Arg_List *t;

    for (i = 0, t = args; t; t = t->next)
	++i;
    for (; i > 0; --i) {
	int j;
	for (j = 1, t = args; j < i; t = t->next)
	    ++j;
	/* if (t->kind == ARG_SPLICE) */
	backopt_expr(opt, t->expr);
    }
}

static void
backopt_scatter(struct Optimizer *opt, Scatter * sc)
{
    int i, count;
    Scatter *t;

    for (i = 0, t = sc; t; t = t->next)
	++i;
    count = i;
    /* the default cases happen after the main scatter */
    for (i = count; i > 0; --i) {
	int j;
	for (j = 1, t = sc; j < i; t = t->next)
	    ++j;
	if (t->kind == SCAT_OPTIONAL && t->expr) {
	    set_putmap(opt, t->id);
	    backopt_expr(opt, t->expr);
	}
    }
    for (i = count; i > 0; --i) {
	int j;
	for (j = 1, t = sc; j < i; t = t->next)
	    ++j;
	switch (t->kind) {
	case SCAT_REST:
	    /* @var */
	    /* fall thru to ... */
	case SCAT_REQUIRED:
	    /* var_names[t->id] */
	    /* scattering sets all of the non-optional variables */
	    set_putmap(opt, t->id);
	    break;
	case SCAT_OPTIONAL:
	    break;
	}
    }
}

static void
markup_stmts_internal(Stmt * top, struct Forw_Attr *f)
{
    struct Optimizer opt;
    int i;

    bf_length_index = number_func_by_name("length");
    bf_typeof_index = number_func_by_name("typeof");
    memset(&opt, 0, sizeof(opt));

    /* at the moment a program ends, all variables can be trashed */
    /* and that happens first thing, when you're going backwards... */
    set_putmap_all(&opt);
    backopt_stmt(&opt, top);
    /*
     * Arriving at the beginning of the verb:
     * The pushmap bits indicate all referenced variables.
     * The putmap bits are slots which are overwritten before referenced.
     * Therefore (putmap | ~pushmap) bits are slots that don't need to
     * be initialized because they are never used!  XXX use that
     */

    /* Cleanup for forward pass */
    opt.pushmap = opt.pushmap_global = 0;
    if (opt.loops)
	panic("leftover loops from backopt_stmt");
    if (opt.try_depth != 0)
	panic("leftover try_depth from backopt_stmt");

    /* forked tasks start with someone else's dirty laundry */
    if (f) {
	foreopt_attr_copy(&opt.f, f);
    } else {
	for (i = 0; i < MAX_VARATTR; ++i) {
	    opt.f.rt_env[i].typemask = TYPEMASK(TYPE_NONE);
	    opt.f.rt_env[i].value_serial = next_value_serial();
	}
	/* everything initialized by execute.c is pre-put */
	foreopt_attr_do_put(&opt.f, SLOT_NUM, TYPEMASK(TYPE_INT), 1);
	foreopt_attr_do_put(&opt.f, SLOT_OBJ, TYPEMASK(TYPE_INT), 1);
	foreopt_attr_do_put(&opt.f, SLOT_STR, TYPEMASK(TYPE_INT), 1);
	foreopt_attr_do_put(&opt.f, SLOT_LIST, TYPEMASK(TYPE_INT), 1);
	foreopt_attr_do_put(&opt.f, SLOT_ERR, TYPEMASK(TYPE_INT), 1);
	foreopt_attr_do_put(&opt.f, SLOT_FLOAT, TYPEMASK(TYPE_INT), 1);
	foreopt_attr_do_put(&opt.f, SLOT_INT, TYPEMASK(TYPE_INT), 1);

	foreopt_attr_do_put(&opt.f, SLOT_PLAYER, TYPEMASK(TYPE_OBJ), 1);
	foreopt_attr_do_put(&opt.f, SLOT_THIS, TYPEMASK(TYPE_OBJ), 1);/*XXX WAIF */
	foreopt_attr_do_put(&opt.f, SLOT_CALLER, TYPEMASK(TYPE_OBJ), 1);/*XXX WAIF */
	foreopt_attr_do_put(&opt.f, SLOT_VERB, TYPEMASK(TYPE_STR), 1);
	foreopt_attr_do_put(&opt.f, SLOT_ARGS, TYPEMASK(TYPE_LIST), 1);

#ifdef THESE_CAN_ACTUALLY_BE_ANYTHING_SADLY
	foreopt_attr_do_put(&opt.f, SLOT_ARGSTR, TYPEMASK(TYPE_STR), 1);
	foreopt_attr_do_put(&opt.f, SLOT_DOBJSTR, TYPEMASK(TYPE_STR), 1);
	foreopt_attr_do_put(&opt.f, SLOT_IOBJ, TYPEMASK(TYPE_OBJ), 1);
	foreopt_attr_do_put(&opt.f, SLOT_DOBJ, TYPEMASK(TYPE_OBJ), 1);
	foreopt_attr_do_put(&opt.f, SLOT_IOBJSTR, TYPEMASK(TYPE_STR), 1);
	foreopt_attr_do_put(&opt.f, SLOT_PREPSTR, TYPEMASK(TYPE_STR), 1);
#else
	foreopt_attr_do_put(&opt.f, SLOT_ARGSTR, TYPEMASK_ANY, 1);
	foreopt_attr_do_put(&opt.f, SLOT_DOBJSTR, TYPEMASK_ANY, 1);
	foreopt_attr_do_put(&opt.f, SLOT_IOBJ, TYPEMASK_ANY, 1);
	foreopt_attr_do_put(&opt.f, SLOT_DOBJ, TYPEMASK_ANY, 1);
	foreopt_attr_do_put(&opt.f, SLOT_IOBJSTR, TYPEMASK_ANY, 1);
	foreopt_attr_do_put(&opt.f, SLOT_PREPSTR, TYPEMASK_ANY, 1);
#endif
    }

    foreopt_stmt(&opt, top);

    /* XXX free_optimizer_members(&opt); */
}

void
markup_stmts(Stmt * top)
{
    markup_stmts_internal(top, NULL);
}

char rcsid_optimize[] = "$Id$";

/* 
 * $Log$
 */
