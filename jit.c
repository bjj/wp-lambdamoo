#include "config.h"
#include "options.h"
#include "eval_env.h"
#include "execute.h"
#include "opcode.h"
#include "storage.h"
#include "streams.h"
#include "structures.h"
#include "sym_table.h"
#include "utils.h"
#include "list.h"
#include "functions.h"
#include "numbers.h"
#include "my-stdio.h"
#include "my-string.h"
#include "jit.h"
#include "optimize.h"
#include <lightning.h>

#define RUN_ACTIV (activ_stack[top_activ_stack])

#define XXX 0

/*
 * Register assignments
 *
 * R0 = scratch
 * R1 = scratch
 * R2 = scratch
 * V0 = RUN_ACTIV.top_activ_stack
 * V1 = RUN_ACTIV
 * V2 = [anything that needs to survive func call]
 */

#undef _jit
#define _jit (*jit)

#define JIM_RTS		JIT_V0
#define JIM_RUN_ACTIV	JIT_V1

#define Offset(type, member) ((size_t) (& ((type *)0)->member))

#define VAR_SHIFT  (sizeof(Var) == 8 ? 3 :	\
		    sizeof(Var) == 16 ? 4 :	\
		    0)

#define PTR_SHIFT  (sizeof(void *) == 4 ? 2 :	\
		    sizeof(void *) == 8 ? 3 :	\
		    0)

#define jit_is_preserved(reg) \
	(reg == JIT_V0 || reg == JIT_V1 || reg == JIT_V2)

#define jim_free_var(varp_reg) do { jit_insn *__ref;			\
	jit_ldxi_ui(JIT_R0, varp_reg, Offset(Var, type));		\
	__ref = jit_bmci_ui(jit_forward(), JIT_R0, TYPE_COMPLEX_FLAG);	\
	if (!jit_is_preserved(varp_reg)) jit_pushr_p(varp_reg);		\
	jit_prepare(1);							\
	jit_pusharg_p(varp_reg);					\
	jit_finish(complex_free_var);					\
	if (!jit_is_preserved(varp_reg)) jit_popr_p(varp_reg);		\
	jit_patch(__ref);						\
} while (0)

/* XXX make a jim_copy_var_ref to share some insns */
#define jim_var_ref(varp_reg) do { jit_insn *__ref;			\
	jit_ldxi_ui(JIT_R0, varp_reg, Offset(Var, type));		\
	__ref = jit_bmci_ui(jit_forward(), JIT_R0, TYPE_COMPLEX_FLAG);	\
	if (!jit_is_preserved(varp_reg)) jit_pushr_p(varp_reg);		\
	jit_prepare(1);							\
	jit_pusharg_p(varp_reg);					\
	jit_finish(complex_var_ref);					\
	if (!jit_is_preserved(varp_reg)) jit_popr_p(varp_reg);		\
	jit_patch(__ref);						\
} while (0)

extern activation *activ_stack;
extern unsigned top_activ_stack;

#define jim_lea_RUN_ACTIV(reg)	(void)(			\
	jit_ldi_ui(JIT_R0, &top_activ_stack),		\
	jit_muli_ui(JIT_R0, JIT_R0, sizeof(activation)), \
	jit_ldi_p(reg, &activ_stack),			\
	jit_addr_p(reg, reg, JIT_R0))

/* assume RUN_ACTIV in JIM_RUN_ACTIV */
#define jim_lea_RT_ENV(reg) \
	jit_ldxi_p(reg, JIM_RUN_ACTIV, Offset(activation, rt_env))

#define jim_lea_RT_ENV_slot_i(reg, slot) (void)(\
	jim_lea_RT_ENV(reg),				\
	jit_addi_p(reg, reg, slot * sizeof(Var)))

#define jim_lea_RT_ENV_slot_r(reg, slot) (void)(\
	jim_lea_RT_ENV(reg),				\
	jit_lshi_i(slot, slot, VAR_SHIFT),		\
	jit_addr_p(reg, reg, slot))

#define jim_LOAD_STATE_VARIABLES() (void)( \
	jim_lea_RUN_ACTIV(JIM_RUN_ACTIV), \
	jit_ldxi_p(JIM_RTS, JIM_RUN_ACTIV, Offset(activation, top_rt_stack)))

#define jim_STORE_STATE_VARIABLES() (void)( \
	jit_stxi_p(Offset(activation, top_rt_stack), JIM_RUN_ACTIV, JIM_RTS))

/* XXX don't store back in RUN_ACTIV every time */
#define jim_PUSHn(n) jit_addi_p(JIM_RTS, JIM_RTS, sizeof(Var) * n)
#define jim_PUSH() jim_PUSHn(1)
#define jim_POPn(n)  jit_subi_p(JIM_RTS, JIM_RTS, sizeof(Var) * n)
#define jim_POP()  jim_POPn(1)

#define jim_copy_var_R0(dst, dstoffs, src, srcoffs) (void)(	\
	jit_ldxi_i(JIT_R0, src, Offset(Var, type) + srcoffs),	\
	jit_stxi_i(Offset(Var, type) + dstoffs, dst, JIT_R0),	\
	jit_ldxi_i(JIT_R0, src, Offset(Var, v) + srcoffs),	\
	jit_stxi_i(Offset(Var, v) + dstoffs, dst, JIT_R0))

#define jim_copy_var_R0R1(dst, dstoffs, src, srcoffs) (void)(	\
	jit_ldxi_i(JIT_R0, src, Offset(Var, type) + srcoffs),	\
	jit_ldxi_i(JIT_R1, src, Offset(Var, v) + srcoffs),	\
	jit_stxi_i(Offset(Var, type) + dstoffs, dst, JIT_R0),	\
	jit_stxi_i(Offset(Var, v) + dstoffs, dst, JIT_R1))

int 
is_truep_free(Var * v)
{
    int i = is_true(*v);
    free_var(*v);
    return i;
}
int 
is_truep(Var * v)
{
    return is_true(*v);
}

#define jim_branch_if_(reg, fre) (		\
	jit_prepare(1),				\
	jit_pusharg_p(reg),			\
	jit_finish(fre ? is_truep_free : is_truep),	\
	jit_retval(JIT_R0))

#define jim_branch_if_true(targ, reg, fre) (	\
	jim_branch_if_(reg, fre),		\
	jit_bnei_i(targ, JIT_R0, 0))

#define jim_branch_if_false(targ, reg, fre) (	\
	jim_branch_if_(reg, fre),		\
	jit_beqi_i(targ, JIT_R0, 0))

#define jim_SPACER_NOP() jit_nop();

jit_insn *L_raise_E_TYPE;
jit_insn *L_raise_E_RANGE;
jit_insn *L_raise_E_VARNF;
jit_insn *L_raise_error;
jit_insn *L_jump_to_pc;
jit_insn *L_jump_to_V2;

extern Bytecodes *current_bytecodes();

void
jim_glue(jit_state * jit)
{
    jit_prolog(0);
    jim_LOAD_STATE_VARIABLES();

    /* work out where to resume */
    L_jump_to_pc = jit_get_label();
    jit_ldxi_i(JIT_V2, JIM_RUN_ACTIV, Offset(activation, pc));
    L_jump_to_V2 = jit_get_label();
    jit_prepare(0);
    jit_finish(current_bytecodes);
    jit_retval(JIT_R0);
    jit_ldxi_p(JIT_R0, JIT_R0, Offset(Bytecodes, jitentries));
    jit_lshi_i(JIT_V2, JIT_V2, PTR_SHIFT);
    jit_ldxr_p(JIT_R0, JIT_V2, JIT_R0);
    jit_jmpr(JIT_R0);


    L_raise_E_TYPE = jit_get_label();
    jit_movi_i(JIT_R0, E_TYPE);
    L_raise_error = jit_get_label();
    jim_STORE_STATE_VARIABLES();
    jit_stxi_p(Offset(activation, jit_error_out), JIM_RUN_ACTIV, JIT_R0);
    jit_movi_i(JIT_RET, RUNACT_RAISE);
    jit_ret();
    L_raise_E_RANGE = jit_get_label();
    jit_movi_i(JIT_R0, E_RANGE);
    jit_jmpi(L_raise_error);
    L_raise_E_VARNF = jit_get_label();
    jit_movi_i(JIT_R0, E_VARNF);
    jit_jmpi(L_raise_error);
}

void
jim_prolog(jit_state * jit)
{
    char *start = jit_get_ip().ptr;
    /* nothing to do with new jit_run_glue! */
    /* well except jit_prolog() is required to initialize */
    jit_prolog(0);
    jit_set_ip(start);
}

void
jim_epilog(jit_state * jit, char *origin)
{
    static unsigned max;
    unsigned len = jit_get_ip().ptr - origin;
    if (len > max) {
	max = len;
	fprintf(stderr, "largest %d\n", max);
    }
    if (len > 65536) {
	fprintf(stderr, "eets too big mum %d\n", len);
	exit(1);
    }
    jit_flush_code(origin, jit_get_ip().ptr);
}

void
jim_POP_AND_FREE(jit_state * jit)
{
    jim_POP();
    jim_free_var(JIM_RTS);
    jim_SPACER_NOP();
}

void
jim_return(jit_state * jit, int value_p)
{
    jim_STORE_STATE_VARIABLES();
    jit_movi_i(JIT_RET, value_p ? RUNACT_RETURN : RUNACT_RETURN0);
    jit_ret();
    jim_SPACER_NOP();
}

static int
equalityp_free(Var * args)
{
    int i = equality(args[0], args[1], 0);
    free_var(args[0]);
    free_var(args[1]);
    return i;
}

void
jim_EQ_NE(jit_state * jit, int is_eq)
{
    jim_POPn(2);
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(equalityp_free);
    jit_retval(JIT_R0);
    if (!is_eq)
	jit_xori_i(JIT_R0, JIT_R0, 1);
    jit_stxi_i(Offset(Var, v.num), JIM_RTS, JIT_R0);
    jit_movi_i(JIT_R0, TYPE_INT);
    jit_stxi_i(Offset(Var, type), JIM_RTS, JIT_R0);
    jim_PUSH();
}

static int
comparison_helper(Var * args, Opcode op)
{
    Var ans;
    int comparison;

    if ((args[-2].type == TYPE_INT || args[-2].type == TYPE_FLOAT)
	&& (args[-1].type == TYPE_INT || args[-1].type == TYPE_FLOAT)) {
	ans = compare_numbers(args[-2], args[-1]);
	if (ans.type == TYPE_ERR) {
	    return ans.v.err;
	} else {
	    comparison = ans.v.num;
	    goto finish_comparison;
	}
    } else if (args[-1].type != args[-2].type || args[-1].type == TYPE_LIST) {
	return E_TYPE;
    } else {
	switch (args[-1].type) {
	case TYPE_INT:
	    comparison = compare_integers(args[-2].v.num, args[-1].v.num);
	    break;
	case TYPE_OBJ:
	    comparison = compare_integers(args[-2].v.obj, args[-1].v.obj);
	    break;
	case TYPE_ERR:
	    comparison = ((int) args[-2].v.err) - ((int) args[-1].v.err);
	    break;
	case TYPE_STR:
	    comparison = mystrcasecmp(args[-2].v.str, args[-1].v.str);
	    break;
	default:
	    errlog("RUN: Impossible type in comparison: %d\n",
		   args[-1].type);
	    comparison = 0;
	}

      finish_comparison:
	ans.type = TYPE_INT;
	switch (op) {
	case OP_LT:
	    ans.v.num = (comparison < 0);
	    break;
	case OP_LE:
	    ans.v.num = (comparison <= 0);
	    break;
	case OP_GT:
	    ans.v.num = (comparison > 0);
	    break;
	case OP_GE:
	    ans.v.num = (comparison >= 0);
	    break;
	default:
	    errlog("RUN: Imposible opcode in comparison: %d\n", op);
	    break;
	}
	free_var(args[-1]);
	free_var(args[-2]);
	args[-2] = ans;
	return E_NONE;
    }
}

void
jim_comparison(jit_state * jit, Opcode op, unsigned l_typemask, unsigned r_typemask)
{
    if (l_typemask == r_typemask && l_typemask == TYPEMASK(TYPE_INT)) {
	jit_insn *rel;

	jim_POP();
	jit_ldxi_i(JIT_R2, JIM_RTS, Offset(Var, v.num) - sizeof(Var));
	jit_ldxi_i(JIT_R1, JIM_RTS, Offset(Var, v.num));
	switch (op) {
	case OP_GE:
	    jit_ger_i(JIT_R0, JIT_R2, JIT_R1);
	    break;
	case OP_GT:
	    jit_gtr_i(JIT_R0, JIT_R2, JIT_R1);
	    break;
	case OP_LE:
	    jit_ler_i(JIT_R0, JIT_R2, JIT_R1);
	    break;
	case OP_LT:
	    jit_ltr_i(JIT_R0, JIT_R2, JIT_R1);
	    break;
	default:
	    panic("unknown comparator in jim_comparison");
	    rel = 0;
	}
	jit_stxi_i(Offset(Var, v.num) - sizeof(Var), JIM_RTS, JIT_R0);
    } else {
	jit_prepare(2);
	jit_movi_i(JIT_R0, op);
	jit_pusharg_i(JIT_R0);
	jit_pusharg_p(JIM_RTS);
	jit_finish(comparison_helper);
	jit_retval(JIT_R0);
	jit_bnei_i(L_raise_error, JIT_R0, E_NONE);
	jim_POP();
    }
    jim_SPACER_NOP();
}

static void 
new_listp(int sz, Var * p)
{
    *p = new_list(sz);
}


/* {elt, list}p{idx, ?} with input freed */
static void
in_helper(Var *args)
{
    int i = ismember(args[-2], args[-1], 0);
    free_var(args[-2]);
    free_var(args[-1]);
    args[-2].v.num = i;
    args[-2].type = TYPE_INT;
}

void
jim_IN(jit_state * jit)
{
    jit_insn *rel1 = 0, *rel2 = 0;

    /* peek at top looking for list */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var));
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_LIST);
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(in_helper);
    jim_POP();
}

void 
complex_free_list_v_list(Var * l)
{
    Var v;
    v.type = TYPE_LIST;
    v.v.list = l;
    complex_free_var(&v);
}

char *
refstr_helper_free(const char *s, int n)
{
    if (n < 1 || n > memo_strlen(s)) {
	return 0;
    } else {
	char *p = str_dup(" ");
	*p = s[n - 1];
	free_str(s);
	return p;
    }
}

char *
refstr_helper(const char *s, int n)
{
    if (n < 1 || n > memo_strlen(s)) {
	return 0;
    } else {
	char *p = str_dup(" ");
	*p = s[n - 1];
	return p;
    }
}

/* XXX on err in the id>0 case we could go push the list anyway to make
 * it look kosher
 */
void
jim_REF(jit_state * jit, int list_only, int keep_ref, int id, int last_use, unsigned typemask)
{
    jit_insn *rel1 = 0, *rel2 = 0;

    list_only = list_only || ((typemask & TYPEMASK(TYPE_STR)) == 0);
    if (keep_ref && id >= 0)
	panic("jim_REF can't keep_ref and work directly on id");

    /* peek at top looking for int */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var));
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_INT);
    /* R1 = index */
    jit_ldxi_i(JIT_R1, JIM_RTS, Offset(Var, v.num) - sizeof(Var));
    /* peek at list/str to select type */
    if (id >= 0) {
	jim_lea_RT_ENV_slot_i(JIT_V2, id);
	jit_ldxi_i(JIT_R0, JIT_V2, Offset(Var, type));
	/* V2 = base pointer */
	jit_ldxi_p(JIT_V2, JIT_V2, Offset(Var, v.list));
    } else {
	jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var) * 2);
	/* V2 = base pointer */
	jit_ldxi_p(JIT_V2, JIM_RTS, Offset(Var, v.list) - sizeof(Var) * 2);
    }
    if (!list_only) {
	rel1 = jit_beqi_i(jit_forward(), JIT_R0, TYPE_STR);
    }
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_LIST);

    /* list case */
    jit_blei_i(L_raise_E_RANGE, JIT_R1, 0);
    jit_ldxi_i(JIT_R2, JIT_V2, Offset(Var, v.num));
    jit_bgtr_i(L_raise_E_RANGE, JIT_R1, JIT_R2);
    jit_lshi_i(JIT_R1, JIT_R1, VAR_SHIFT);
    jit_addr_p(JIT_R2, JIT_V2, JIT_R1);		/* R2 = &elt */
    if (id >= 0)
	jim_POP();    /* there is no spoon.  err, list */
    else if (!keep_ref)
	jim_POPn(2);
    jim_copy_var_R0R1(JIM_RTS, 0, JIT_R2, 0);
    jim_var_ref(JIM_RTS);
    /* got it, now we can free src list */
    if (id >= 0 && last_use) {
	jim_lea_RT_ENV_slot_i(JIT_V2, id);
	jit_prepare(1);
	jit_pusharg_p(JIT_V2);
	jit_finish(complex_free_var);
	jit_movi_i(JIT_R0, TYPE_NONE);
	jit_stxi_i(Offset(Var, type), JIT_V2, JIT_R0);
    }
    if (!keep_ref && id < 0) {
	/* all we saved was a list.v.list, so some help please */
	jit_prepare(1);
	jit_pusharg_p(JIT_V2);
	jit_finish(complex_free_list_v_list);
    }
    jim_PUSH();

    if (!list_only) {
	rel2 = jit_jmpi(jit_forward());
	jit_patch(rel1);
	/* str case */
	/* str ptr in JIT_V2, idx in JIT_R1 */
	jit_prepare(2);
	jit_pusharg_i(JIT_R1);
	jit_pusharg_p(JIT_V2);
	jit_finish((id >= 0 && !last_use) ? refstr_helper : refstr_helper_free);
	/* => new string with old string consumed in stack case, OR
	 * NULL with old string not consumed.
	 * since we haven't popped anything yet, branching to
	 * error will clean everything.
	 */
	jit_retval(JIT_R0);
	jit_beqi_p(L_raise_E_RANGE, JIT_R0, 0);
	if (id < 0) {
	    /* ok, now we pop the index */
	    jim_POP();
	} else {
	    /* set type on the index which is about to be the str */
	    jit_movi_i(JIT_R1, TYPE_STR);
	    jit_stxi_i(Offset(Var, type) - sizeof(Var), JIM_RTS, JIT_R1);
	    if (last_use) {
		jim_lea_RT_ENV_slot_i(JIT_R2, id);
		jit_movi_i(JIT_R1, TYPE_NONE);
		jit_stxi_i(Offset(Var, type), JIT_R2, JIT_R1);
	    }
	}
	/* and smash the old str (now free'd) with new */
	jit_stxi_p(Offset(Var, v.str) - sizeof(Var), JIM_RTS, JIT_R0);

/* XXX KEEP REF? */
	jit_patch(rel2);
    }
    jim_SPACER_NOP();
}

/* XXX will need per-type versions of this */
void
jim_EOP_LENGTH(jit_state * jit, int stackoffs)
{
    jit_insn *rel1 = 0, *rel2 = 0;

    /* an odd little op that references from the base of the stack */
    jit_ldxi_p(JIT_V2, JIM_RUN_ACTIV, Offset(activation, base_rt_stack));
    /* peek at type */
    jit_ldxi_i(JIT_R0, JIT_V2, Offset(Var, type) + stackoffs * sizeof(Var));
    /* grab pointer for either case, and set type for either case */
    jit_ldxi_i(JIT_R1, JIT_V2, Offset(Var, v.list) + stackoffs * sizeof(Var));
    jit_movi_i(JIT_R2, TYPE_INT);
    jit_stxi_i(Offset(Var, type), JIM_RTS, JIT_R2);

    rel1 = jit_beqi_i(jit_forward(), JIT_R0, TYPE_STR);
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_LIST);

    /* list case */
    jit_ldxi_i(JIT_R0, JIT_R1, Offset(Var, v.num));
    rel2 = jit_jmpi(jit_forward());

    jit_patch(rel1);
    /* string case */
    /* memo_strlen */
    jit_ldxi_i(JIT_R0, JIT_R1, -2 * sizeof(int));

    jit_patch(rel2);
    jit_stxi_i(Offset(Var, v.num), JIM_RTS, JIT_R0);
    jim_PUSH();
    jim_SPACER_NOP();
}

void
jim_PUSH_int(jit_state * jit, int n, int type)
{
    /* PUSH A NUMBER WOO HOO */
    jit_movi_i(JIT_R0, n);
    jit_stxi_i(Offset(Var, v.num), JIM_RTS, JIT_R0);
    if (type != n)
	jit_movi_i(JIT_R0, type);
    jit_stxi_i(Offset(Var, type), JIM_RTS, JIT_R0);
    jim_PUSH();
    jim_SPACER_NOP();
}

void
jim_PUSH_NUM(jit_state * jit, int n)
{
    jim_PUSH_int(jit, n, TYPE_INT);
}

void
jim_PUSH_OBJ(jit_state * jit, int n)
{
    jim_PUSH_int(jit, n, TYPE_OBJ);
}

/*
 * XXX why have a table when we know the literal here?  dur
 */
void
jim_IMM(jit_state * jit, int lit)
{
    jit_ldxi_p(JIT_V2, JIM_RUN_ACTIV, Offset(activation, prog));
    jit_ldxi_p(JIT_V2, JIT_V2, Offset(Program, literals));
    jim_copy_var_R0R1(JIM_RTS, 0, JIT_V2, lit * sizeof(Var));
    jim_var_ref(JIM_RTS);
    jim_PUSH();
    jim_SPACER_NOP();
}

static void 
new_emptylistp(Var * p)
{
    *p = new_list(0);
}

void
jim_OP_MAKE_EMPTY_LIST(jit_state * jit)
{
    /* push(new_list(0)) */
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(new_emptylistp);
    jim_PUSH();
    jim_SPACER_NOP();
}

static void 
new_singletonlistp(Var * p)
{
    *p = new_list(1);
}

void
jim_OP_MAKE_SINGLETON_LIST(jit_state * jit)
{
    /* save the top of the rts */
    jim_POP();
    jit_pushr_i(JIT_V1);
    jit_ldxi_p(JIT_V1, JIM_RTS, Offset(Var, v));
    jit_ldxi_p(JIT_V2, JIM_RTS, Offset(Var, type));

    /* push(new_list(1)) */
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(new_singletonlistp);

    /* list.v.list[1] = POP() from before */

    jit_ldxi_p(JIT_R0, JIM_RTS, Offset(Var, v.list));
    jit_stxi_i(Offset(Var, v) + sizeof(Var), JIT_R0, JIT_V1);
    jit_stxi_i(Offset(Var, type) + sizeof(Var), JIT_R0, JIT_V2);
    jit_popr_i(JIT_V1);
    jim_PUSH();
    jim_SPACER_NOP();
}


static void
new_ndletonlistp(int n, Var * contents)
{
    Var t = new_list(n);
    memcpy(&t.v.list[1], contents, sizeof(Var) * n);
    *contents = t;
}

void
jim_OP_MAKE_NDLETON_LIST(jit_state * jit, int n)
{
    if (n == 1) {
	jim_OP_MAKE_SINGLETON_LIST(jit);
	return;
    }
    /* save the top of the rts */
    jim_POPn(n);
    jit_prepare(2);
    jit_pusharg_p(JIM_RTS);
    jit_movi_i(JIT_R0, n);
    jit_pusharg_i(JIT_R0);
    jit_finish(new_ndletonlistp);
    jim_PUSH();
    jim_SPACER_NOP();
}

void
jim_OP_CHECK_LIST_FOR_SPLICE(jit_state * jit)
{
    /* peek top type */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var));
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_LIST);
    /* XXX in the !debug case you'd have to replace this with err */
    /* don't even need to free it here because error handling is
     * going to unwind it off the stack for us
     */
    jim_SPACER_NOP();
}

void 
listappendp(Var * args)
{
    args[0] = listappend(args[0], args[1]);
}

void
jim_OP_LIST_ADD_TAIL(jit_state * jit)
{
#if 0
    /* in a +d verb we can't reach this point without the right type
     * being on the stack already.  -d would allow a previous expr to
     * become a TYPE_ERR, but we will raise instead.
     */
    /* peek 2nd top type */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var) * 2);
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_LIST);
    /* XXX in the !debug case you'd have to replace this with err */
    /* don't even need to free it here because error handling is
     * going to unwind it off the stack for us
     */
#endif
    jit_prepare(1);
    jim_POPn(2);
    jit_pusharg_p(JIM_RTS);
    jit_finish(listappendp);
    jim_PUSH();
    jim_SPACER_NOP();
}

void 
listconcatp(Var * args)
{
    args[0] = listconcat(args[0], args[1]);
}

void
jim_OP_LIST_APPEND(jit_state * jit)
{
    /* peek top 2 types */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var) * 2);
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_LIST);
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var) * 1);
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_LIST);
    /* XXX in the !debug case you'd have to replace this with err */
    /* don't even need to free it here because error handling is
     * going to unwind it off the stack for us
     */
    jit_prepare(1);
    jim_POPn(2);
    jit_pusharg_p(JIM_RTS);
    jit_finish(listconcatp);
    jim_PUSH();
    jim_SPACER_NOP();
}

/* {oid, str} => {result, ?} + E_NONE (found)
 *
 *            => {?, ?} + err (error)
 *
 *  either way with free_var(oid) free_var(str)
 */
enum error
get_prop_helper(Var * args)
{
    if (args[1].type != TYPE_STR || args[0].type != TYPE_OBJ) {
	free_var(args[0]);
	free_var(args[1]);
	return E_TYPE;
    } else if (!valid(args[0].v.obj)) {
	free_var(args[0]);
	free_var(args[1]);
	return E_INVIND;
    } else {
	db_prop_handle h;
	Var prop;

	h = db_find_property(args[0].v.obj, args[1].v.str, &prop);
	free_var(args[0]);
	free_var(args[1]);
	if (!h.ptr)
	    return E_PROPNF;
	else if (h.built_in
		 ? bi_prop_protected(h.built_in, RUN_ACTIV.progr)
		 : !db_property_allows(h, RUN_ACTIV.progr, PF_READ))
	    return E_PERM;
	else if (h.built_in)
	    args[0] = prop;	/* it's already freshly allocated */
	else
	    args[0] = var_ref(prop);
    }
    return E_NONE;
}

void
jim_GET_PROP(jit_state * jit)
{
    jim_POPn(2);
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(get_prop_helper);
    jit_retval(JIT_R0);
    jit_bnei_i(L_raise_error, JIT_R0, E_NONE);
    jim_PUSH();
    jim_SPACER_NOP();
}

/* {oid, str}p{?} => {oid, str}p{res} + E_NONE (found)
 *                => {oit, str}p{?} + err (error)
 */
enum error
push_get_prop_helper(Var * args)
{
    if (args[-1].type != TYPE_STR || args[-2].type != TYPE_OBJ) {
	return E_TYPE;
    } else if (!valid(args[-2].v.obj)) {
	return E_INVIND;
    } else {
	db_prop_handle h;
	Var prop;

	h = db_find_property(args[-2].v.obj, args[-1].v.str, &prop);
	if (!h.ptr)
	    return E_PROPNF;
	else if (h.built_in
		 ? bi_prop_protected(h.built_in, RUN_ACTIV.progr)
		 : !db_property_allows(h, RUN_ACTIV.progr, PF_READ))
	    return E_PERM;
	else if (h.built_in)
	    args[0] = prop;	/* it's already freshly allocated */
	else
	    args[0] = var_ref(prop);
    }
    return E_NONE;
}

void
jim_PUSH_GET_PROP(jit_state * jit)
{
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(push_get_prop_helper);
    jit_retval(JIT_R0);
    jit_bnei_i(L_raise_error, JIT_R0, E_NONE);
    jim_PUSH();
    jim_SPACER_NOP();
}

#if 0
case OP_PUT_PROP:
{
    Var obj, propname, rhs;

    rhs = POP();		/* any type */
    propname = POP();		/* should be string */
    obj = POP();		/* should be objid */
    if (obj.type != TYPE_OBJ || propname.type != TYPE_STR) {
	free_var(rhs);
	free_var(propname);
	free_var(obj);
	PUSH_ERROR(E_TYPE);
    } else if (!valid(obj.v.obj)) {
	free_var(rhs);
	free_var(propname);
	free_var(obj);
	PUSH_ERROR(E_INVIND);
    } else {
	db_prop_handle h;
	enum error err = E_NONE;
	Objid progr = RUN_ACTIV.progr;

	h = db_find_property(obj.v.obj, propname.v.str, 0);
	if (!h.ptr)
	    err = E_PROPNF;
	else {
	    switch (h.built_in) {
	    case BP_NONE:	/* Not a built-in property */
		if (!db_property_allows(h, progr, PF_WRITE))
		    err = E_PERM;
		break;
	    case BP_NAME:
		if (rhs.type != TYPE_STR)
		    err = E_TYPE;
		else if (!is_wizard(progr)
			 && (is_user(obj.v.obj)
			     || progr != db_object_owner(obj.v.obj)))
		    err = E_PERM;
		break;
	    case BP_OWNER:
		if (rhs.type != TYPE_OBJ)
		    err = E_TYPE;
		else if (!is_wizard(progr))
		    err = E_PERM;
		break;
	    case BP_PROGRAMMER:
	    case BP_WIZARD:
		if (!is_wizard(progr))
		    err = E_PERM;
		else if (h.built_in == BP_WIZARD
			 && !is_true(rhs) != !is_wizard(obj.v.obj)) {
		    /* Notify only on changes in state; the !'s above
		     * serve to canonicalize the truth values.
		     */
		    /* First make sure traceback will be accurate. */
		    STORE_STATE_VARIABLES();
		    oklog("%sWIZARDED: #%d by programmer #%d\n",
			  is_wizard(obj.v.obj) ? "DE" : "",
			  obj.v.obj, progr);
		    print_error_backtrace(is_wizard(obj.v.obj)
					  ? "Wizard bit unset."
					  : "Wizard bit set.",
					  output_to_log);
		}
		break;
	    case BP_R:
	    case BP_W:
	    case BP_F:
		if (progr != db_object_owner(obj.v.obj)
		    && !is_wizard(progr))
		    err = E_PERM;
		break;
	    case BP_LOCATION:
	    case BP_CONTENTS:
		err = E_PERM;
		break;
	    default:
		panic("Unknown built-in property in OP_PUT_PROP!");
	    }
	}

	free_var(propname);
	free_var(obj);
	if (err == E_NONE) {
	    db_set_property_value(h, var_ref(rhs));
	    PUSH(rhs);
	} else {
	    free_var(rhs);
	    PUSH_ERROR(err);
	}
    }
}
break;
#endif

/* {list, index, value}p => {list, ?, ?} + E_NONE
 * with value, index, old list freed
 *
 * {list, index, value}p => {list, index, value}
 *                              + err
 */
static enum error
indexset_helper(Var *args)
{	
    int idx;

    if (args[-2].type != TYPE_INT)
	return E_TYPE;
    idx = args[-2].v.num;

    if (args[-3].type == TYPE_LIST) {
	Var res;

	if (idx < 1 || idx > args[-3].v.list[0].v.num)
	    return E_RANGE;
	if (var_refcount(args[-3]) == 1)
	    res = args[-3];
	else {
	    res = var_dup(args[-3]);
	    free_var(args[-3]);
	}
	free_var(res.v.list[idx]);
	res.v.list[idx] = args[-1];
	args[-3] = res;
    } else if (args[-3].type == TYPE_STR) {
	char *tmp_str;

	if (args[-1].type != TYPE_STR)
	    return E_TYPE;
	if (idx < 1 || idx > memo_strlen(args[-3].v.str))
	    return E_RANGE;
	if (memo_strlen(args[-1].v.str) != 1)
	    return E_INVARG;

	if (var_refcount(args[-3]) == 1) {
	    /* blah blah discards const blah */
	    char *p = &args[-3].v.str[idx - 1];
	    *p = args[-1].v.str[0];
	} else {
	    tmp_str = str_dup(args[-3].v.str);
	    tmp_str[idx - 1] = args[-1].v.str[0];
	    free_str(args[-3].v.str);
	    args[-3].v.str = tmp_str;
	}
	free_str(args[-1].v.str);
    } else
	return E_TYPE;

    return E_NONE;
}

void
jim_INDEXSET(jit_state * jit, unsigned list_typemask, unsigned idx_typemask)
{
    jit_insn *bot = 0, *alt = 0;

    if (list_typemask == TYPEMASK(TYPE_LIST) && idx_typemask == TYPEMASK(TYPE_INT)) {
	jit_ldxi_p(JIT_V2, JIM_RTS, Offset(Var, v.list) - 3*sizeof(Var));
	jit_ldxi_i(JIT_R0, JIT_V2, -sizeof(int));
	alt = jit_bnei_i(jit_forward(), JIT_R0, 1);
	jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, v.num) - 2*sizeof(Var));
	jit_blei_i(L_raise_E_RANGE, JIT_R0, 0);
	jit_ldxi_i(JIT_R1, JIT_V2, Offset(Var, v.num));
	jit_bgtr_i(L_raise_E_RANGE, JIT_R0, JIT_R1);
	jim_POPn(2);
	jit_lshi_i(JIT_R0, JIT_R0, VAR_SHIFT);
	jit_addr_p(JIT_V2, JIT_V2, JIT_R0);
	jim_free_var(JIT_V2);
	jim_copy_var_R0R1(JIT_V2, 0, JIM_RTS, sizeof(Var));
	bot = jit_jmpi(jit_forward());
    }
    if (alt)
	jit_patch(alt);
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(indexset_helper);
    jit_retval(JIT_R0);
    jit_bnei_i(L_raise_error, JIT_R0, E_NONE);
    jim_POPn(2);
    if (bot)
	jit_patch(bot);
    jim_SPACER_NOP();
}


void
make_for_list(jit_state * jit, int slot, jit_insn ** topp, jit_insn ** rel1p)
{
    jit_insn *rel1, *top;

    jim_PUSH_NUM(jit, 0);
    /* peek at type of list we're iterating over */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var) * 2);
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_LIST);

    /* arguments validated, main body */
    top = jit_get_label();
    /* R0 = idx */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, v.num) - sizeof(Var));
    /* V2 = list.v.list */
    jit_ldxi_i(JIT_V2, JIM_RTS, Offset(Var, v.list) - sizeof(Var) * 2);
    /* R2 = list.v.list[0].v.num */
    jit_ldxi_i(JIT_R2, JIT_V2, Offset(Var, v.num));
    rel1 = jit_bger_i(jit_forward(), JIT_R0, JIT_R2);

    /* good to go */
    /* increment counter in register and on stack */
    jit_addi_i(JIT_R0, JIT_R0, 1);
    jit_stxi_i(Offset(Var, v.num) - sizeof(Var), JIM_RTS, JIT_R0);
    /* get list offset in safe reg */
    jit_lshi_i(JIT_R0, JIT_R0, VAR_SHIFT);
    jit_addr_p(JIT_V2, JIT_V2, JIT_R0);
    /* get slot and free it */
    jim_lea_RT_ENV_slot_i(JIT_R2, slot);
    jim_free_var(JIT_R2);
    jim_copy_var_R0R1(JIT_R2, 0, JIT_V2, 0);
    jim_var_ref(JIT_V2);

    *topp = top;
    *rel1p = rel1;
    jim_SPACER_NOP();
}

void
make_for_list_end(jit_state * jit, int slot, jit_insn * top, jit_insn * rel1, jit_insn * rel2)
{
    jit_jmpi(top);
    jit_patch(rel1);
    /* it was a good loop while it lasted */
    /* jim_free_var(JIM_RTS); not necessary for known int */
    jim_POPn(2);
    /* but free the list */
    jim_free_var(JIM_RTS);
    if (rel2)
	jit_patch(rel2);
    jim_SPACER_NOP();
}


void
make_for_range(jit_state * jit, int slot, unsigned typemask, jit_insn ** topp, jit_insn ** rel1p)
{
    jit_insn *rel1, *top;

    /* check arg types */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var));
    jit_ldxi_i(JIT_R2, JIM_RTS, Offset(Var, type) - sizeof(Var) * 2);

    rel1 = jit_beqi_i(jit_forward(), JIT_R0, TYPE_INT);
    jit_bnei_i(L_raise_E_TYPE, JIT_R0, TYPE_OBJ);
    jit_patch(rel1);
    jit_bner_i(L_raise_E_TYPE, JIT_R0, JIT_R2);

    /* arguments validated, main body */
    top = jit_get_label();
    /* check limit */
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, v.num) - sizeof(Var));
    jit_ldxi_i(JIT_V2, JIM_RTS, Offset(Var, v.num) - sizeof(Var) * 2);
    rel1 = jit_bgtr_i(jit_forward(), JIT_V2, JIT_R0);

    /* good to go */
    jim_lea_RT_ENV_slot_i(JIT_R2, slot);
    /*
     * If the loop var could be complex, free it
     */
    if (typemask & TYPEMASK_COMPLEX)
        jim_free_var(JIT_R2);
    jit_stxi_i(Offset(Var, v.num), JIT_R2, JIT_V2);
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) - sizeof(Var));
    jit_stxi_i(Offset(Var, type), JIT_R2, JIT_R0);
    jit_addi_i(JIT_V2, JIT_V2, 1);
    jit_stxi_i(Offset(Var, v.num) - sizeof(Var) * 2, JIM_RTS, JIT_V2);

    *topp = top;
    *rel1p = rel1;
    jim_SPACER_NOP();
}

void
make_for_range_end(jit_state * jit, int slot, jit_insn * top, jit_insn * rel1, jit_insn * rel2)
{
    jit_jmpi(top);
    jit_patch(rel1);
    /* it was a good loop while it lasted */
    /* jim_free_var(JIM_RTS); not necessary for known int */
    jim_POPn(2);
    if (rel2)
	jit_patch(rel2);
    jim_SPACER_NOP();
}

void
make_while(jit_state * jit, int slot, jit_insn ** topp)
{
    *topp = jit_get_label();
}

void
make_while_test(jit_state * jit, int slot, unsigned c_typemask, jit_insn ** rel1p)
{
    jim_POP();
    if (slot >= 0) {
	jim_lea_RT_ENV_slot_i(JIT_V2, slot);
	jim_free_var(JIT_V2);
	jim_copy_var_R0R1(JIT_V2, 0, JIM_RTS, 0);
	jim_var_ref(JIT_V2);
    }
    if (c_typemask == TYPEMASK(TYPE_INT)) {
	jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, v.num));
	*rel1p = jit_beqi_i(jit_forward(), JIT_R0, 0);
    } else {
	*rel1p = jim_branch_if_false(jit_forward(), JIM_RTS, 1);
    }
    jim_SPACER_NOP();
}

void
make_while_end(jit_state * jit, int slot, jit_insn * top, jit_insn * rel1, jit_insn * rel2, jit_insn *rel3)
{
    jit_jmpi(top);
    jit_patch(rel1);
    if (rel2)
	jit_patch(rel2);
    if (rel3)
	jit_patch(rel3);
    jim_SPACER_NOP();
}




void
make_if_test(jit_state * jit, jit_insn ** elsep)
{
    jim_POP();
    *elsep = jim_branch_if_false(jit_forward(), JIM_RTS, 1);
    jim_SPACER_NOP();
}

void
make_if_middle(jit_state * jit, jit_insn * elsep, jit_insn ** endp)
{
    *endp = jit_jmpi(jit_forward());
    jit_patch(elsep);
    jim_SPACER_NOP();
}

void
make_if_end(jit_state * jit, jit_insn * endp)
{
    jit_patch(endp);
}




void
make_and_or(jit_state * jit, int is_and, jit_insn ** relp)
{
    jit_subi_p(JIT_R0, JIM_RTS, sizeof(Var));
    *relp = is_and ? jim_branch_if_false(jit_forward(), JIT_R0, 0) :
	jim_branch_if_true(jit_forward(), JIT_R0, 0);
    /* if we don't branch, ditch this value and fall through to the next */
    jim_POP_AND_FREE(jit);
}

void
jim_NOT(jit_state * jit)
{
    jit_insn *rel;

    jim_POP();
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(is_truep_free);
    jit_retval(JIT_R0);
    jit_xori_i(JIT_R1, JIT_R0, 1);
    jit_movi_i(JIT_R0, TYPE_INT);
    jit_stxi_i(Offset(Var, v.num), JIM_RTS, JIT_R1);
    jit_stxi_i(Offset(Var, type), JIM_RTS, JIT_R0);
    jim_PUSH();
    jim_SPACER_NOP();
}

void
make_and_or_end(jit_state * jit, int is_and, jit_insn * relp)
{
    jit_patch(relp);
}


void
jim_PUSH_SLOT(jit_state * jit, int slot, int clear, int guaranteed, unsigned typemask)
{
    jim_lea_RT_ENV(JIT_R2);

    if (!guaranteed) {
	jit_ldxi_i(JIT_R0, JIT_R2, Offset(Var, type) + slot * sizeof(Var));
	jit_beqi_i(L_raise_E_VARNF, JIT_R0, TYPE_NONE);
    }
    jim_copy_var_R0R1(JIM_RTS, 0, JIT_R2, slot * sizeof(Var));
    if (typemask & TYPEMASK_COMPLEX) {
	if (clear) {
	    jit_movi_i(JIT_R0, TYPE_NONE);
	    jit_stxi_i(Offset(Var, type) + slot * sizeof(Var), JIT_R2, JIT_R0);
	} else
	    jim_var_ref(JIM_RTS);
    }
    jim_PUSH();
    jim_SPACER_NOP();
}

void
jim_PUT_SLOT(jit_state * jit, int slot, int keep_on_stack, int was_last_use, unsigned old_typemask, unsigned new_typemask)
{
    int same_type;

    same_type = (TYPEMASK_IS_EXACT(old_typemask) && old_typemask == new_typemask) && !(was_last_use && (old_typemask & TYPEMASK_COMPLEX));

    jim_lea_RT_ENV_slot_i(JIT_V2, slot);
    /* free old value */
    if (!was_last_use && (old_typemask & TYPEMASK_COMPLEX))
	jim_free_var(JIT_V2);
    /* copy into env slot */
    if (!same_type)
	jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type) -sizeof(Var));
    jit_ldxi_i(JIT_R1, JIM_RTS, Offset(Var, v) -sizeof(Var));
    if (!same_type)
	jit_stxi_i(Offset(Var, type), JIT_V2, JIT_R0);
    jit_stxi_i(Offset(Var, v), JIT_V2, JIT_R1);
    /* now either pop it or ref it */
    if (keep_on_stack) {
	if (new_typemask & TYPEMASK_COMPLEX)
	    jim_var_ref(JIT_V2);
    } else {
	jim_POP();
    }
    jim_SPACER_NOP();
}

void *
jim_call_verb(jit_state * jit, int entry_id)
{
    jim_STORE_STATE_VARIABLES();
    jit_movi_i(JIT_R0, entry_id);
    jit_stxi_i(Offset(activation, pc), JIM_RUN_ACTIV, JIT_R0);
    jit_movi_i(JIT_RET, RUNACT_CALL_VERB);
    jit_ret();
    jim_SPACER_NOP();
    return jit_get_ip().vptr;
}

void *
jim_call_bi(jit_state * jit, int func_id, int entry_id)
{
    jim_STORE_STATE_VARIABLES();
    jit_movi_i(JIT_R0, entry_id);
    jit_stxi_i(Offset(activation, pc), JIM_RUN_ACTIV, JIT_R0);
    jit_movi_i(JIT_R0, func_id);
    jit_stxi_i(Offset(activation, jit_func_id), JIM_RUN_ACTIV, JIT_R0);
    jit_movi_i(JIT_RET, RUNACT_CALL_BI);
    jit_ret();
    jim_SPACER_NOP();
    return jit_get_ip().vptr;
}

static enum error
do_addp(Var * args)
{
    if (args[-2].type == TYPE_STR && args[-1].type == TYPE_STR) {
	char *str;
	int l;

	str = mymalloc(((l = memo_strlen(args[-2].v.str)) +
			memo_strlen(args[-1].v.str) + 1), M_STRING);
	strcpy(str, args[-2].v.str);
	strcpy(str + l, args[-1].v.str);
	free_str(args[-2].v.str);
	free_str(args[-1].v.str);
	args[-2].v.str = str;
	return E_NONE;
    } else {
	Var t = do_add(args - 2, args - 1);
	if (t.type == TYPE_ERR)
	    return t.v.err;
	else {
	    free_var(args[-2]);
	    free_var(args[-1]);
	    args[-2] = t;
	    return E_NONE;
	}
    }
}

void
jim_ADD_VVs(jit_state * jit, int slot_s2, int slot_src)
{
    jit_movi_i(JIT_R2, TYPE_INT);
    jim_lea_RT_ENV(JIT_V2);
    jit_ldxi_i(JIT_R0, JIT_V2, Offset(Var, v.num) + sizeof(Var) * slot_src);
    jit_ldxi_i(JIT_R1, JIT_V2, Offset(Var, v.num) + sizeof(Var) * slot_s2);
    jit_stxi_i(Offset(Var, type), JIM_RTS, JIT_R2);
    jit_addr_i(JIT_R1, JIT_R1, JIT_R0);
    jit_stxi_i(Offset(Var, v.num), JIM_RTS, JIT_R1);
    jim_PUSH();
    jim_SPACER_NOP();
}

void
jim_ADD(jit_state * jit, unsigned l_typemask, unsigned r_typemask)
{
    if (l_typemask == TYPEMASK(TYPE_INT) && r_typemask == TYPEMASK(TYPE_INT)) {
	jim_POPn(1);
	jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, v.num));
	jit_ldxi_i(JIT_R1, JIM_RTS, Offset(Var, v.num) - sizeof(Var));
	jit_addr_i(JIT_R1, JIT_R0, JIT_R1);
	jit_stxi_i(Offset(Var, v.num) - sizeof(Var), JIM_RTS, JIT_R1);
    } else {
	jit_prepare(1);
	jit_pusharg_p(JIM_RTS);
	jit_finish(do_addp);
	jit_retval(JIT_R0);
	jit_bnei_i(L_raise_error, JIT_R0, E_NONE);
	jim_POP();
    }
    jim_SPACER_NOP();
}

static enum error
do_dividep(Var *args)
{
    Var t = do_divide(args - 2, args - 1);
    if (t.type == TYPE_ERR)
	return t.v.err;
    free_var(args[-2]);
    free_var(args[-1]);
    args[-2] = t;
    return E_NONE;
}

void
jim_DIVIDE(jit_state *jit)
{
    jit_insn *rel;

    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(do_dividep);
    jit_retval(JIT_R0);
    jit_bnei_i(L_raise_error, JIT_R0, E_NONE);
    jim_POP();
    jim_SPACER_NOP();
}


static enum error
do_modulusp(Var *args)
{
    Var t = do_modulus(args - 2, args - 1);
    if (t.type == TYPE_ERR)
	return t.v.err;
    free_var(args[-2]);
    free_var(args[-1]);
    args[-2] = t;
    return E_NONE;
}

void
jim_MODULUS(jit_state *jit)
{
    jit_insn *rel;

    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(do_modulusp);
    jit_retval(JIT_R0);
    jit_bnei_i(L_raise_error, JIT_R0, E_NONE);
    jim_POP();
    jim_SPACER_NOP();
}

void
do_subtractp(Var * args)
{
    Var t = do_subtract(args, args + 1);
    free_var(args[0]);
    free_var(args[1]);
    args[0] = t;
}

void
jim_SUBTRACT(jit_state * jit)
{
    jit_insn *rel;

    jim_POPn(2);
    jit_prepare(1);
    jit_pusharg_p(JIM_RTS);
    jit_finish(do_subtractp);
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, type));
    rel = jit_bnei_i(jit_forward(), JIT_R0, TYPE_ERR);
    jit_ldxi_i(JIT_R0, JIM_RTS, Offset(Var, v.err));
    jit_jmpi(L_raise_error);
    jit_patch(rel);
    jim_PUSH();
    jim_SPACER_NOP();
}

/* indexed by char, can't overflow */
static struct scatter_helper_info {
    int id;
    int flag;
} scatter_info[257];

static enum error
scatter_helper(Var * args, int nargs_nreq_rest, int base_pc)
{
    /* pavel guaranteed these are bytes, so put them in a single word to
     * keep generated code size down
     */
    int nargs = nargs_nreq_rest & 0xff;
    int nreq = (nargs_nreq_rest >> 8) & 0xff;
    int rest = (nargs_nreq_rest >> 16) & 0xff;
    int have_rest = (rest > nargs ? 0 : 1);
    Var list;
    int len = 0, nopt_avail, nrest, i, offset;
    int done, where = 0;
    int label = base_pc;

    list = args[-1];
    /* always +d, so we don't have to worry about freeing list or
     * jumping over defaults, just return error to be raised
     */
    if (list.type != TYPE_LIST)
	return E_TYPE;
    else if ((len = list.v.list[0].v.num) < nreq
	     || (!have_rest && len > nargs))
	return E_ARGS;

    nopt_avail = len - nreq;
    nrest = (have_rest && len >= nargs ? len - nargs + 1 : 0);
    for (offset = 0, i = 1; i <= nargs; i++) {
	int id = scatter_info[i].id;
	int flag = scatter_info[i].flag;

	if (i == rest) {	/* rest */
	    free_var(RUN_ACTIV.rt_env[id]);
	    RUN_ACTIV.rt_env[id] = sublist(var_ref(list),
					   i,
					   i + nrest - 1);
	    offset += nrest - 1;
	} else if (flag == 0) {	/* required */
	    free_var(RUN_ACTIV.rt_env[id]);
	    RUN_ACTIV.rt_env[id] =
		var_ref(list.v.list[i + offset]);
	} else {		/* optional */
	    if (nopt_avail > 0) {
		nopt_avail--;
		free_var(RUN_ACTIV.rt_env[id]);
		RUN_ACTIV.rt_env[id] =
		    var_ref(list.v.list[i + offset]);
	    } else {
		offset--;
		if (where == 0 && flag == 2)
		    where = label;
	    }
	    if (flag == 2)	/* has default */
		++label;
	}
    }

    if (where == 0)
	RUN_ACTIV.pc = label;	/* to the end */
    else
	RUN_ACTIV.pc = where;	/* to the defaults */
    return E_NONE;
}

void
jim_SCATTER_start_id(jit_state * jit)
{
    jit_movi_p(JIT_R1, scatter_info);
}

void
jim_SCATTER_add_id(jit_state * jit, int offset, int id, int flag)
{
    jit_movi_i(JIT_R0, id);
    jit_stxi_i(offset * sizeof(scatter_info[1]), JIT_R1, JIT_R0);
    jit_movi_i(JIT_R0, flag);
    jit_stxi_i(offset * sizeof(scatter_info[1]) + sizeof(int), JIT_R1, JIT_R0);
}

void
jim_SCATTER_body(jit_state * jit, int nargs, int nreq, int rest, int base_pc)
{
    /* stuff it into one word to keep it short.  255 is the limit for
     * all of these */
    unsigned nargs_nreq_rest = (nargs & 0xff) |
    ((nreq & 0xff) << 8) |
    ((rest & 0xff) << 16);
    jit_prepare(3);
    jit_movi_i(JIT_R0, base_pc);
    jit_pusharg_i(JIT_R0);
    jit_movi_i(JIT_R0, nargs_nreq_rest);
    jit_pusharg_i(JIT_R0);
    jit_pusharg_p(JIM_RTS);
    jit_finish(scatter_helper);
    jit_retval(JIT_R0);
    jit_bnei_i(L_raise_error, JIT_R0, E_NONE);
    if (base_pc != -1)
	jit_jmpi(L_jump_to_pc);
}
void *
jim_SCATTER_next_default(jit_state * jit)
{
    return jit_get_ip().ptr;
}

void
jim_JMP(jit_state * jit, jit_insn * dest)
{
    jit_jmpi(dest);
}

void
jim_unwind_stack_to_goal(jit_state * jit, int goal)
{
    jit_insn *rel, *loop;

    /* goal is how many items to leave on the stack */
    jit_ldxi_p(JIT_V2, JIM_RUN_ACTIV, Offset(activation, base_rt_stack));
    jit_addi_p(JIT_V2, JIT_V2, goal * sizeof(Var));
    loop = jit_get_label();
    rel = jit_beqr_ui(jit_forward(), JIM_RTS, JIT_V2);
    jim_POP();
    jim_free_var(JIM_RTS);
    jit_jmpi(loop);
    jit_patch(rel);
}
void
jim_CONTINUE(jit_state * jit, jit_insn ** continuer, jit_insn * top, int goal)
{
    /* if we made one, just reuse it */
    if (*continuer) {
	jit_jmpi(*continuer);
    } else {
	*continuer = jit_get_label();
	jim_unwind_stack_to_goal(jit, goal);
	jit_jmpi(top);
    }
}

void
jim_BREAK(jit_state * jit, jit_insn ** breaker, jit_insn ** fixup, int goal)
{
    /* if we made one, just reuse it */
    if (*breaker) {
	jit_jmpi(*breaker);
    } else {
	*breaker = jit_get_label();
	jim_unwind_stack_to_goal(jit, goal);
	*fixup = jit_jmpi(jit_forward());
    }
}

#define POP() (*(--RUN_ACTIV.top_rt_stack))
#define PUSH(x) (*(RUN_ACTIV.top_rt_stack++) = x)

int (*jit_run_glue) ();
char jit_run_glue_buf[1024];

enum outcome
jit_run(Var * result)
{
    enum outcome outcome;

    while (1) {
	int act = (*jit_run_glue) ();

	switch (act) {
	case RUNACT_ABORT:
	    unwind_stack(FIN_ABORT, zero, 0);
	    return OUTCOME_ABORTED;

	case RUNACT_BLOCK:
	    return OUTCOME_BLOCKED;

	case RUNACT_RETURN:
	case RUNACT_RETURN0:{
		Var ret_val;

		if (act == RUNACT_RETURN)
		    ret_val = POP();
		else
		    ret_val = zero;

		if (unwind_stack(FIN_RETURN, ret_val, &outcome)) {
		    if (result && outcome == OUTCOME_DONE)
			*result = ret_val;
		    else
			free_var(ret_val);
		    return outcome;
		}
		break;
	    }

	case RUNACT_CALL_VERB:{
		enum error err;

		Var args, verb, obj;

		args = POP();
		verb = POP();
		obj = POP();

		if (args.type != TYPE_LIST || verb.type != TYPE_STR ||
		    obj.type != TYPE_OBJ)
		    err = E_TYPE;
		else if (!valid(obj.v.obj))
		    err = E_INVIND;
		else
		    err = call_verb2(obj.v.obj, verb.v.str, args, 0);
		free_var(obj);
		free_var(verb);

		if (err != E_NONE) {
		    free_var(args);
		    RUN_ACTIV.jit_error_out = err;
		    goto runact_raise;
		} else
		    break;
	    }

	case RUNACT_CALL_BI:{
		unsigned func_id;
		Var args;

		func_id = RUN_ACTIV.jit_func_id;
		args = POP();
		if (args.type != TYPE_LIST) {	/* XXX possible +d? no? */
		    free_var(args);
		    RUN_ACTIV.jit_error_out = E_TYPE;
		    goto runact_raise;
		} else {
		    package p;

		    p = call_bi_func(func_id, args, 1,
				     RUN_ACTIV.progr, 0);
		    switch (p.kind) {
		    case BI_RETURN:
			PUSH(p.u.ret);
			break;
		    case BI_RAISE:
			/* always +d in jit */
			if (raise_error(&p, 0))
			    return OUTCOME_ABORTED;
			break;
		    case BI_CALL:
			/* builtin fn called a verb */
			RUN_ACTIV.bi_func_id = func_id;
			RUN_ACTIV.bi_func_data = p.u.call.data;
			RUN_ACTIV.bi_func_pc = p.u.call.pc;
			break;
		    case BI_SUSPEND:
			RUN_ACTIV.jit_error_out =
			    suspend_task(p);
			if (RUN_ACTIV.jit_error_out == E_NONE)
			    return OUTCOME_BLOCKED;
			else
			    goto runact_raise;
			/*NOTREACHED */
		    case BI_KILL:
			unwind_stack(FIN_ABORT, zero, 0);
			return OUTCOME_ABORTED;
		    }
		}
		break;
	    }

	  runact_raise:
	case RUNACT_RAISE:{
		package p = make_error_pack(RUN_ACTIV.jit_error_out);
		if (raise_error(&p, 0))
		    return OUTCOME_ABORTED;
		break;
	    }
	}
    }
}

jit_state jjj;
void
init_jit()
{
    jit_state *jit = &jjj;

    if (VAR_SHIFT == 0 || PTR_SHIFT == 0)
	panic("init_jit sizeof assertions for lshi");

    jit_run_glue = (int (*)()) (jit_set_ip(jit_run_glue_buf).iptr);

    jim_glue(jit);
    jit_flush_code(jit_run_glue_buf, jit_get_ip().ptr);
}
