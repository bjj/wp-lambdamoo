#define jim_FOR_RANGE(jit, slot, typemask) do { \
	jit_insn *__t, *__r;		\
	make_for_range(jit, slot, typemask, &__t, &__r);

#define jim_FOR_RANGE_end(jit, slot, extra) \
	make_for_range_end(jit, slot, __t, __r, extra); \
	} while (0)

#define jim_FOR_LIST(jit, slot) do { \
	jit_insn *__t, *__r;		\
	make_for_list(jit, slot, &__t, &__r);

#define jim_FOR_LIST_end(jit, slot, extra) \
	make_for_list_end(jit, slot, __t, __r, extra); \
	} while (0)

#define jim_WHILE(jit, slot) do { \
	jit_insn *__t, *__r;		\
	make_while(jit, slot, &__t);

#define jim_WHILE_test(jit, slot, typemask) \
	make_while_test(jit, slot, typemask, &__r)

#define jim_WHILE_end(jit, slot, extra) \
	make_while_end(jit, slot, __t, __r, extra); \
	} while (0)

#define jim_AND_OR(jit, is_and) do { \
	jit_insn *__r;			\
	make_and_or(jit, is_and, &__r);

#define jim_AND_OR_end(jit, is_and) \
	make_and_or_end(jit, is_and, __r); \
	} while (0);
