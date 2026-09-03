/* term.h — Term space of Autostasis, C rewrite.
 *
 * Terms are a tagged union on an arena. Sharing is real: two pointers
 * to the same node are the same structure (Fix's zero-copy economy).
 */
#ifndef AUTOSTASIS_TERM_H
#define AUTOSTASIS_TERM_H

#include <stddef.h>

/* ── arena: bump allocator, chained blocks, 8-byte aligned ── */
typedef struct ArenaBlock {
    struct ArenaBlock *next;
    size_t used, cap;
    char mem[];
} ArenaBlock;

typedef struct Arena {
    ArenaBlock *head, *tail;
} Arena;

void *a_alloc(Arena *a, size_t sz);

/* ── tags ── */
typedef enum {
    T_VAR, T_LAM, T_APP, T_QUOTE, T_EVAL, T_FIX,
    T_NAT, T_PRIM, T_PARTIAL, T_CONS, T_CLOSURE, T_THUNK
} Tag;

typedef struct Term Term;
typedef struct Env Env;

struct Term {
    Tag tag;
    union {
        const char *name;                        /* T_VAR, T_PRIM */
        struct { const char *param; Term *body; } lam;
        struct { Term *f; Term *a; } app;
        Term *inner;                             /* T_QUOTE */
        struct { Term *q; Term *a; } ev;         /* T_EVAL */
        Term *func;                              /* T_FIX */
        long n;                                  /* T_NAT */
        struct { const char *name; Term *a1; } partial;
        struct { Term *car; Term *cdr; } cons;
        struct { Term *lam; Env *env; } closure; /* T_CLOSURE: WHNF of Lam */
        struct { Term *term; Env *env; } thunk;  /* T_THUNK: frozen arg */
    } u;
};

/* constructors (allocate on arena) */
Term *t_var(Arena *a, const char *name);
Term *t_lam(Arena *a, const char *param, Term *body);
Term *t_app(Arena *a, Term *f, Term *x);
Term *t_quote(Arena *a, Term *t);
Term *t_eval(Arena *a, Term *q, Term *x);
Term *t_fix(Arena *a, Term *f);
Term *t_nat(Arena *a, long n);
Term *t_prim(Arena *a, const char *name);
Term *t_partial(Arena *a, const char *name, Term *a1);
Term *t_cons(Arena *a, Term *car, Term *cdr);
Term *t_closure(Arena *a, Term *lam, Env *env);
Term *t_thunk(Arena *a, Term *term, Env *env);

/* symbol interning: same string -> same pointer forever */
const char *intern(const char *s);

/* lexical environment: chained name/value bindings */
struct Env { const char *name; Term *val; Env *parent; };

/* core operations */
Term *eval_term(Arena *a, Term *t, Env *env, int fd);
void print_term(Term *t);          /* s-expression to stdout */
int term_equal(Term *x, Term *y);  /* structural equality (==) */

/* Church booleans (global unique pointers: == is truth testing) */
extern Term *G_TRUE, *G_FALSE;

#endif
