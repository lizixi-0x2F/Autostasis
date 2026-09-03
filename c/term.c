/* term.c — constructors, arena, symbol interning, structural equality. */
#include "term.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_CAP (1 << 23)   /* 8 MB blocks */

void *a_alloc(Arena *a, size_t sz) {
    sz = (sz + 7) & ~(size_t)7;
    if (!a->tail || a->tail->used + sz > a->tail->cap) {
        ArenaBlock *b = malloc(sizeof(ArenaBlock) + BLOCK_CAP);
        if (!b) { perror("malloc"); exit(1); }
        b->next = NULL; b->used = 0; b->cap = BLOCK_CAP;
        if (a->tail) a->tail->next = b; else a->head = b;
        a->tail = b;
    }
    void *p = a->tail->mem + a->tail->used;
    a->tail->used += sz;
    return p;
}

/* ── symbol interning ── */
#define SYM_TABLE_BITS 12
#define SYM_TABLE_SIZE (1 << SYM_TABLE_BITS)
typedef struct Sym { const char *name; struct Sym *next; } Sym;
static Sym *sym_table[SYM_TABLE_SIZE];

static unsigned hash_str(const char *s) {
    unsigned h = 5381;
    while (*s) h = h * 33 + (unsigned char)*s++;
    return h & (SYM_TABLE_SIZE - 1);
}

const char *intern(const char *s) {
    unsigned h = hash_str(s);
    for (Sym *p = sym_table[h]; p; p = p->next)
        if (strcmp(p->name, s) == 0) return p->name;
    Sym *e = malloc(sizeof(Sym) + strlen(s) + 1);
    e->name = (char *)(e + 1);
    strcpy((char *)e->name, s);
    e->next = sym_table[h];
    sym_table[h] = e;
    return e->name;
}

/* ── constructors ── */
Term *t_var(Arena *a, const char *name) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_VAR; t->u.name = intern(name);
    return t;
}
Term *t_lam(Arena *a, const char *param, Term *body) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_LAM; t->u.lam.param = intern(param); t->u.lam.body = body;
    return t;
}
Term *t_app(Arena *a, Term *f, Term *x) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_APP; t->u.app.f = f; t->u.app.a = x;
    return t;
}
Term *t_quote(Arena *a, Term *x) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_QUOTE; t->u.inner = x;
    return t;
}
Term *t_eval(Arena *a, Term *q, Term *x) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_EVAL; t->u.ev.q = q; t->u.ev.a = x;
    return t;
}
Term *t_fix(Arena *a, Term *f) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_FIX; t->u.func = f;
    return t;
}
Term *t_nat(Arena *a, long n) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_NAT; t->u.n = n;
    return t;
}
Term *t_prim(Arena *a, const char *name) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_PRIM; t->u.name = intern(name);
    return t;
}
Term *t_partial(Arena *a, const char *name, Term *a1) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_PARTIAL; t->u.partial.name = intern(name); t->u.partial.a1 = a1;
    return t;
}
Term *t_cons(Arena *a, Term *car, Term *cdr) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_CONS; t->u.cons.car = car; t->u.cons.cdr = cdr;
    return t;
}
Term *t_closure(Arena *a, Term *lam, Env *env) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_CLOSURE; t->u.closure.lam = lam; t->u.closure.env = env;
    return t;
}
Term *t_thunk(Arena *a, Term *term, Env *env) {
    Term *t = a_alloc(a, sizeof(Term));
    t->tag = T_THUNK; t->u.thunk.term = term; t->u.thunk.env = env;
    return t;
}

/* ── Church booleans as global unique pointers ── */
static Term _true_lam, _false_lam;
static Term _true_body, _false_body;
static Term _t_var, _f_var, _tf_var, _ff_var;
Term *G_TRUE, *G_FALSE;

static void init_church(void) {
    /* TRUE = λt. λf. t */
    _t_var.tag = T_VAR; _t_var.u.name = intern("t");
    _f_var.tag = T_VAR; _f_var.u.name = intern("f");
    _true_lam.tag = T_LAM; _true_lam.u.lam.param = intern("f"); _true_lam.u.lam.body = &_t_var;
    _true_body.tag = T_LAM; _true_body.u.lam.param = intern("t"); _true_body.u.lam.body = &_true_lam;
    /* FALSE = λt. λf. f */
    _tf_var.tag = T_VAR; _tf_var.u.name = intern("t");
    _ff_var.tag = T_VAR; _ff_var.u.name = intern("f");
    _false_lam.tag = T_LAM; _false_lam.u.lam.param = intern("f"); _false_lam.u.lam.body = &_ff_var;
    _false_body.tag = T_LAM; _false_body.u.lam.param = intern("t"); _false_body.u.lam.body = &_false_lam;
    G_TRUE = &_true_body; G_FALSE = &_false_body;
}

void __attribute__((constructor)) autostasis_init(void) { init_church(); }

/* ── structural equality (dataclass ==) ── */
int term_equal(Term *x, Term *y) {
    if (x == y) return 1;
    if (x->tag != y->tag) return 0;
    switch (x->tag) {
        case T_VAR: case T_PRIM:
            return x->u.name == y->u.name;
        case T_NAT: return x->u.n == y->u.n;
        case T_LAM:
            return x->u.lam.param == y->u.lam.param
                && term_equal(x->u.lam.body, y->u.lam.body);
        case T_APP: return term_equal(x->u.app.f, y->u.app.f)
                         && term_equal(x->u.app.a, y->u.app.a);
        case T_QUOTE: return term_equal(x->u.inner, y->u.inner);
        case T_EVAL: return term_equal(x->u.ev.q, y->u.ev.q)
                          && term_equal(x->u.ev.a, y->u.ev.a);
        case T_FIX: return term_equal(x->u.func, y->u.func);
        case T_PARTIAL: return x->u.partial.name == y->u.partial.name
                             && term_equal(x->u.partial.a1, y->u.partial.a1);
        case T_CONS: return term_equal(x->u.cons.car, y->u.cons.car)
                          && term_equal(x->u.cons.cdr, y->u.cons.cdr);
        case T_CLOSURE: return term_equal(x->u.closure.lam, y->u.closure.lam);
        case T_THUNK: return term_equal(x->u.thunk.term, y->u.thunk.term);
    }
    return 0;
}
