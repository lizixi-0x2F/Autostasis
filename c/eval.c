/* eval.c — substitute (capture-avoiding), WHNF evaluator, sexpr printer.
 *
 * Semantics are a faithful port of src/eval.py + src/ops.py so the two
 * implementations can serve as oracles for each other.
 */
#include "term.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FIX 5000

static void die(const char *msg) {
    fprintf(stderr, "autostasis: %s\n", msg);
    exit(1);
}

/* ── free variables (into arena-allocated array) ── */
typedef struct { const char **names; size_t n, cap; } FvSet;

static void fv_add(Arena *a, FvSet *s, const char *name) {
    (void)a;
    for (size_t i = 0; i < s->n; i++)
        if (s->names[i] == name) return;      /* interned: pointer eq */
    if (s->n == s->cap) die("free-var set overflow");
    s->names[s->n++] = name;
}

static void collect_fv(Term *t, FvSet *s) {
    switch (t->tag) {
        case T_VAR: fv_add(NULL, s, t->u.name); break;
        case T_LAM:
            collect_fv(t->u.lam.body, s); break;   /* param shadowed: skip below */
        case T_APP:
            collect_fv(t->u.app.f, s); collect_fv(t->u.app.a, s); break;
        case T_QUOTE: collect_fv(t->u.inner, s); break;
        case T_EVAL:
            collect_fv(t->u.ev.q, s); collect_fv(t->u.ev.a, s); break;
        case T_FIX: collect_fv(t->u.func, s); break;
        case T_PARTIAL: collect_fv(t->u.partial.a1, s); break;
        case T_CONS:
            collect_fv(t->u.cons.car, s); collect_fv(t->u.cons.cdr, s); break;
        default: break;
    }
}

static int fv_contains(FvSet *s, const char *name) {
    for (size_t i = 0; i < s->n; i++)
        if (s->names[i] == name) return 1;
    return 0;
}

static long fresh_counter = 0;

static const char *fresh_var(FvSet *avoid, const char *prefix) {
    char buf[64];
    long start = fresh_counter++;
    for (long k = start;; k++) {
        snprintf(buf, sizeof buf, "%s%ld", prefix, k);
        const char *nm = intern(buf);
        if (!fv_contains(avoid, nm)) return nm;
    }
}

/* ── capture-avoiding substitution ── */
Term *substitute(Arena *a, Term *t, const char *var, Term *repl) {
    switch (t->tag) {
        case T_VAR:
            return t->u.name == var ? repl : t;
        case T_LAM: {
            const char *param = t->u.lam.param;
            if (param == var) return t;
            const char *rv_names[64];
            FvSet rv = {rv_names, 0, 64};
            collect_fv(repl, &rv);
            if (!fv_contains(&rv, param))
                return t_lam(a, param, substitute(a, t->u.lam.body, var, repl));
            /* capture: rename the binder */
            const char *bv_names[128];
            FvSet bv = {bv_names, 0, 128};
            collect_fv(t->u.lam.body, &bv);
            for (size_t i = 0; i < rv.n; i++)
                if (!fv_contains(&bv, rv.names[i]))
                    bv.names[bv.n++] = rv.names[i];
            const char *np = fresh_var(&bv, param);
            Term *nb = substitute(a, t->u.lam.body, param, t_var(a, np));
            return t_lam(a, np, substitute(a, nb, var, repl));
        }
        case T_APP:
            return t_app(a, substitute(a, t->u.app.f, var, repl),
                            substitute(a, t->u.app.a, var, repl));
        case T_QUOTE:
            return t_quote(a, substitute(a, t->u.inner, var, repl));
        case T_EVAL:
            return t_eval(a, substitute(a, t->u.ev.q, var, repl),
                             substitute(a, t->u.ev.a, var, repl));
        case T_FIX:
            return t_fix(a, substitute(a, t->u.func, var, repl));
        case T_PARTIAL:
            return t_partial(a, t->u.partial.name,
                             substitute(a, t->u.partial.a1, var, repl));
        case T_CONS:
            return t_cons(a, substitute(a, t->u.cons.car, var, repl),
                             substitute(a, t->u.cons.cdr, var, repl));
        default:
            return t;
    }
}

/* ── environment ── */
static Term *env_lookup(Env *env, const char *name) {
    for (Env *e = env; e; e = e->parent)
        if (e->name == name) return e->val;
    char buf[256];
    snprintf(buf, sizeof buf, "unbound variable: %s", name);
    die(buf);
    return NULL;
}

/* ── prim classification ── */
static int prim_binop(const char *nm) {
    static const char *bin[] = {"add","sub","mult","div","mod","pow",
                                "min","max","le","lt","ge","gt"};
    for (size_t i = 0; i < sizeof bin / sizeof bin[0]; i++)
        if (nm == intern(bin[i])) return 1;
    return 0;
}
static int prim_partial(const char *nm) {
    return nm == intern("eq") || nm == intern("eq_nat")
        || nm == intern("mk_app");
}

/* ── evaluator: WHNF, faithful port of src/eval.py ── */
Term *eval_term(Arena *a, Term *term, Env *env, int fd) {
    while (1) {
        switch (term->tag) {
            case T_VAR:
                term = env_lookup(env, term->u.name);
                break;

            case T_LAM: case T_NAT: case T_QUOTE:
            case T_PARTIAL: case T_CONS:
                return term;

            case T_APP: {
                Term *fval = eval_term(a, term->u.app.f, env, fd);
                Term *arg  = term->u.app.a;
                switch (fval->tag) {
                    case T_LAM:
                        term = substitute(a, fval->u.lam.body,
                                          fval->u.lam.param, arg);
                        break;

                    case T_PRIM: {
                        const char *nm = fval->u.name;
                        Term *av;
                        if (nm == intern("pred")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_NAT) die("pred expects Nat");
                            term = t_nat(a, av->u.n > 0 ? av->u.n - 1 : 0);
                        } else if (nm == intern("iszero")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_NAT) die("iszero expects Nat");
                            term = av->u.n == 0 ? G_TRUE : G_FALSE;
                        } else if (nm == intern("car")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_CONS) die("car expects Cons");
                            term = eval_term(a, av->u.cons.car, env, fd);
                        } else if (nm == intern("cdr")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_CONS) die("cdr expects Cons");
                            term = eval_term(a, av->u.cons.cdr, env, fd);
                        } else if (nm == intern("mk_nat")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_NAT) die("mk_nat expects Nat");
                            term = t_quote(a, av);
                        } else if (nm == intern("mk_lam")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE) die("mk_lam expects quoted body");
                            term = t_quote(a, t_lam(a, "x", av->u.inner));
                        } else if (nm == intern("get_body")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE || av->u.inner->tag != T_LAM)
                                die("get_body expects quoted Lam");
                            term = t_quote(a, av->u.inner->u.lam.body);
                        } else if (nm == intern("get_func")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE || av->u.inner->tag != T_APP)
                                die("get_func expects quoted App");
                            term = t_quote(a, av->u.inner->u.app.f);
                        } else if (nm == intern("get_arg")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE || av->u.inner->tag != T_APP)
                                die("get_arg expects quoted App");
                            term = t_quote(a, av->u.inner->u.app.a);
                        } else if (nm == intern("get_quoted")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE) die("get_quoted expects quoted Quote/Eval");
                            Term *in = av->u.inner;
                            if (in->tag == T_QUOTE) term = t_quote(a, in->u.inner);
                            else if (in->tag == T_EVAL) term = t_quote(a, in->u.ev.q);
                            else die("get_quoted expects quoted Quote/Eval");
                        } else if (nm == intern("get_eval_arg")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE || av->u.inner->tag != T_EVAL)
                                die("get_eval_arg expects quoted Eval");
                            term = t_quote(a, av->u.inner->u.ev.a);
                        } else if (nm == intern("get_fix_func")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE || av->u.inner->tag != T_FIX)
                                die("get_fix_func expects quoted Fix");
                            term = t_quote(a, av->u.inner->u.func);
                        } else if (nm == intern("get_car")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE || av->u.inner->tag != T_CONS)
                                die("get_car expects quoted Cons");
                            term = t_quote(a, av->u.inner->u.cons.car);
                        } else if (nm == intern("get_cdr")) {
                            av = eval_term(a, arg, env, fd);
                            if (av->tag != T_QUOTE || av->u.inner->tag != T_CONS)
                                die("get_cdr expects quoted Cons");
                            term = t_quote(a, av->u.inner->u.cons.cdr);
                        } else if (nm == intern("is_var")) {
                            av = eval_term(a, arg, env, fd);
                            term = (av->tag == T_QUOTE && av->u.inner->tag == T_VAR)
                                       ? G_TRUE : G_FALSE;
                        } else if (nm == intern("is_lam")) {
                            av = eval_term(a, arg, env, fd);
                            term = (av->tag == T_QUOTE && av->u.inner->tag == T_LAM)
                                       ? G_TRUE : G_FALSE;
                        } else if (nm == intern("is_app")) {
                            av = eval_term(a, arg, env, fd);
                            term = (av->tag == T_QUOTE && av->u.inner->tag == T_APP)
                                       ? G_TRUE : G_FALSE;
                        } else if (nm == intern("is_nat")) {
                            av = eval_term(a, arg, env, fd);
                            term = (av->tag == T_QUOTE && av->u.inner->tag == T_NAT)
                                       ? G_TRUE : G_FALSE;
                        } else if (nm == intern("is_quote")) {
                            av = eval_term(a, arg, env, fd);
                            term = (av->tag == T_QUOTE && av->u.inner->tag == T_QUOTE)
                                       ? G_TRUE : G_FALSE;
                        } else if (nm == intern("is_eval")) {
                            av = eval_term(a, arg, env, fd);
                            term = (av->tag == T_QUOTE && av->u.inner->tag == T_EVAL)
                                       ? G_TRUE : G_FALSE;
                        } else if (nm == intern("is_fix")) {
                            av = eval_term(a, arg, env, fd);
                            term = (av->tag == T_QUOTE && av->u.inner->tag == T_FIX)
                                       ? G_TRUE : G_FALSE;
                        } else if (nm == intern("is_cons")) {
                            av = eval_term(a, arg, env, fd);
                            term = (av->tag == T_QUOTE && av->u.inner->tag == T_CONS)
                                       ? G_TRUE : G_FALSE;
                        } else if (prim_partial(nm)) {
                            term = t_partial(a, nm, arg);
                        } else if (prim_binop(nm)) {
                            term = t_partial(a, nm, arg);
                        } else {
                            die("unknown prim");
                        }
                        break;
                    }

                    case T_PARTIAL: {
                        const char *nm = fval->u.partial.name;
                        Term *stored = fval->u.partial.a1;
                        if (nm == intern("mk_app")) {
                            Term *sv = eval_term(a, stored, env, fd);
                            Term *av = eval_term(a, arg, env, fd);
                            if (sv->tag != T_QUOTE || av->tag != T_QUOTE)
                                die("mk_app expects (quoted func, quoted arg)");
                            term = t_quote(a, t_app(a, sv->u.inner, av->u.inner));
                        } else if (nm == intern("eq")) {
                            Term *sv = eval_term(a, stored, env, fd);
                            Term *av = eval_term(a, arg, env, fd);
                            term = term_equal(sv, av) ? G_TRUE : G_FALSE;
                        } else if (nm == intern("eq_nat")) {
                            Term *sv = eval_term(a, stored, env, fd);
                            Term *av = eval_term(a, arg, env, fd);
                            if (sv->tag != T_NAT || av->tag != T_NAT)
                                die("eq_nat expects (Nat, Nat)");
                            term = sv->u.n == av->u.n ? G_TRUE : G_FALSE;
                        } else {
                            Term *sv = eval_term(a, stored, env, fd);
                            Term *av = eval_term(a, arg, env, fd);
                            if (sv->tag != T_NAT || av->tag != T_NAT) {
                                char buf[128];
                                snprintf(buf, sizeof buf,
                                         "%s expects (Nat, Nat)", nm);
                                die(buf);
                            }
                            long x = sv->u.n, y = av->u.n;
                            if      (nm == intern("add"))  term = t_nat(a, x + y);
                            else if (nm == intern("sub"))  term = t_nat(a, x > y ? x - y : 0);
                            else if (nm == intern("mult")) term = t_nat(a, x * y);
                            else if (nm == intern("div"))  term = t_nat(a, y != 0 ? x / y : 0);
                            else if (nm == intern("mod"))  term = t_nat(a, y != 0 ? x % y : 0);
                            else if (nm == intern("pow")) {
                                if (y > 20) die("pow exponent too large");
                                long r = 1;
                                for (long i = 0; i < y; i++) r *= x;
                                term = t_nat(a, r);
                            } else if (nm == intern("min")) term = t_nat(a, x < y ? x : y);
                            else if (nm == intern("max")) term = t_nat(a, x > y ? x : y);
                            else if (nm == intern("le"))  term = x <= y ? G_TRUE : G_FALSE;
                            else if (nm == intern("lt"))  term = x <  y ? G_TRUE : G_FALSE;
                            else if (nm == intern("ge"))  term = x >= y ? G_TRUE : G_FALSE;
                            else if (nm == intern("gt"))  term = x >  y ? G_TRUE : G_FALSE;
                        }
                        break;
                    }

                    case T_FIX: {
                        if (fd >= MAX_FIX) die("Fix expansion exceeded limit");
                        Term *fx = t_fix(a, fval->u.func);
                        term = t_app(a, fval->u.func, fx);
                        fd++;
                        break;
                    }

                    default:
                        return t_app(a, fval, arg);   /* stuck */
                }
                break;
            }

            case T_EVAL: {
                Term *q = term->u.ev.q;
                if (q->tag == T_QUOTE)
                    term = t_app(a, q->u.inner, term->u.ev.a);
                else
                    term = t_eval(a, eval_term(a, q, env, fd), term->u.ev.a);
                break;
            }

            case T_FIX: {
                if (fd >= MAX_FIX) die("Fix expansion exceeded limit");
                Term *fx = t_fix(a, term->u.func);
                term = t_app(a, term->u.func, fx);
                fd++;
                break;
            }

            default:
                return term;
        }
    }
}

/* ── sexpr printing (term_to_sexpr port) ── */
void print_term(Term *t) {
    switch (t->tag) {
        case T_VAR: printf("%s", t->u.name); break;
        case T_LAM:
            printf("(lam %s ", t->u.lam.param);
            print_term(t->u.lam.body);
            printf(")");
            break;
        case T_APP:
            printf("(app ");
            print_term(t->u.app.f);
            printf(" ");
            print_term(t->u.app.a);
            printf(")");
            break;
        case T_QUOTE:
            printf("(quote ");
            print_term(t->u.inner);
            printf(")");
            break;
        case T_EVAL:
            printf("(eval ");
            print_term(t->u.ev.q);
            printf(" ");
            print_term(t->u.ev.a);
            printf(")");
            break;
        case T_FIX:
            printf("(fix ");
            print_term(t->u.func);
            printf(")");
            break;
        case T_NAT: printf("%ld", t->u.n); break;
        case T_PRIM: printf("#%s", t->u.name); break;
        case T_PARTIAL:
            printf("(#%s ", t->u.partial.name);
            print_term(t->u.partial.a1);
            printf(")");
            break;
        case T_CONS:
            printf("(cons ");
            print_term(t->u.cons.car);
            printf(" ");
            print_term(t->u.cons.cdr);
            printf(")");
            break;
    }
}
