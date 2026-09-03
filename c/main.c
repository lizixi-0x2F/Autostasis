/* main.c — benchmark driver + fixed cases (oracle pairs for the Python port). */
#include "term.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ── default environment (static, built once) ── */
static Env *make_env(void) {
    static const char *names[] = {
        "add","sub","mult","div","mod","pow","min","max",
        "le","lt","ge","gt","pred","iszero","eq","eq_nat",
        "car","cdr",
        "is_var","is_lam","is_app","is_nat","is_quote","is_eval",
        "is_fix","is_cons",
        "get_body","get_func","get_arg","get_quoted","get_eval_arg",
        "get_fix_func","get_car","get_cdr",
        "mk_nat","mk_lam","mk_app",
    };
    enum { N = sizeof names / sizeof names[0] };
    static Term prims[N];
    static Env envs[N];
    static int done = 0;
    if (!done) {
        for (size_t i = 0; i < N; i++) {
            prims[i].tag = T_PRIM;
            prims[i].u.name = intern(names[i]);
            envs[i].name = intern(names[i]);
            envs[i].val = &prims[i];
            envs[i].parent = (i == 0) ? NULL : &envs[i - 1];
        }
        done = 1;
    }
    return &envs[N - 1];
}

/* ── fib ── */
static Term *build_fib(Arena *a) {
    Term *x = t_var(a, "x");
    Term *self = t_var(a, "self");
    Term *cond = t_app(a, t_app(a, t_prim(a, "le"), x), t_nat(a, 1));
    Term *p = t_app(a, t_prim(a, "pred"), x);
    Term *pp = t_app(a, t_prim(a, "pred"), p);
    Term *add = t_app(a, t_app(a, t_prim(a, "add"),
                                t_app(a, self, p)), t_app(a, self, pp));
    Term *body = t_app(a, t_app(a, cond, x), add);
    return t_fix(a, t_lam(a, "self", t_lam(a, "x", body)));
}

/* evaluate to WHNF: eval_term already loops to weak head normal form */
static Term *deep(Arena *a, Term *t, Env *env) {
    return eval_term(a, t, env, 0);
}

int main(void) {
    Arena ar = {0};
    Env *env = make_env();

    /* ── benchmarks ── */
    {
        Term *fib = build_fib(&ar);
        for (long n = 15; n <= 25; n += 5) {
            double t0 = now_s();
            Term *r = eval_term(&ar, t_app(&ar, fib, t_nat(&ar, n)), env, 0);
            double t1 = now_s();
            printf("B fib(%ld) = %ld  in %.4fs\n", n,
                   r->tag == T_NAT ? r->u.n : -1, t1 - t0);
        }
    }
    {
        long n = 20000;
        double t0 = now_s();
        Term *t = t_nat(&ar, 0);
        for (long i = 0; i < n; i++)
            t = t_app(&ar, t_app(&ar, t_prim(&ar, "add"), t), t_nat(&ar, 1));
        Term *r = deep(&ar, t, env);
        double t1 = now_s();
        printf("B arith chain %ld adds = %ld  in %.4fs\n", n,
               r->tag == T_NAT ? r->u.n : -1, t1 - t0);
    }

    /* ── fixed cases (oracle pairs for src/eval.py) ── */
    #define CASE(name, body) do { printf("%s: ", name); print_term(body); printf("\n"); } while (0)
    {
        Term *two = t_nat(&ar, 2), *three = t_nat(&ar, 3),
             *four = t_nat(&ar, 4), *five = t_nat(&ar, 5), *zero = t_nat(&ar, 0),
             *one = t_nat(&ar, 1), *seven = t_nat(&ar, 7), *ten = t_nat(&ar, 10);

        CASE("add23", deep(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "add"), two), three), env));
        CASE("nested", deep(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "mult"),
                t_app(&ar, t_app(&ar, t_prim(&ar, "add"), two), three)), four), env));
        CASE("monus35", deep(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "sub"), three), five), env));
        CASE("div52", deep(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "div"), five), two), env));
        CASE("div50", deep(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "div"), five), zero), env));
        CASE("pow23", deep(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "pow"), two), three), env));
        CASE("min42", deep(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "min"), four), two), env));
        CASE("le23", deep(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "le"), two), three), env));
        CASE("iszero0", deep(&ar, t_app(&ar, t_prim(&ar, "iszero"), zero), env));
        CASE("pred0", deep(&ar, t_app(&ar, t_prim(&ar, "pred"), zero), env));
        CASE("pred7", deep(&ar, t_app(&ar, t_prim(&ar, "pred"), seven), env));
        /* Church bools */
        CASE("true10", deep(&ar, t_app(&ar, t_app(&ar, G_TRUE, one), zero), env));
        CASE("false10", deep(&ar, t_app(&ar, t_app(&ar, G_FALSE, one), zero), env));
        /* Quote/Eval */
        Term *qadd = t_quote(&ar, t_app(&ar, t_app(&ar, t_prim(&ar, "add"), two), three));
        CASE("eval_qadd", deep(&ar, t_eval(&ar, qadd, zero), env));
        /* introspection */
        Term *lamx = t_quote(&ar, t_lam(&ar, "x", t_var(&ar, "x")));
        CASE("is_lam_lamx", deep(&ar, t_app(&ar, t_prim(&ar, "is_lam"), lamx), env));
        CASE("is_lam_nat", deep(&ar, t_app(&ar, t_prim(&ar, "is_lam"), t_quote(&ar, two)), env));
        CASE("get_body_lamx", deep(&ar, t_app(&ar, t_prim(&ar, "get_body"), lamx), env));
        /* construction */
        CASE("mk_nat7", deep(&ar, t_app(&ar, t_prim(&ar, "mk_nat"), seven), env));
        /* Cons */
        Term *lst = t_cons(&ar, one, t_cons(&ar, two, zero));
        CASE("car_cons", deep(&ar, t_app(&ar, t_prim(&ar, "car"), lst), env));
        CASE("cdr_cons", deep(&ar, t_app(&ar, t_prim(&ar, "cdr"), lst), env));
        /* closure semantics: nested lambda keeps outer binding */
        Term *addxy = t_app(&ar, t_app(&ar, t_prim(&ar, "add"), t_var(&ar, "x")), t_var(&ar, "y"));
        Term *nested_lam = t_app(&ar, t_app(&ar, t_lam(&ar, "x", t_lam(&ar, "y", addxy)), two), three);
        CASE("closure_nested", deep(&ar, nested_lam, env));
        /* quote is opaque: β does not touch the quoted tree */
        Term *qx = t_app(&ar, t_lam(&ar, "x", t_quote(&ar, t_var(&ar, "x"))), five);
        CASE("quote_opaque", deep(&ar, qx, env));
        /* fib(10) via Fix */
        CASE("fib10", deep(&ar, t_app(&ar, build_fib(&ar), ten), env));
    }

    /* ── divergence guard: Fix(λself. self) applied — must hit MAX_FIX ── */
    {
        Arena da = {0};
        Term *self = t_var(&da, "self");
        Term *omega = t_app(&da, t_fix(&da, t_lam(&da, "self", self)),
                            t_nat(&da, 0));
        /* run in a child process to catch the die() cleanly */
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) {
            eval_term(&da, omega, env, 0);
            printf("DIVERGE-NOT-CAUGHT\n");
            _exit(0);
        }
        int st = 0;
        waitpid(pid, &st, 0);
        printf("divergence: child exit %s\n",
               (WIFEXITED(st) && WEXITSTATUS(st) == 1)
                   ? "caught (MAX_FIX guard)" : "unexpected");
    }

    return 0;
}
