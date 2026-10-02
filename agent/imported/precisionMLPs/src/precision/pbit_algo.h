/* The expC11 computation (experiments/expC11_true_precision_law/SPEC.md), written once.
 *
 * The including file defines the number type T and the operations
 *   ADD SUB MUL DIV SQRT  -- rounded arithmetic in the working format
 *   NEG ABS LDEXP RINT    -- exact operations
 *   FROMD(x)              -- a double that is already a format value, as T
 *   I2T(n)                -- an integer converted to the format (rounded, like Fortran DBLE)
 *   TOD(a)                -- T to double (exact)
 *   SETFMT(p, emin, emax) -- select the format (emulator) or check it (native types)
 * Every arithmetic step below goes through them, in the order the spec fixes.
 */
#include <stdlib.h>
#include <string.h>

#ifndef STAGE
#define STAGE(k)
#endif

#define PBIT_MAX_COEFFS 64

typedef struct {
    int p, emin, emax;
    /* tanh */
    T invln2, ln2hi, ln2lo, sat;
    /* LAPACK machine parameters of the format */
    T eps, prec, sfmin, huge, safmin, safmax, tsml, tbig, ssml, sbig, epspow, hndrth;
    int d;
    T a[PBIT_MAX_COEFFS];
} Consts;

/* Packed constants (all doubles, each exactly a format value):
 *  0 p, 1 emin, 2 emax, 3 INVLN2, 4 LN2HI, 5 LN2LO, 6 SAT,
 *  7 eps, 8 prec, 9 sfmin, 10 huge, 11 safmin, 12 safmax, 13 tsml, 14 tbig, 15 ssml, 16 sbig,
 *  17 epspow = eps**(-1/8), 18 hndrth = 0.01, 19 d, 20.. a_0 .. a_{d-1}. */
static int load_consts(Consts *k, const double *c) {
    k->p = (int)c[0]; k->emin = (int)c[1]; k->emax = (int)c[2];
    if (SETFMT(k->p, k->emin, k->emax)) return 1;
    k->invln2 = FROMD(c[3]); k->ln2hi = FROMD(c[4]); k->ln2lo = FROMD(c[5]); k->sat = FROMD(c[6]);
    k->eps = FROMD(c[7]); k->prec = FROMD(c[8]); k->sfmin = FROMD(c[9]); k->huge = FROMD(c[10]);
    k->safmin = FROMD(c[11]); k->safmax = FROMD(c[12]);
    k->tsml = FROMD(c[13]); k->tbig = FROMD(c[14]); k->ssml = FROMD(c[15]); k->sbig = FROMD(c[16]);
    k->epspow = FROMD(c[17]); k->hndrth = FROMD(c[18]);
    k->d = (int)c[19];
    if (k->d < 1 || k->d > PBIT_MAX_COEFFS) return 1;
    for (int j = 0; j < k->d; ++j) k->a[j] = FROMD(c[20 + j]);
    return 0;
}

#include "lapack_gelss.h"

static T tanh_p(T z, const Consts *k) {
    const T one = FROMD(1.0), two = FROMD(2.0);
    T a = ABS(z);
    if (a >= k->sat) return z < FROMD(0.0) ? NEG(one) : one;
    T u = NEG(LDEXP(a, 1));
    T kf = RINT(MUL(u, k->invln2));
    int kk = (int)TOD(kf);
    T r = u;
    if (kk != 0) r = SUB(SUB(u, MUL(kf, k->ln2hi)), MUL(kf, k->ln2lo));
    T q = k->a[k->d - 1];
    for (int j = k->d - 2; j >= 0; --j) q = ADD(k->a[j], MUL(r, q));
    T P = MUL(r, q);
    T E = P;
    if (kk != 0) E = ADD(LDEXP(P, kk), SUB(LDEXP(one, kk), one));
    T t = NEG(DIV(E, ADD(E, two)));
    return z < FROMD(0.0) ? NEG(t) : t;
}

/* Elementwise tanh_p, for tests. */
int pbit_tanh(const double *cst, size_t count, const double *z, double *out) {
    Consts k;
    if (load_consts(&k, cst)) return 1;
    for (size_t i = 0; i < count; ++i) out[i] = TOD(tanh_p(FROMD(z[i]), &k));
    return 0;
}

/* Geometry: h = 2 / N, c_j = -1 + j h, gamma = lambda / h. */
static void geometry(double Nd, size_t W, const double *jd, double lamd, T *cen, T *gam) {
    const T two = FROMD(2.0), mone = FROMD(-1.0);
    T h = DIV(two, FROMD(Nd));
    *gam = DIV(FROMD(lamd), h);
    for (size_t j = 0; j < W; ++j) cen[j] = ADD(mone, MUL(FROMD(jd[j]), h));
}

/* Model output at each x: s = bias; s += tanh(gamma (x - c_j)) w_j for j = 0..W-1. */
static void forward(const Consts *k, size_t Me, const double *xed, size_t W, const T *cen, T gam,
                    const T *w, double *fit) {
    for (size_t i = 0; i < Me; ++i) {
        T x = FROMD(xed[i]);
        T s = w[W];
        for (size_t j = 0; j < W; ++j) s = ADD(s, MUL(tanh_p(MUL(gam, SUB(x, cen[j])), k), w[j]));
        fit[i] = TOD(s);
    }
}

/* Evaluate given weights (each a format value, bias last). */
int pbit_eval(const double *cst, size_t Me, const double *xed, double Nd, size_t W,
              const double *jd, double lamd, const double *weights, double *fit) {
    Consts k;
    if (load_consts(&k, cst)) return 1;
    T *cen = malloc(sizeof(T) * W), *w = malloc(sizeof(T) * (W + 1)), gam;
    if (!cen || !w) { free(cen); free(w); return 2; }
    geometry(Nd, W, jd, lamd, cen, &gam);
    for (size_t j = 0; j <= W; ++j) w[j] = FROMD(weights[j]);
    forward(&k, Me, xed, W, cen, gam, w, fit);
    free(cen); free(w);
    return 0;
}

/* Training features, M x W row-major (for tests and the reference-LAPACK comparison). */
int pbit_features(const double *cst, size_t M, const double *xd, double Nd, size_t W,
                  const double *jd, double lamd, double *phi) {
    Consts k;
    if (load_consts(&k, cst)) return 1;
    T *cen = malloc(sizeof(T) * W), gam;
    if (!cen) return 2;
    geometry(Nd, W, jd, lamd, cen, &gam);
    for (size_t i = 0; i < M; ++i) {
        T x = FROMD(xd[i]);
        for (size_t j = 0; j < W; ++j) phi[i * W + j] = TOD(tanh_p(MUL(gam, SUB(x, cen[j])), &k));
    }
    free(cen);
    return 0;
}

/* The ported DGELSS alone on a given matrix (column-major m x n) and right-hand side, for tests.
 * Returns the lp_gelss status. */
int pbit_gelss(const double *cst, int m, int n, const double *ad, const double *bd, int ncut,
               const double *rconds, double *x_out, int *ranks, double *sigma_out) {
    Consts k;
    if (load_consts(&k, cst)) return -10;
    T *a = malloc(sizeof(T) * (size_t)m * n), *b = malloc(sizeof(T) * m), *s = malloc(sizeof(T) * n);
    T *x = malloc(sizeof(T) * (size_t)n * ncut), *rc = malloc(sizeof(T) * ncut);
    if (!a || !b || !s || !x || !rc) { free(a); free(b); free(s); free(x); free(rc); return -4; }
    for (size_t i = 0; i < (size_t)m * n; ++i) a[i] = FROMD(ad[i]);
    for (int i = 0; i < m; ++i) b[i] = FROMD(bd[i]);
    for (int q = 0; q < ncut; ++q) rc[q] = FROMD(rconds[q]);
    int status = lp_gelss(&k, m, n, a, b, s, ncut, rc, x, ranks);
    if (status == 0) {
        for (int i = 0; i < n; ++i) sigma_out[i] = TOD(s[i]);
        for (size_t i = 0; i < (size_t)n * ncut; ++i) x_out[i] = TOD(x[i]);
    }
    free(a); free(b); free(s); free(x); free(rc);
    return status;
}

/* The whole model and solve. Inputs are doubles that are already format values.
 * rconds: ncut DGELSS cutoffs; weights (ncut x n, n = W + 1, bias last), fit (ncut x Me) and
 * ranks (ncut) are filled per cutoff from the one decomposition; sigma (n) the singular values.
 * Returns 0, or 100 + the nonzero DGELSS status (see lp_gelss). */
int pbit_run(const double *cst, size_t M, const double *xd, const double *yd, size_t Me,
             const double *xed, double Nd, size_t W, const double *jd, double lamd, size_t ncut,
             const double *rconds, double *weights, double *fit, int *ranks, double *sigma_out) {
    Consts k;
    if (load_consts(&k, cst)) return 1;
    const size_t n = W + 1;
    if (n > M || ncut < 1) return 2;
    const T one = FROMD(1.0);
    T *A = malloc(sizeof(T) * M * n), *b = malloc(sizeof(T) * M), *cen = malloc(sizeof(T) * W);
    T *s = malloc(sizeof(T) * n), *x = malloc(sizeof(T) * n * ncut), *rc = malloc(sizeof(T) * ncut);
    if (!A || !b || !cen || !s || !x || !rc) {
        free(A); free(b); free(cen); free(s); free(x); free(rc);
        return 3;
    }
    T gam;
    STAGE(1);
    geometry(Nd, W, jd, lamd, cen, &gam);
    for (size_t j = 0; j < W; ++j)
        for (size_t i = 0; i < M; ++i)
            A[i + j * M] = tanh_p(MUL(gam, SUB(FROMD(xd[i]), cen[j])), &k);
    for (size_t i = 0; i < M; ++i) { A[i + W * M] = one; b[i] = FROMD(yd[i]); }
    for (size_t q = 0; q < ncut; ++q) rc[q] = FROMD(rconds[q]);
    STAGE(2);
    int status = lp_gelss(&k, (int)M, (int)n, A, b, s, (int)ncut, rc, x, ranks);
    free(A); free(b);
    if (status == 0) {
        for (size_t i = 0; i < n; ++i) sigma_out[i] = TOD(s[i]);
        for (size_t i = 0; i < n * ncut; ++i) weights[i] = TOD(x[i]);
        STAGE(3);
        for (size_t q = 0; q < ncut; ++q)
            forward(&k, Me, xed, W, cen, gam, x + q * n, fit + q * Me);
    }
    free(cen); free(s); free(x); free(rc);
    return status ? 100 + status : 0;
}
