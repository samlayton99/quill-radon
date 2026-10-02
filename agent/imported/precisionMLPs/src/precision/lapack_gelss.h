/* Reference LAPACK 3.12.1 DGELSS (the M >= N path) and every routine it reaches, ported statement
 * by statement to the rounding macros of pbit_algo.h. Source: netlib Reference-LAPACK v3.12.1
 * (SRC/ and BLAS/SRC/). Configuration: unblocked (ILAENV block size 1), so DGEQRF, DORMQR,
 * DGEBRD and DORGLQ run their level-2 kernels DGEQR2, DORM2R, DGEBD2 and DORGL2.
 *
 * Every floating-point operation of the Fortran appears here as one macro call, in the Fortran
 * evaluation order (left to right, parentheses respected, no contraction). Fortran 1-based
 * indexing is kept through the IX macro. Branches unreachable from DGELSS with M >= N and
 * NRHS = 1 (lower bidiagonal, NRU > 0, TOL < 0) are omitted and asserted instead.
 *
 * Machine parameters come from the format (see Consts): DLAMCH('E') = eps = 2^-p,
 * DLAMCH('P') = prec = 2^(1-p), DLAMCH('S') = sfmin, DLAMCH('O') = huge, LA_CONSTANTS safmin and
 * safmax, and the DNRM2 thresholds tsml, tbig, ssml, sbig.
 */
#include <math.h>

#define IX(i, j, ld) ((size_t)((i) - 1) + (size_t)((j) - 1) * (size_t)(ld))

/* Fortran MAX/MIN as gfortran evaluates them (first argument kept on ties); SIGN(a, b) = |a|
   with the sign bit of b (gfortran -fsign-zero). */
static inline T lp_max(T a, T b) { return b > a ? b : a; }
static inline T lp_min(T a, T b) { return b < a ? b : a; }
/* the sign bit of b, by copysign (exact; no conversion to another floating kind) */
static inline int lp_signbit(T b) { return copysign(TOD(FROMD(1.0)), TOD(b)) < TOD(FROMD(0.0)); }
static inline T lp_sign(T a, T b) { T aa = ABS(a); return lp_signbit(b) ? NEG(aa) : aa; }

/* ---------------------------------------------------------------- level-1/2 BLAS (BLAS/SRC) */

static void lp_scal(int n, T da, T *x, int incx) {
    if (n <= 0 || incx <= 0 || da == FROMD(1.0)) return;
    for (int i = 0; i < n * incx; i += incx) x[i] = MUL(da, x[i]);
}

static void lp_axpy(int n, T da, const T *x, int incx, T *y, int incy) {
    if (n <= 0) return;
    if (da == FROMD(0.0)) return;
    int ix = 0, iy = 0;
    if (incx < 0) ix = (-n + 1) * incx;
    if (incy < 0) iy = (-n + 1) * incy;
    for (int i = 0; i < n; ++i) {
        y[iy] = ADD(y[iy], MUL(da, x[ix]));
        ix += incx;
        iy += incy;
    }
}

static void lp_gemv(char trans, int m, int n, T alpha, const T *a, int lda, const T *x, int incx,
                    T beta, T *y, int incy) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (m == 0 || n == 0 || (alpha == zero && beta == one)) return;
    int lenx = trans == 'N' ? n : m, leny = trans == 'N' ? m : n;
    int kx = incx > 0 ? 1 : 1 - (lenx - 1) * incx;
    int ky = incy > 0 ? 1 : 1 - (leny - 1) * incy;
    if (beta != one) {
        int iy = ky;
        for (int i = 1; i <= leny; ++i) {
            y[iy - 1] = beta == zero ? zero : MUL(beta, y[iy - 1]);
            iy += incy;
        }
    }
    if (alpha == zero) return;
    if (trans == 'N') {
        int jx = kx;
        for (int j = 1; j <= n; ++j) {
            T temp = MUL(alpha, x[jx - 1]);
            int iy = ky;
            for (int i = 1; i <= m; ++i) {
                y[iy - 1] = ADD(y[iy - 1], MUL(temp, a[IX(i, j, lda)]));
                iy += incy;
            }
            jx += incx;
        }
    } else {
        int jy = ky;
        for (int j = 1; j <= n; ++j) {
            T temp = zero;
            int ix = kx;
            for (int i = 1; i <= m; ++i) {
                temp = ADD(temp, MUL(a[IX(i, j, lda)], x[ix - 1]));
                ix += incx;
            }
            y[jy - 1] = ADD(y[jy - 1], MUL(alpha, temp));
            jy += incy;
        }
    }
}

static void lp_ger(int m, int n, T alpha, const T *x, int incx, const T *y, int incy, T *a,
                   int lda) {
    const T zero = FROMD(0.0);
    if (m == 0 || n == 0 || alpha == zero) return;
    int jy = incy > 0 ? 1 : 1 - (n - 1) * incy;
    int kx = incx > 0 ? 1 : 1 - (m - 1) * incx;
    for (int j = 1; j <= n; ++j) {
        if (y[jy - 1] != zero) {
            T temp = MUL(alpha, y[jy - 1]);
            int ix = kx;
            for (int i = 1; i <= m; ++i) {
                a[IX(i, j, lda)] = ADD(a[IX(i, j, lda)], MUL(x[ix - 1], temp));
                ix += incx;
            }
        }
        jy += incy;
    }
}

static void lp_rot(int n, T *x, int incx, T *y, int incy, T c, T s) {
    if (n <= 0) return;
    int ix = 0, iy = 0;
    if (incx < 0) ix = (-n + 1) * incx;
    if (incy < 0) iy = (-n + 1) * incy;
    for (int i = 0; i < n; ++i) {
        T dtemp = ADD(MUL(c, x[ix]), MUL(s, y[iy]));
        y[iy] = SUB(MUL(c, y[iy]), MUL(s, x[ix]));
        x[ix] = dtemp;
        ix += incx;
        iy += incy;
    }
}

static void lp_swap(int n, T *x, int incx, T *y, int incy) {
    if (n <= 0) return;
    int ix = 0, iy = 0;
    if (incx < 0) ix = (-n + 1) * incx;
    if (incy < 0) iy = (-n + 1) * incy;
    for (int i = 0; i < n; ++i) {
        T dtemp = x[ix];
        x[ix] = y[iy];
        y[iy] = dtemp;
        ix += incx;
        iy += incy;
    }
}

/* DNRM2 (BLAS/SRC/dnrm2.f90, Blue's algorithm). */
static T lp_nrm2(const Consts *k, int n, const T *x, int incx) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (n <= 0) return zero;
    T scl, sumsq, asml = zero, amed = zero, abig = zero;
    int notbig = 1;
    int ix = 1;
    if (incx < 0) ix = 1 - (n - 1) * incx;
    for (int i = 1; i <= n; ++i) {
        T ax = ABS(x[ix - 1]);
        if (ax > k->tbig) {
            T t = MUL(ax, k->sbig);
            abig = ADD(abig, MUL(t, t));
            notbig = 0;
        } else if (ax < k->tsml) {
            if (notbig) {
                T t = MUL(ax, k->ssml);
                asml = ADD(asml, MUL(t, t));
            }
        } else {
            amed = ADD(amed, MUL(ax, ax));
        }
        ix += incx;
    }
    if (abig > zero) {
        if (amed > zero || amed > k->huge || amed != amed)
            abig = ADD(abig, MUL(MUL(amed, k->sbig), k->sbig));
        scl = DIV(one, k->sbig);
        sumsq = abig;
    } else if (asml > zero) {
        if (amed > zero || amed > k->huge || amed != amed) {
            amed = SQRT(amed);
            asml = DIV(SQRT(asml), k->ssml);
            T ymin, ymax;
            if (asml > amed) { ymin = amed; ymax = asml; } else { ymin = asml; ymax = amed; }
            scl = one;
            T q = DIV(ymin, ymax);
            sumsq = MUL(MUL(ymax, ymax), ADD(one, MUL(q, q)));
        } else {
            scl = DIV(one, k->ssml);
            sumsq = asml;
        }
    } else {
        scl = one;
        sumsq = amed;
    }
    return MUL(scl, SQRT(sumsq));
}

/* ---------------------------------------------------------------- LAPACK auxiliaries (SRC) */

static T lp_lapy2(const Consts *k, T x, T y) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    T xabs = ABS(x), yabs = ABS(y);
    T w = lp_max(xabs, yabs), z = lp_min(xabs, yabs);
    if (z == zero || w > k->huge) return w;
    T q = DIV(z, w);
    return MUL(w, SQRT(ADD(one, MUL(q, q))));
}

static int lp_iladlc(int m, int n, const T *a, int lda) {
    const T zero = FROMD(0.0);
    if (n == 0) return n;
    if (a[IX(1, n, lda)] != zero || a[IX(m, n, lda)] != zero) return n;
    for (int c = n; c >= 1; --c)
        for (int i = 1; i <= m; ++i)
            if (a[IX(i, c, lda)] != zero) return c;
    return 0;
}

static int lp_iladlr(int m, int n, const T *a, int lda) {
    const T zero = FROMD(0.0);
    if (m == 0) return m;
    if (a[IX(m, 1, lda)] != zero || a[IX(m, n, lda)] != zero) return m;
    int r = 0;
    for (int j = 1; j <= n; ++j) {
        int i = m;
        while (a[IX(i > 1 ? i : 1, j, lda)] == zero && i >= 1) --i;
        if (i > r) r = i;
    }
    return r;
}

static void lp_larfg(const Consts *k, int n, T *alpha, T *x, int incx, T *tau) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (n <= 1) { *tau = zero; return; }
    T xnorm = lp_nrm2(k, n - 1, x, incx);
    if (xnorm == zero) { *tau = zero; return; }
    T beta = NEG(lp_sign(lp_lapy2(k, *alpha, xnorm), *alpha));
    T safmin = DIV(k->sfmin, k->eps);
    int knt = 0;
    if (ABS(beta) < safmin) {
        T rsafmn = DIV(one, safmin);
        do {
            ++knt;
            lp_scal(n - 1, rsafmn, x, incx);
            beta = MUL(beta, rsafmn);
            *alpha = MUL(*alpha, rsafmn);
        } while (ABS(beta) < safmin && knt < 20);
        xnorm = lp_nrm2(k, n - 1, x, incx);
        beta = NEG(lp_sign(lp_lapy2(k, *alpha, xnorm), *alpha));
    }
    *tau = DIV(SUB(beta, *alpha), beta);
    lp_scal(n - 1, DIV(one, SUB(*alpha, beta)), x, incx);
    for (int j = 1; j <= knt; ++j) beta = MUL(beta, safmin);
    *alpha = beta;
}

/* DLARF1F: apply H = I - tau v v^T with v(1) = 1 implicit (v(1) is never read). */
static void lp_larf1f(char side, int m, int n, const T *v, int incv, T tau, T *c, int ldc,
                      T *work) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    int applyleft = side == 'L';
    int lastv = 1, lastc = 0;
    if (tau != zero) {
        lastv = applyleft ? m : n;
        int i = incv > 0 ? 1 + (lastv - 1) * incv : 1;
        while (lastv > 1 && v[i - 1] == zero) { --lastv; i -= incv; }
        lastc = applyleft ? lp_iladlc(lastv, n, c, ldc) : lp_iladlr(m, lastv, c, ldc);
    }
    if (lastc == 0) return;
    if (applyleft) {
        if (lastv == 1) {
            lp_scal(lastc, SUB(one, tau), c, ldc);
        } else {
            lp_gemv('T', lastv - 1, lastc, one, c + 1, ldc, v + incv, incv, zero, work, 1);
            lp_axpy(lastc, one, c, ldc, work, 1);
            lp_axpy(lastc, NEG(tau), work, 1, c, ldc);
            lp_ger(lastv - 1, lastc, NEG(tau), v + incv, incv, work, 1, c + 1, ldc);
        }
    } else {
        if (lastv == 1) {
            lp_scal(lastc, SUB(one, tau), c, 1);
        } else {
            lp_gemv('N', lastc, lastv - 1, one, c + ldc, ldc, v + incv, incv, zero, work, 1);
            lp_axpy(lastc, one, c, 1, work, 1);
            lp_axpy(lastc, NEG(tau), work, 1, c, 1);
            lp_ger(lastc, lastv - 1, NEG(tau), work, 1, v + incv, incv, c + ldc, ldc);
        }
    }
}

static void lp_geqr2(const Consts *k, int m, int n, T *a, int lda, T *tau, T *work) {
    int kk = m < n ? m : n;
    for (int i = 1; i <= kk; ++i) {
        lp_larfg(k, m - i + 1, &a[IX(i, i, lda)], &a[IX(i + 1 < m ? i + 1 : m, i, lda)], 1,
                 &tau[i - 1]);
        if (i < n)
            lp_larf1f('L', m - i + 1, n - i, &a[IX(i, i, lda)], 1, tau[i - 1],
                      &a[IX(i, i + 1, lda)], lda, work);
    }
}

/* DORM2R, SIDE = 'L' (the only side DGELSS uses). */
static void lp_orm2r_left(char trans, int m, int n, int k, const T *a, int lda, const T *tau,
                          T *c, int ldc, T *work) {
    if (m == 0 || n == 0 || k == 0) return;
    int notran = trans == 'N';
    int i1, i2, i3;
    if (!notran) { i1 = 1; i2 = k; i3 = 1; } else { i1 = k; i2 = 1; i3 = -1; }
    for (int i = i1; i3 > 0 ? i <= i2 : i >= i2; i += i3)
        lp_larf1f('L', m - i + 1, n, &a[IX(i, i, lda)], 1, tau[i - 1], &c[IX(i, 1, ldc)], ldc,
                  work);
}

/* DGEBD2, M >= N (upper bidiagonal). */
static void lp_gebd2(const Consts *k, int m, int n, T *a, int lda, T *d, T *e, T *tauq, T *taup,
                     T *work) {
    for (int i = 1; i <= n; ++i) {
        lp_larfg(k, m - i + 1, &a[IX(i, i, lda)], &a[IX(i + 1 < m ? i + 1 : m, i, lda)], 1,
                 &tauq[i - 1]);
        d[i - 1] = a[IX(i, i, lda)];
        if (i < n)
            lp_larf1f('L', m - i + 1, n - i, &a[IX(i, i, lda)], 1, tauq[i - 1],
                      &a[IX(i, i + 1, lda)], lda, work);
        if (i < n) {
            lp_larfg(k, n - i, &a[IX(i, i + 1, lda)], &a[IX(i, i + 2 < n ? i + 2 : n, lda)], lda,
                     &taup[i - 1]);
            e[i - 1] = a[IX(i, i + 1, lda)];
            lp_larf1f('R', m - i, n - i, &a[IX(i, i + 1, lda)], lda, taup[i - 1],
                      &a[IX(i + 1, i + 1, lda)], lda, work);
        } else {
            taup[i - 1] = FROMD(0.0);
        }
    }
}

static void lp_orgl2(int m, int n, int k, T *a, int lda, const T *tau, T *work) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (m <= 0) return;
    if (k < m) {
        for (int j = 1; j <= n; ++j) {
            for (int l = k + 1; l <= m; ++l) a[IX(l, j, lda)] = zero;
            if (j > k && j <= m) a[IX(j, j, lda)] = one;
        }
    }
    for (int i = k; i >= 1; --i) {
        if (i < n) {
            if (i < m)
                lp_larf1f('R', m - i, n - i + 1, &a[IX(i, i, lda)], lda, tau[i - 1],
                          &a[IX(i + 1, i, lda)], lda, work);
            lp_scal(n - i, NEG(tau[i - 1]), &a[IX(i, i + 1, lda)], lda);
        }
        a[IX(i, i, lda)] = SUB(one, tau[i - 1]);
        for (int l = 1; l <= i - 1; ++l) a[IX(i, l, lda)] = zero;
    }
}

/* DORGBR with VECT = 'P', M = N = K (the call DGELSS makes); DORGLQ -> DORGL2 unblocked. */
static void lp_orgbr_p(int n, T *a, int lda, const T *tau, T *work) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (n == 0) return;
    a[IX(1, 1, lda)] = one;
    for (int i = 2; i <= n; ++i) a[IX(i, 1, lda)] = zero;
    for (int j = 2; j <= n; ++j) {
        for (int i = j - 1; i >= 2; --i) a[IX(i, j, lda)] = a[IX(i - 1, j, lda)];
        a[IX(1, j, lda)] = zero;
    }
    if (n > 1) lp_orgl2(n - 1, n - 1, n - 1, &a[IX(2, 2, lda)], lda, tau, work);
}

static void lp_lartg(const Consts *k, T f, T g, T *c, T *s, T *r) {
    const T zero = FROMD(0.0), one = FROMD(1.0), two = FROMD(2.0);
    T rtmin = SQRT(k->safmin);
    T rtmax = SQRT(DIV(k->safmax, two));
    T f1 = ABS(f), g1 = ABS(g);
    if (g == zero) {
        *c = one; *s = zero; *r = f;
    } else if (f == zero) {
        *c = zero; *s = lp_sign(one, g); *r = g1;
    } else if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
        T d = SQRT(ADD(MUL(f, f), MUL(g, g)));
        *c = DIV(f1, d);
        *r = lp_sign(d, f);
        *s = DIV(g, *r);
    } else {
        T u = lp_min(k->safmax, lp_max(lp_max(k->safmin, f1), g1));
        T fs = DIV(f, u), gs = DIV(g, u);
        T d = SQRT(ADD(MUL(fs, fs), MUL(gs, gs)));
        *c = DIV(ABS(fs), d);
        *r = lp_sign(d, f);
        *s = DIV(gs, *r);
        *r = MUL(*r, u);
    }
}

static void lp_las2(T f, T g, T h, T *ssmin, T *ssmax) {
    const T zero = FROMD(0.0), one = FROMD(1.0), two = FROMD(2.0);
    T fa = ABS(f), ga = ABS(g), ha = ABS(h);
    T fhmn = lp_min(fa, ha), fhmx = lp_max(fa, ha);
    if (fhmn == zero) {
        *ssmin = zero;
        if (fhmx == zero) {
            *ssmax = ga;
        } else {
            T q = DIV(lp_min(fhmx, ga), lp_max(fhmx, ga));
            *ssmax = MUL(lp_max(fhmx, ga), SQRT(ADD(one, MUL(q, q))));
        }
    } else if (ga < fhmx) {
        T as = ADD(one, DIV(fhmn, fhmx));
        T at = DIV(SUB(fhmx, fhmn), fhmx);
        T q = DIV(ga, fhmx);
        T au = MUL(q, q);
        T c = DIV(two, ADD(SQRT(ADD(MUL(as, as), au)), SQRT(ADD(MUL(at, at), au))));
        *ssmin = MUL(fhmn, c);
        *ssmax = DIV(fhmx, c);
    } else {
        T au = DIV(fhmx, ga);
        if (au == zero) {
            *ssmin = DIV(MUL(fhmn, fhmx), ga);
            *ssmax = ga;
        } else {
            T as = ADD(one, DIV(fhmn, fhmx));
            T at = DIV(SUB(fhmx, fhmn), fhmx);
            T q1 = MUL(as, au), q2 = MUL(at, au);
            T c = DIV(one, ADD(SQRT(ADD(one, MUL(q1, q1))), SQRT(ADD(one, MUL(q2, q2)))));
            T sm = MUL(MUL(fhmn, c), au);
            *ssmin = ADD(sm, sm);
            *ssmax = DIV(ga, ADD(c, c));
        }
    }
}

static void lp_lasv2(const Consts *k, T f, T g, T h, T *ssmin, T *ssmax, T *snr, T *csr, T *snl,
                     T *csl) {
    const T zero = FROMD(0.0), half = FROMD(0.5), one = FROMD(1.0), two = FROMD(2.0),
            four = FROMD(4.0);
    T ft = f, fa = ABS(ft), ht = h, ha = ABS(h);
    int pmax = 1;
    int swap = ha > fa;
    if (swap) {
        pmax = 3;
        T temp = ft; ft = ht; ht = temp;
        temp = fa; fa = ha; ha = temp;
    }
    T gt = g, ga = ABS(gt);
    T clt = zero, crt = zero, slt = zero, srt = zero;
    if (ga == zero) {
        *ssmin = ha; *ssmax = fa;
        clt = one; crt = one; slt = zero; srt = zero;
    } else {
        int gasmal = 1;
        if (ga > fa) {
            pmax = 2;
            if (DIV(fa, ga) < k->eps) {
                gasmal = 0;
                *ssmax = ga;
                if (ha > one) *ssmin = DIV(fa, DIV(ga, ha));
                else *ssmin = MUL(DIV(fa, ga), ha);
                clt = one;
                slt = DIV(ht, gt);
                srt = one;
                crt = DIV(ft, gt);
            }
        }
        if (gasmal) {
            T d = SUB(fa, ha), l, m, t, mm, tt, s, r, a;
            if (d == fa) l = one; else l = DIV(d, fa);
            m = DIV(gt, ft);
            t = SUB(two, l);
            mm = MUL(m, m);
            tt = MUL(t, t);
            s = SQRT(ADD(tt, mm));
            if (l == zero) r = ABS(m); else r = SQRT(ADD(MUL(l, l), mm));
            a = MUL(half, ADD(s, r));
            *ssmin = DIV(ha, a);
            *ssmax = MUL(fa, a);
            if (mm == zero) {
                if (l == zero) t = MUL(lp_sign(two, ft), lp_sign(one, gt));
                else t = ADD(DIV(gt, lp_sign(d, ft)), DIV(m, t));
            } else {
                t = MUL(ADD(DIV(m, ADD(s, t)), DIV(m, ADD(r, l))), ADD(one, a));
            }
            l = SQRT(ADD(MUL(t, t), four));
            crt = DIV(two, l);
            srt = DIV(t, l);
            clt = DIV(ADD(crt, MUL(srt, m)), a);
            slt = DIV(MUL(DIV(ht, ft), srt), a);
        }
    }
    if (swap) { *csl = srt; *snl = crt; *csr = slt; *snr = clt; }
    else { *csl = clt; *snl = slt; *csr = crt; *snr = srt; }
    T tsign = one;
    if (pmax == 1) tsign = MUL(MUL(lp_sign(one, *csr), lp_sign(one, *csl)), lp_sign(one, f));
    if (pmax == 2) tsign = MUL(MUL(lp_sign(one, *snr), lp_sign(one, *csl)), lp_sign(one, g));
    if (pmax == 3) tsign = MUL(MUL(lp_sign(one, *snr), lp_sign(one, *snl)), lp_sign(one, h));
    *ssmax = lp_sign(*ssmax, tsign);
    *ssmin = lp_sign(*ssmin, MUL(MUL(tsign, lp_sign(one, f)), lp_sign(one, h)));
}

/* DLASR with SIDE = 'L', PIVOT = 'V'; c[0] is C(1), s[0] is S(1). */
static void lp_lasr_lv(char direct, int m, int n, const T *c, const T *s, T *a, int lda) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (m == 0 || n == 0) return;
    int j0 = direct == 'F' ? 1 : m - 1, j1 = direct == 'F' ? m - 1 : 1, dj = direct == 'F' ? 1 : -1;
    for (int j = j0; dj > 0 ? j <= j1 : j >= j1; j += dj) {
        T ctemp = c[j - 1], stemp = s[j - 1];
        if (ctemp != one || stemp != zero) {
            for (int i = 1; i <= n; ++i) {
                T temp = a[IX(j + 1, i, lda)];
                a[IX(j + 1, i, lda)] = SUB(MUL(ctemp, temp), MUL(stemp, a[IX(j, i, lda)]));
                a[IX(j, i, lda)] = ADD(MUL(stemp, temp), MUL(ctemp, a[IX(j, i, lda)]));
            }
        }
    }
}

/* DRSCL: x := x / sa, done as DLAMCH-safe multiplications. */
static void lp_rscl(const Consts *k, int n, T sa, T *sx, int incx) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    if (n <= 0) return;
    T smlnum = k->sfmin;
    T bignum = DIV(one, smlnum);
    T cden = sa, cnum = one, mul;
    int done;
    do {
        T cden1 = MUL(cden, smlnum);
        T cnum1 = DIV(cnum, bignum);
        if (ABS(cden1) > ABS(cnum) && cnum != zero) {
            mul = smlnum; done = 0; cden = cden1;
        } else if (ABS(cnum1) > ABS(cden)) {
            mul = bignum; done = 0; cnum = cnum1;
        } else {
            mul = DIV(cnum, cden); done = 1;
        }
        lp_scal(n, mul, sx, incx);
    } while (!done);
}

/* DBDSQR with UPLO = 'U', NRU = 0 (VT receives NCVT columns, C receives NCC columns).
   d, e, work are 0-based arrays; the code indexes them 1-based through the D, E, WK macros.
   Returns INFO (0, or the number of unconverged superdiagonals). */
static int lp_bdsqr(const Consts *k, int n, int ncvt, int ncc, T *d_, T *e_, T *vt, int ldvt,
                    T *c, int ldc, T *work_) {
#define D(i) d_[(i) - 1]
#define E(i) e_[(i) - 1]
#define WK(i) work_[(i) - 1]
    const T zero = FROMD(0.0), one = FROMD(1.0), negone = NEG(FROMD(1.0)), ten = I2T(10),
            hndrd = I2T(100);
    const int maxitr = 6;
    int info = 0;
    if (n == 0) return 0;
    int nm1 = 0, nm12 = 0, nm13 = 0, idir = 0, i, j, ll = 0, lll, m, iter, iterdivn, maxitdivn,
        oldll, oldm, isub;
    T eps, unfl, tolmul, tol, smax, smin, sminoa, mu, thresh, abss, abse, sigmn, sigmx, sinr,
        cosr, sinl, cosl, shift, r, sll, f, g, h, cs, sn, oldcs, oldsn = zero;
    if (n == 1) goto L160;
    nm1 = n - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    idir = 0;
    eps = k->eps;
    unfl = k->sfmin;
    tolmul = lp_max(ten, lp_min(hndrd, k->epspow));
    tol = MUL(tolmul, eps);
    smax = zero;
    for (i = 1; i <= n; ++i) smax = lp_max(smax, ABS(D(i)));
    for (i = 1; i <= n - 1; ++i) smax = lp_max(smax, ABS(E(i)));
    smin = zero;
    /* TOL >= 0: relative accuracy */
    sminoa = ABS(D(1));
    if (sminoa == zero) goto L50;
    mu = sminoa;
    for (i = 2; i <= n; ++i) {
        mu = MUL(ABS(D(i)), DIV(mu, ADD(mu, ABS(E(i - 1)))));
        sminoa = lp_min(sminoa, mu);
        if (sminoa == zero) goto L50;
    }
L50:
    sminoa = DIV(sminoa, SQRT(I2T(n)));
    thresh = lp_max(MUL(tol, sminoa), MUL(I2T(maxitr), MUL(I2T(n), MUL(I2T(n), unfl))));
    maxitdivn = maxitr * n;
    iterdivn = 0;
    iter = -1;
    oldll = -1;
    oldm = -1;
    m = n;
L60:
    if (m <= 1) goto L160;
    if (iter >= n) {
        iter = iter - n;
        iterdivn = iterdivn + 1;
        if (iterdivn >= maxitdivn) goto L200;
    }
    smax = ABS(D(m));
    for (lll = 1; lll <= m - 1; ++lll) {
        ll = m - lll;
        abss = ABS(D(ll));
        abse = ABS(E(ll));
        if (abse <= thresh) goto L80;
        smax = lp_max(lp_max(smax, abss), abse);
    }
    ll = 0;
    goto L90;
L80:
    E(ll) = zero;
    if (ll == m - 1) {
        m = m - 1;
        goto L60;
    }
L90:
    ll = ll + 1;
    if (ll == m - 1) {
        lp_lasv2(k, D(m - 1), E(m - 1), D(m), &sigmn, &sigmx, &sinr, &cosr, &sinl, &cosl);
        D(m - 1) = sigmx;
        E(m - 1) = zero;
        D(m) = sigmn;
        if (ncvt > 0) lp_rot(ncvt, &vt[IX(m - 1, 1, ldvt)], ldvt, &vt[IX(m, 1, ldvt)], ldvt, cosr, sinr);
        if (ncc > 0) lp_rot(ncc, &c[IX(m - 1, 1, ldc)], ldc, &c[IX(m, 1, ldc)], ldc, cosl, sinl);
        m = m - 2;
        goto L60;
    }
    if (ll > oldm || m < oldll) {
        if (ABS(D(ll)) >= ABS(D(m))) idir = 1;
        else idir = 2;
    }
    if (idir == 1) {
        if (ABS(E(m - 1)) <= MUL(ABS(tol), ABS(D(m)))) {
            E(m - 1) = zero;
            goto L60;
        }
        mu = ABS(D(ll));
        smin = mu;
        for (lll = ll; lll <= m - 1; ++lll) {
            if (ABS(E(lll)) <= MUL(tol, mu)) {
                E(lll) = zero;
                goto L60;
            }
            mu = MUL(ABS(D(lll + 1)), DIV(mu, ADD(mu, ABS(E(lll)))));
            smin = lp_min(smin, mu);
        }
    } else {
        if (ABS(E(ll)) <= MUL(ABS(tol), ABS(D(ll)))) {
            E(ll) = zero;
            goto L60;
        }
        mu = ABS(D(m));
        smin = mu;
        for (lll = m - 1; lll >= ll; --lll) {
            if (ABS(E(lll)) <= MUL(tol, mu)) {
                E(lll) = zero;
                goto L60;
            }
            mu = MUL(ABS(D(lll)), DIV(mu, ADD(mu, ABS(E(lll)))));
            smin = lp_min(smin, mu);
        }
    }
    oldll = ll;
    oldm = m;
    if (MUL(MUL(I2T(n), tol), DIV(smin, smax)) <= lp_max(eps, MUL(k->hndrth, tol))) {
        shift = zero;
    } else {
        if (idir == 1) {
            sll = ABS(D(ll));
            lp_las2(D(m - 1), E(m - 1), D(m), &shift, &r);
        } else {
            sll = ABS(D(m));
            lp_las2(D(ll), E(ll), D(ll + 1), &shift, &r);
        }
        if (sll > zero) {
            T q = DIV(shift, sll);
            if (MUL(q, q) < eps) shift = zero;
        }
    }
    iter = iter + m - ll;
    if (shift == zero) {
        if (idir == 1) {
            cs = one;
            oldcs = one;
            for (i = ll; i <= m - 1; ++i) {
                lp_lartg(k, MUL(D(i), cs), E(i), &cs, &sn, &r);
                if (i > ll) E(i - 1) = MUL(oldsn, r);
                lp_lartg(k, MUL(oldcs, r), MUL(D(i + 1), sn), &oldcs, &oldsn, &D(i));
                WK(i - ll + 1) = cs;
                WK(i - ll + 1 + nm1) = sn;
                WK(i - ll + 1 + nm12) = oldcs;
                WK(i - ll + 1 + nm13) = oldsn;
            }
            h = MUL(D(m), cs);
            D(m) = MUL(h, oldcs);
            E(m - 1) = MUL(h, oldsn);
            if (ncvt > 0) lp_lasr_lv('F', m - ll + 1, ncvt, &WK(1), &WK(n), &vt[IX(ll, 1, ldvt)], ldvt);
            if (ncc > 0) lp_lasr_lv('F', m - ll + 1, ncc, &WK(nm12 + 1), &WK(nm13 + 1), &c[IX(ll, 1, ldc)], ldc);
            if (ABS(E(m - 1)) <= thresh) E(m - 1) = zero;
        } else {
            cs = one;
            oldcs = one;
            for (i = m; i >= ll + 1; --i) {
                lp_lartg(k, MUL(D(i), cs), E(i - 1), &cs, &sn, &r);
                if (i < m) E(i) = MUL(oldsn, r);
                lp_lartg(k, MUL(oldcs, r), MUL(D(i - 1), sn), &oldcs, &oldsn, &D(i));
                WK(i - ll) = cs;
                WK(i - ll + nm1) = NEG(sn);
                WK(i - ll + nm12) = oldcs;
                WK(i - ll + nm13) = NEG(oldsn);
            }
            h = MUL(D(ll), cs);
            D(ll) = MUL(h, oldcs);
            E(ll) = MUL(h, oldsn);
            if (ncvt > 0) lp_lasr_lv('B', m - ll + 1, ncvt, &WK(nm12 + 1), &WK(nm13 + 1), &vt[IX(ll, 1, ldvt)], ldvt);
            if (ncc > 0) lp_lasr_lv('B', m - ll + 1, ncc, &WK(1), &WK(n), &c[IX(ll, 1, ldc)], ldc);
            if (ABS(E(ll)) <= thresh) E(ll) = zero;
        }
    } else {
        if (idir == 1) {
            f = MUL(SUB(ABS(D(ll)), shift), ADD(lp_sign(one, D(ll)), DIV(shift, D(ll))));
            g = E(ll);
            for (i = ll; i <= m - 1; ++i) {
                lp_lartg(k, f, g, &cosr, &sinr, &r);
                if (i > ll) E(i - 1) = r;
                f = ADD(MUL(cosr, D(i)), MUL(sinr, E(i)));
                E(i) = SUB(MUL(cosr, E(i)), MUL(sinr, D(i)));
                g = MUL(sinr, D(i + 1));
                D(i + 1) = MUL(cosr, D(i + 1));
                lp_lartg(k, f, g, &cosl, &sinl, &r);
                D(i) = r;
                f = ADD(MUL(cosl, E(i)), MUL(sinl, D(i + 1)));
                D(i + 1) = SUB(MUL(cosl, D(i + 1)), MUL(sinl, E(i)));
                if (i < m - 1) {
                    g = MUL(sinl, E(i + 1));
                    E(i + 1) = MUL(cosl, E(i + 1));
                }
                WK(i - ll + 1) = cosr;
                WK(i - ll + 1 + nm1) = sinr;
                WK(i - ll + 1 + nm12) = cosl;
                WK(i - ll + 1 + nm13) = sinl;
            }
            E(m - 1) = f;
            if (ncvt > 0) lp_lasr_lv('F', m - ll + 1, ncvt, &WK(1), &WK(n), &vt[IX(ll, 1, ldvt)], ldvt);
            if (ncc > 0) lp_lasr_lv('F', m - ll + 1, ncc, &WK(nm12 + 1), &WK(nm13 + 1), &c[IX(ll, 1, ldc)], ldc);
            if (ABS(E(m - 1)) <= thresh) E(m - 1) = zero;
        } else {
            f = MUL(SUB(ABS(D(m)), shift), ADD(lp_sign(one, D(m)), DIV(shift, D(m))));
            g = E(m - 1);
            for (i = m; i >= ll + 1; --i) {
                lp_lartg(k, f, g, &cosr, &sinr, &r);
                if (i < m) E(i) = r;
                f = ADD(MUL(cosr, D(i)), MUL(sinr, E(i - 1)));
                E(i - 1) = SUB(MUL(cosr, E(i - 1)), MUL(sinr, D(i)));
                g = MUL(sinr, D(i - 1));
                D(i - 1) = MUL(cosr, D(i - 1));
                lp_lartg(k, f, g, &cosl, &sinl, &r);
                D(i) = r;
                f = ADD(MUL(cosl, E(i - 1)), MUL(sinl, D(i - 1)));
                D(i - 1) = SUB(MUL(cosl, D(i - 1)), MUL(sinl, E(i - 1)));
                if (i > ll + 1) {
                    g = MUL(sinl, E(i - 2));
                    E(i - 2) = MUL(cosl, E(i - 2));
                }
                WK(i - ll) = cosr;
                WK(i - ll + nm1) = NEG(sinr);
                WK(i - ll + nm12) = cosl;
                WK(i - ll + nm13) = NEG(sinl);
            }
            E(ll) = f;
            if (ABS(E(ll)) <= thresh) E(ll) = zero;
            if (ncvt > 0) lp_lasr_lv('B', m - ll + 1, ncvt, &WK(nm12 + 1), &WK(nm13 + 1), &vt[IX(ll, 1, ldvt)], ldvt);
            if (ncc > 0) lp_lasr_lv('B', m - ll + 1, ncc, &WK(1), &WK(n), &c[IX(ll, 1, ldc)], ldc);
        }
    }
    goto L60;
L160:
    for (i = 1; i <= n; ++i) {
        if (D(i) == zero) D(i) = zero;
        if (D(i) < zero) {
            D(i) = NEG(D(i));
            if (ncvt > 0) lp_scal(ncvt, negone, &vt[IX(i, 1, ldvt)], ldvt);
        }
    }
    for (i = 1; i <= n - 1; ++i) {
        isub = 1;
        smin = D(1);
        for (j = 2; j <= n + 1 - i; ++j) {
            if (D(j) <= smin) {
                isub = j;
                smin = D(j);
            }
        }
        if (isub != n + 1 - i) {
            D(isub) = D(n + 1 - i);
            D(n + 1 - i) = smin;
            if (ncvt > 0) lp_swap(ncvt, &vt[IX(isub, 1, ldvt)], ldvt, &vt[IX(n + 1 - i, 1, ldvt)], ldvt);
            if (ncc > 0) lp_swap(ncc, &c[IX(isub, 1, ldc)], ldc, &c[IX(n + 1 - i, 1, ldc)], ldc);
        }
    }
    goto L220;
L200:
    info = 0;
    for (i = 1; i <= n - 1; ++i)
        if (E(i) != zero) ++info;
L220:
    return info;
#undef D
#undef E
#undef WK
}

/* DGELSS, M >= N, NRHS = 1, one decomposition applied at ncut values of RCOND (each as a
 * separate DGELSS call would, from the state after DBDSQR). a (m x n, lda = m) and b (length m)
 * are overwritten. s receives the singular values (descending), x (ncut x n) the solutions,
 * rank (ncut) the effective ranks. Returns 0, the DBDSQR failure count (> 0), or < 0 when a
 * DGELSS branch this port omits would be taken (-1: M < N; -2: DLASCL scaling; -3: zero matrix;
 * -4: allocation). */
static int lp_gelss(const Consts *k, int m, int n, T *a, T *b, T *s, int ncut, const T *rcond,
                    T *x, int *rank) {
    const T zero = FROMD(0.0), one = FROMD(1.0);
    const int lda = m, ldb = m;
    if (m < n) return -1;
    int mnthr = (int)((float)(m < n ? m : n) * 1.6f); /* ILAENV(6, 'DGELSS', ...) */
    T eps = k->prec;                                   /* DLAMCH('P') */
    T sfmin = k->sfmin;                                /* DLAMCH('S') */
    T smlnum = DIV(sfmin, eps);
    T bignum = DIV(one, smlnum);
    T anrm = zero, bnrm = zero;                        /* DLANGE('M', ...) */
    for (int j = 1; j <= n; ++j)
        for (int i = 1; i <= m; ++i) {
            T temp = ABS(a[IX(i, j, lda)]);
            if (anrm < temp || temp != temp) anrm = temp;
        }
    if (anrm > zero && anrm < smlnum) return -2;
    if (anrm > bignum) return -2;
    if (anrm == zero) return -3;
    for (int i = 1; i <= m; ++i) {
        T temp = ABS(b[i - 1]);
        if (bnrm < temp || temp != temp) bnrm = temp;
    }
    if ((bnrm > zero && bnrm < smlnum) || bnrm > bignum) return -2;

    T *tau = malloc(sizeof(T) * (size_t)n), *e = malloc(sizeof(T) * (size_t)n);
    T *tauq = malloc(sizeof(T) * (size_t)n), *taup = malloc(sizeof(T) * (size_t)n);
    T *work = malloc(sizeof(T) * (size_t)(4 * n + m + 1)), *bq = malloc(sizeof(T) * (size_t)ldb);
    if (!tau || !e || !tauq || !taup || !work || !bq) {
        free(tau); free(e); free(tauq); free(taup); free(work); free(bq);
        return -4;
    }
    int mm = m;
    if (m >= mnthr) {
        mm = n;
        lp_geqr2(k, m, n, a, lda, tau, work);                  /* DGEQRF, NB = 1 */
        lp_orm2r_left('T', m, 1, n, a, lda, tau, b, ldb, work); /* DORMQR -> DORM2R */
        if (n > 1)                                             /* DLASET('L', N-1, N-1, 0, 0, A(2,1)) */
            for (int j = 1; j <= n - 1; ++j)
                for (int i = j + 1; i <= n; ++i) a[IX(i, j, lda)] = zero;
    }
    lp_gebd2(k, mm, n, a, lda, s, e, tauq, taup, work);          /* DGEBRD, NB = 1 */
    lp_orm2r_left('T', mm, 1, n, a, lda, tauq, b, ldb, work);    /* DORMBR('Q','L','T') */
    lp_orgbr_p(n, a, lda, taup, work);                           /* DORGBR('P') */
    int info = lp_bdsqr(k, n, n, 1, s, e, a, lda, b, ldb, work);
    if (info == 0) {
        for (int q = 0; q < ncut; ++q) {
            for (int i = 0; i < ldb; ++i) bq[i] = b[i];
            T thr = lp_max(MUL(rcond[q], s[0]), sfmin);
            int r = 0;
            for (int i = 1; i <= n; ++i) {
                if (s[i - 1] > thr) {
                    lp_rscl(k, 1, s[i - 1], &bq[i - 1], ldb);
                    ++r;
                } else {
                    bq[i - 1] = zero;
                }
            }
            lp_gemv('T', n, n, one, a, lda, bq, 1, zero, work, 1);
            for (int i = 0; i < n; ++i) x[(size_t)q * n + i] = work[i];
            rank[q] = r;
        }
    }
    free(tau); free(e); free(tauq); free(taup); free(work); free(bq);
    return info;
}
