/* p-bit floating-point emulator: every +, -, *, /, sqrt returns the correctly rounded result in a
 * binary format with a p-bit significand and exponent range [emin, emax], with gradual underflow
 * (IEEE 754 semantics, round to nearest, ties to even).
 *
 * The carrier is binary64. Each operation computes the binary64 result s and the exact sign of
 * (true - s) with an error-free transform (TwoSum; FMA residuals for *, /, sqrt), then rounds s to
 * the target format, using that sign to settle exact midpoints. The FMA appears only inside these
 * transforms to obtain an exact residual; the model's own operations are never fused. Verified
 * operation by operation against MPFR in tests/test_pbit.py.
 *
 * Exact binary64 (p = 53, emin = -1022, emax = 1023) is the hardware itself. Any other format must
 * have emin >= -1022 + p + 1, so all of its values, subnormals included, are normal binary64
 * numbers and a binary64 subnormal is far below its smallest subnormal (it rounds to zero).
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

static int g_p = 53, g_shift = 0, g_emin = -1022, g_emax = 1023, g_native64 = 1;
static long g_bad = 0, g_notp = 0, g_over = 0, g_under = 0;
static double g_first_bad = 0.0;  /* first offending binary64 value, for diagnosis */
static int g_stage = 0, g_first_bad_stage = 0;

#define SIGNBIT 0x8000000000000000ull
#define FRACMASK 0x000fffffffffffffull

/* Round binary64 s to the target format; dir is the sign of (exact value - s). */
static inline double rnd(double s, int dir) {
    if (s == 0.0) return s;
    uint64_t u;
    memcpy(&u, &s, 8);
    int ex = (int)((u >> 52) & 0x7ff);
    if (ex == 0x7ff) { if (!g_bad++) { g_first_bad = s; g_first_bad_stage = g_stage; } return s; }
    if (g_native64) { if (ex == 0) ++g_under; return s; }
    uint64_t sign = u & SIGNBIT, mag = u ^ sign;
    if (ex == 0) { ++g_under; u = sign; memcpy(&s, &u, 8); return s; }
    int e = ex - 1023;
    int shift = g_shift + (e < g_emin ? g_emin - e : 0);
    int mdir = sign ? -dir : dir;
    if (shift == 0) {
        /* already representable */
    } else if (shift <= 52) {
        uint64_t one = 1ull << shift, half = one >> 1, low = mag & (one - 1);
        mag -= low;
        /* parity of the retained significand; at shift 52 only the implicit leading 1 remains */
        int odd = shift == 52 ? 1 : (mag & one) != 0;
        if (low > half || (low == half && (mdir > 0 || (mdir == 0 && odd)))) mag += one;
    } else if (shift == 53) {
        /* |s| in [q/2, q) with quantum q = 2^(e+1): above the midpoint rounds up; the midpoint
           itself goes by the residual, or to zero (the even neighbour) when exact. */
        mag = ((mag & FRACMASK) != 0 || mdir > 0) ? (uint64_t)(ex + 1) << 52 : 0;
    } else {
        mag = 0;
    }
    u = sign | mag;
    memcpy(&s, &u, 8);
    if (mag) {
        int e2 = (int)(mag >> 52) - 1023;
        if (e2 > g_emax) { ++g_over; s = copysign(INFINITY, s); }
        else if (e2 < g_emin) ++g_under;
    } else {
        ++g_under;
    }
    return s;
}

static inline int sgn(double e) { return (e > 0) - (e < 0); }

static inline double e_add(double a, double b) {
    double s = a + b, bb = s - a, e = (a - (s - bb)) + (b - bb);
    return rnd(s, sgn(e));
}
static inline double e_sub(double a, double b) { return e_add(a, -b); }
/* The FMA residuals below are exact whenever they cannot underflow binary64. For results under
 * 2^-900 they are formed on operands scaled by an exact power of two (2^256, or 2^512 under the
 * square root), which keeps them exact: this is the standard error-free-transform condition
 * (e.g. Boldo & Muller, Handbook of Floating-Point Arithmetic, sec. 4.4) made unconditional. */
#define TINY 0x1p-900
static inline double e_mul(double a, double b) {
    double s = a * b, r;
    if (s != 0 && fabs(s) < TINY) {
        if (fabs(a) < fabs(b)) r = fma(ldexp(a, 256), b, -ldexp(s, 256));
        else r = fma(a, ldexp(b, 256), -ldexp(s, 256));
    } else {
        r = fma(a, b, -s);
    }
    return rnd(s, sgn(r));
}
static inline double e_div(double a, double b) {
    double q = a / b, r;
    if (a != 0 && fabs(a) < TINY) r = fma(-ldexp(q, 256), b, ldexp(a, 256));
    else r = fma(-q, b, a);
    int d = sgn(r);
    return rnd(q, b < 0 ? -d : d);
}
static inline double e_sqrt(double a) {
    double s = sqrt(a), r;
    if (a != 0 && a < TINY) r = fma(-ldexp(s, 256), ldexp(s, 256), ldexp(a, 512));
    else r = fma(-s, s, a);
    return rnd(s, sgn(r));
}
static inline double e_fromd(double x) {
    if (rnd(x, 0) != x) ++g_notp;
    return x;
}
static inline double e_i2t(int n) { return rnd((double)n, 0); }
static inline int e_setfmt(int p, int emin, int emax) {
    if (p < 2 || p > 53 || emax > 1023 || emin > emax) return 1;
    g_native64 = (p == 53 && emin == -1022 && emax == 1023);
    if (!g_native64 && emin < -1022 + p + 1) return 1;
    g_p = p; g_shift = 53 - p; g_emin = emin; g_emax = emax;
    return 0;
}

#define T double
#define ADD e_add
#define SUB e_sub
#define MUL e_mul
#define DIV e_div
#define SQRT e_sqrt
#define NEG(a) (-(a))
#define ABS(a) fabs(a)
#define LDEXP(a, k) rnd(ldexp((a), (k)), 0)
#define RINT(a) nearbyint(a)
#define FROMD(x) e_fromd(x)
#define I2T(n) e_i2t(n)
#define TOD(a) (a)
#define SETFMT(p, emin, emax) e_setfmt((p), (emin), (emax))
#define STAGE(k) (g_stage = (k))
#include "pbit_algo.h"

/* ---- test and data entry points ---- */

int pbit_round(int p, int emin, int emax, size_t n, const double *x, double *out) {
    if (e_setfmt(p, emin, emax)) return 1;
    for (size_t i = 0; i < n; ++i) out[i] = rnd(x[i], 0);
    return 0;
}

/* op: 0 add, 1 sub, 2 mul, 3 div, 4 sqrt (b ignored). */
int pbit_op(int p, int emin, int emax, int op, size_t n, const double *a, const double *b,
            double *out) {
    if (e_setfmt(p, emin, emax)) return 1;
    for (size_t i = 0; i < n; ++i) {
        double x = a[i], y = b[i];
        switch (op) {
            case 0: out[i] = e_add(x, y); break;
            case 1: out[i] = e_sub(x, y); break;
            case 2: out[i] = e_mul(x, y); break;
            case 3: out[i] = e_div(x, y); break;
            case 4: out[i] = e_sqrt(x); break;
            default: return 2;
        }
    }
    return 0;
}

double pbit_first_bad(void) { return g_first_bad; }
int pbit_first_bad_stage(void) { return g_first_bad_stage; }

/* [0] non-finite results, [1] inputs that were not format values, [2] overflows, [3] results in
   the subnormal range (or flushed to zero); all reset on read. */
void pbit_counters(long *out) {
    out[0] = g_bad; out[1] = g_notp; out[2] = g_over; out[3] = g_under;
    g_bad = g_notp = g_over = g_under = 0;
}
