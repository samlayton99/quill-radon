/* The same algorithm on a native hardware type, selected at compile time with -DPBIT_TYPE=n:
 *   1 double (binary64: p = 53, emin = -1022, emax = 1023),
 *   2 float  (binary32: p = 24, emin = -126,  emax = 127).
 * Built with -ffp-contract=off (no fused multiply-add) and without auto-vectorization. Each
 * operation is a separate inline function returning T, so every result is rounded to T.
 */
#include <math.h>
#include <stddef.h>

#if PBIT_TYPE == 1
typedef double T;
#define NATIVE_P 53
#define NATIVE_EMIN -1022
#define NATIVE_EMAX 1023
static inline T n_sqrt(T a) { return sqrt(a); }
static inline T n_rint(T a) { return nearbyint(a); }
static inline T n_ldexp(T a, int k) { return ldexp(a, k); }
static inline T n_abs(T a) { return fabs(a); }
#elif PBIT_TYPE == 2
typedef float T;
#define NATIVE_P 24
#define NATIVE_EMIN -126
#define NATIVE_EMAX 127
static inline T n_sqrt(T a) { return sqrtf(a); }
static inline T n_rint(T a) { return nearbyintf(a); }
static inline T n_ldexp(T a, int k) { return ldexpf(a, k); }
static inline T n_abs(T a) { return fabsf(a); }
#else
#error "PBIT_TYPE must be 1 (double) or 2 (float)"
#endif

static inline T n_add(T a, T b) { T r = a + b; return r; }
static inline T n_sub(T a, T b) { T r = a - b; return r; }
static inline T n_mul(T a, T b) { T r = a * b; return r; }
static inline T n_div(T a, T b) { T r = a / b; return r; }
static inline T n_neg(T a) { T r = -a; return r; }
/* Inputs arrive as doubles that are already values of T, so the conversion is exact. */
static inline T n_fromd(double x) { return (T)x; }
static inline T n_i2t(int n) { return (T)n; }
static inline int n_setfmt(int p, int emin, int emax) {
    return p != NATIVE_P || emin != NATIVE_EMIN || emax != NATIVE_EMAX;
}

#define ADD n_add
#define SUB n_sub
#define MUL n_mul
#define DIV n_div
#define SQRT n_sqrt
#define NEG n_neg
#define ABS n_abs
#define LDEXP n_ldexp
#define RINT n_rint
#define FROMD(x) n_fromd(x)
#define I2T(n) n_i2t(n)
#define TOD(a) ((double)(a))
#define SETFMT(p, emin, emax) n_setfmt((p), (emin), (emax))
#include "pbit_algo.h"

/* op: 0 add, 1 sub, 2 mul, 3 div, 4 sqrt (b ignored); inputs must be T values. For tests. */
int pbit_op(int p, int emin, int emax, int op, size_t n, const double *a, const double *b,
            double *out) {
    if (n_setfmt(p, emin, emax)) return 1;
    for (size_t i = 0; i < n; ++i) {
        T x = n_fromd(a[i]), y = n_fromd(b[i]), r;
        switch (op) {
            case 0: r = n_add(x, y); break;
            case 1: r = n_sub(x, y); break;
            case 2: r = n_mul(x, y); break;
            case 3: r = n_div(x, y); break;
            case 4: r = n_sqrt(x); break;
            default: return 2;
        }
        out[i] = (double)r;
    }
    return 0;
}
