# Preconditioner control: fixed p32 high-contrast diffusion

All fits use only prescribed PDE forcing and boundary values. The field errors below are independent, after-fit manufactured-reference checks. Fixed network N257, lambda0.2, 561 independent coordinates; normalized pointwise PDE objective with the original boundary weighting. Every method uses the same unit-column-scaled matrix within an invocation. Dense 5222-by-561 matrices are diagnostic only; no production solver changes were made.

## Main finding

Production's native preset uses inferred parity groups, ideal columns, and requested block width1024 capped at floor(P/2)=280. There are four parity groups and none exceeds that cap. The actual coordinate-group singular values matter: maximum within-block condition is 5.14e10 and the smallest singular value is1.67e-10. Normal equations square that condition. Production's relative Gram ridge1e-10 prevents full whitening of those small directions. Guarded block QR changes the overall whitened matrix condition from1.39e5 to1.42e3.

At800 identical LSMR iterations, that improves ordinary field error from1.26e-7 to4.66e-10 and raw PDE RMS from2.39e-4 to4.70e-7. However, the advantage does not persist to the same precision level under longer iteration: Gram at4000 reaches6.27e-11 field error, while QR stops at2175 with4.65e-10. Both raw PDE errors are about4e-7. Neither reproduces the dense reference's near-floor field accuracy.

This is evidence for faster early progress from QR, **not** evidence that QR fixes the precision gap. Formally well-conditioned whitened coordinates can still be evaluated by the unstable composition A(R^-1 z): R retains a condition near5e10. That is a plausible numerical mechanism for the plateau; no direct rounding-error ablation was performed here. Repairing the underlying domain-coordinate mismatch remains the stronger next lead.

## Controls and provenance correction

P001/P002 used contiguous64 blocks, and P003 used contiguous280 blocks. They intentionally remain available as grouping controls, but their original protocol language calling contiguous grouping the production default was too broad. That is the low-level default field_parity=none; solve_route2_native sets field_parity=auto. This distinction was discovered during review and recorded in scratchpad before the native-equivalent P004/P005 controls. Do not compare the contiguous and parity groups as if only their factorization differs.

Contiguous64 blocks have within-block condition only366 and the Gram ridge is negligible; QR scarcely changes their performance. This shows why diagnosing the actual grouping is necessary.

## Results

| Run | Preconditioner | LSMR iterations | Ordinary field relative L2 | Raw ordinary PDE RMS | Algebraic objective residual norm |
|---|---|---:|---:|---:|---:|
| P001_actual_contiguous64 | diagonal | 800 | 4.37635e-06 | 0.00433892 | 6.9349e-05 |
| P001_actual_contiguous64 | gram_ridge | 800 | 1.06706e-05 | 0.0105164 | 0.00020647 |
| P001_actual_contiguous64 | qr | 800 | 1.06497e-05 | 0.0105071 | 0.000206159 |
| P002_actual_contiguous64_long | diagonal | 10000 | 1.22086e-07 | 0.000185322 | 3.45017e-06 |
| P002_actual_contiguous64_long | gram_ridge | 10000 | 3.30554e-07 | 0.000421471 | 7.15288e-06 |
| P002_actual_contiguous64_long | qr | 10000 | 3.29212e-07 | 0.000420342 | 7.13145e-06 |
| P003_actual_production280 | diagonal | 800 | 4.37635e-06 | 0.00433892 | 6.9349e-05 |
| P003_actual_production280 | gram_ridge | 800 | 4.69821e-06 | 0.00604627 | 9.84885e-05 |
| P003_actual_production280 | qr | 800 | 4.5707e-06 | 0.00574365 | 9.50979e-05 |
| P004_ideal_native_parity | diagonal | 800 | 4.35007e-06 | 0.00432917 | 6.92405e-05 |
| P004_ideal_native_parity | gram_ridge | 800 | 1.25854e-07 | 0.000239469 | 5.37545e-06 |
| P004_ideal_native_parity | qr | 800 | 4.66003e-10 | 4.70131e-07 | 1.76543e-08 |
| P005_ideal_native_parity_long | diagonal | 4000 | 3.81131e-07 | 0.000317534 | 6.89136e-06 |
| P005_ideal_native_parity_long | gram_ridge | 4000 | 6.27189e-11 | 4.4201e-07 | 6.73182e-09 |
| P005_ideal_native_parity_long | qr | 2175 | 4.64935e-10 | 4.25983e-07 | 1.73184e-08 |

## Scaling implication

A streamed row-wise QR update can retain one triangular factor per bounded column block. With P total coordinates, block cap s and row-tile size b, factor storage is O(Ps) and temporary storage O(bs+s²), with factor-construction arithmetic O(QPs) for Q residual rows. These are the same broad factor-storage and arithmetic orders as the bounded Gram scheme; no dense global Jacobian is mathematically required. But those bounds do not control condition, iteration count or how accurately a very ill-conditioned triangular inverse can be applied. For this reason production integration needs additional validation.

The actual P004/P005 triangular factors occupy631176 bytes. Their diagnostic process also holds a full matrix and temporary transformed matrix for singular-value analysis, and reaches roughly430MiB; that is not a measurement of a streamed QR implementation. LSMR operator products are dense matrix operations here, so wall times must not be compared against production streamed neural products.

Exact arrays, run protocols, source snapshots, work counts and independent audits are preserved in each linked sibling P directory. No target values or SVD solution entered the fit. Global singular values were computed only as diagnostics. No generalization or uniform-PDE certificate is claimed from sampled checks.
