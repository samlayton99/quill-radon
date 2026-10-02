# Matched box-chart preconditioner control

P006 repeats the p32 c1000 normalized ideal-column control with BoxProfileOperator, rather than AffineDiskProfileOperator. The PDE, boundary conditions, collocation locations, column normalization and native inferred parity grouping remain matched. No target data enter the solve. Dense diagnostic products do not measure production streamed runtime.

| Preconditioner | LSMR steps | Ordinary field relative L2 | Raw ordinary PDE RMS | Condition after preconditioning |
|---|---:|---:|---:|---:|
| Column scaling only | 4000, capped | 1.77e-8 | 2.33e-6 | 48197 |
| Bounded Gram, ridge1e-10 | 716 | 3.30e-14 | 4.48e-11 | 1418.689 |
| Bounded QR | 725 | 3.51e-14 | 3.88e-11 | 1418.689 |

The domain-compatible chart eliminates the Gram-versus-QR difference. Both reach the same formal condition as disk-chart QR, but without its roughly4.65e-10 ordinary-field plateau. This strengthens the coordinate-conditioning diagnosis. It does not prove a general bound or certify sampled residuals as global solution error. Further production integration should favor the existing box chart and appropriate Krylov accuracy rather than introducing QR solely on this evidence.

The original real-valued single-hidden-layer tanh network is exported and audited after fitting. Reference field values are used only for those audits. All protocols, counts, singular-value diagnostics and coefficients remain in this directory.
