# Full matched-parameter interpolation benchmark

Budget: **P <= 1156 scalar fitted coefficients**.
These are oracle representation errors, not PDE/ODE solve errors.

| suite | problem | BWLer rel L2 | Radon rel L2 | tensor-QI rel L2 | winner |
|---|---|---:|---:|---:|---|
| pdes | convection_c40 | 4.148e-06 | 4.154e-06 | 5.314e-13 | tensor_qi |
| pdes | convection_c80 | 1.053e+00 | 7.101e-01 | 3.500e-11 | tensor_qi |
| pdes | reaction | 3.528e-06 | 2.211e-05 | 1.306e-06 | tensor_qi |
| pdes | wave | 2.195e-09 | 1.798e-07 | 7.416e-14 | tensor_qi |
| pdes | burgers | 1.403e-01 | 1.443e-01 | 1.252e-02 | tensor_qi |
| pdes | poisson_cg | 3.460e-02 | 3.645e-02 | 1.177e-03 | tensor_qi |
| pdes | poisson_man | 1.545e-04 | 8.615e-06 | 2.119e-06 | tensor_qi |
| dysts | Lorenz | 4.538e-14 | 3.984e-14 | 4.178e-14 | radon |
| dysts | Rossler | 7.121e-09 | 1.386e-11 | 1.449e-11 | radon |
| dysts | Thomas | 1.714e-13 | 1.607e-13 | 1.722e-13 | radon |
| dysts | Halvorsen | 9.709e-14 | 9.386e-13 | 1.474e-12 | bwler |
| dysts | Lorenz96 | 4.681e-14 | 8.847e-14 | 6.697e-14 | bwler |
| dysts | InteriorSquirmer | 5.189e-05 | 4.567e-06 | 4.595e-06 | radon |
| dysts | DoublePendulum | 1.203e-15 | 1.041e-14 | 7.962e-15 | bwler |
| dysts | MacArthur | 3.195e-06 | 9.207e-07 | 2.859e-07 | tensor_qi |

BWLer's published PDE-solve errors are retained in `data.json` as a separate field; they are not directly comparable to the oracle interpolation columns above.
