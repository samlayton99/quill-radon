# expF17 Plot 2 -- best configuration per method

geomean rel L2 over seeds at each method's best (W, config)


## convection_c40

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 2025 | 3.97e-07 | 1.30 |
| ELM | oracle | 2025 | 8.04e-07 | 1.68 |
| QI-Radon | dynamic | 2025 | 5.05e-10 | 1.32 |
| QI-Radon | oracle | 2025 | 6.31e-10 | 1.42 |
| QI-tensor | dynamic | 2025 | 9.59e-10 | 1.25 |
| QI-tensor | oracle | 2025 | 1.04e-09 | 1.93 |

## convection_c80

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 2025 | 8.24e-02 | 3.06 |
| ELM | oracle | 2025 | 4.23e-02 | 1.66 |
| QI-Radon | dynamic | 2025 | 7.37e-01 | 1.07 |
| QI-Radon | oracle | 2025 | 6.54e-02 | 1.81 |
| QI-tensor | dynamic | 2025 | 6.89e-05 | 1.29 |
| QI-tensor | oracle | 2025 | 2.16e-04 | 2.19 |

## reaction

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 2025 | 4.72e-05 | 2.02 |
| ELM | oracle | 2025 | 2.28e-06 | 1.88 |
| QI-Radon | dynamic | 2025 | 1.82e-04 | 1.98 |
| QI-Radon | oracle | 2025 | 2.40e-06 | 1.56 |
| QI-tensor | dynamic | 2025 | 6.66e-06 | 8.17 |
| QI-tensor | oracle | 2025 | 4.62e-07 | 1.42 |

## wave

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 2025 | 2.77e-08 | 1.73 |
| ELM | oracle | 2025 | 2.81e-08 | 2.41 |
| QI-Radon | dynamic | 2025 | 2.18e-10 | 1.61 |
| QI-Radon | oracle | 2025 | 4.66e-10 | 2.11 |
| QI-tensor | dynamic | 2025 | 4.93e-11 | 1.34 |
| QI-tensor | oracle | 2025 | 2.89e-10 | 2.14 |

## burgers

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 2025 | 3.15e-01 | 1.02 |
| ELM | oracle | 2025 | 1.39e-01 | 1.06 |
| QI-Radon | dynamic | 2025 | 3.13e-01 | 1.01 |
| QI-Radon | oracle | 2025 | 1.56e-01 | 1.17 |
| QI-tensor | dynamic | 1024 | 2.92e-01 | 1.01 |
| QI-tensor | oracle | 1024 | 1.36e-01 | 1.07 |

## poisson_man

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 2025 | 3.71e-06 | 1.48 |
| ELM | oracle | 2025 | 3.54e-06 | 3.02 |
| QI-Radon | dynamic | 2025 | 1.45e-05 | 1.27 |
| QI-Radon | oracle | 2025 | 6.85e-06 | 1.87 |
| QI-tensor | dynamic | 2025 | 1.26e-06 | 1.12 |
| QI-tensor | oracle | 2025 | 4.52e-07 | 1.48 |

## darcy_man

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 2025 | 8.15e-13 | 1.49 |
| ELM | oracle | 2025 | 1.92e-11 | 2.29 |
| QI-Radon | dynamic | 2025 | 3.33e-14 | 1.12 |
| QI-Radon | oracle | 2025 | 4.42e-13 | 3.68 |
| QI-tensor | dynamic | 2025 | 6.68e-14 | 1.44 |
| QI-tensor | oracle | 2025 | 9.27e-13 | 1.65 |

## darcy_orig

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|

## dysts_Lorenz

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 512 | 7.47e-10 | 3.24 |
| ELM | oracle | 512 | 5.26e-11 | 4.35 |
| QI | dynamic | 512 | 7.38e-12 | 2.05 |
| QI | oracle | 256 | 1.21e-13 | 3.46 |

## dysts_Rossler

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 512 | 2.64e-06 | 33.66 |
| ELM | oracle | 512 | 1.09e-07 | 7.82 |
| QI | dynamic | 384 | 1.07e-12 | 1.13 |
| QI | oracle | 512 | 1.99e-13 | 5.20 |

## dysts_Thomas

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 512 | 6.96e-09 | 5.64 |
| ELM | oracle | 512 | 4.18e-12 | 2.33 |
| QI | dynamic | 384 | 4.38e-12 | 1.23 |
| QI | oracle | 384 | 1.80e-13 | 5.54 |

## dysts_Halvorsen

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 512 | 1.23e-09 | 12.90 |
| ELM | oracle | 512 | 4.34e-11 | 3.55 |
| QI | dynamic | 384 | 1.65e-12 | 1.42 |
| QI | oracle | 512 | 2.14e-13 | 3.55 |

## dysts_Lorenz96

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 384 | 3.82e-10 | 6.94 |
| ELM | oracle | 512 | 3.55e-12 | 11.68 |
| QI | dynamic | 384 | 3.91e-13 | 1.74 |
| QI | oracle | 192 | 9.74e-14 | 1.84 |

## dysts_InteriorSquirmer

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 512 | 8.82e-03 | 1.17 |
| ELM | oracle | 512 | 5.48e-05 | 1.41 |
| QI | dynamic | 512 | 5.70e-04 | 1.06 |
| QI | oracle | 512 | 1.29e-06 | 1.49 |

## dysts_DoublePendulum

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | dynamic | 384 | 2.01e-10 | 2.55 |
| ELM | oracle | 384 | 2.96e-15 | 1.23 |
| QI | dynamic | 192 | 2.35e-14 | 1.16 |
| QI | oracle | 128 | 6.08e-15 | 1.29 |

## dysts_MacArthur

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| ELM | oracle | 512 | 1.50e-07 | 1.32 |
| QI | dynamic | 96 | 2.52e-04 | 1.00 |
| QI | oracle | 512 | 3.96e-08 | 1.91 |
