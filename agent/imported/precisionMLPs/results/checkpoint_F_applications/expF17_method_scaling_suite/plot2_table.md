# expF17 Plot 2 -- best configuration per method

geomean rel L2 over seeds at each method's best (W, config)


## convection_c40

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 2024 | 5.22e-13 | 1.15 |
| BWLer (explicit) | oracle | 2024 | 6.87e-15 | 1.03 |
| ELM | dynamic | 4096 | 2.63e-09 | 1.21 |
| ELM | oracle | 4096 | 1.89e-09 | 1.32 |
| PINN (trained) | dynamic | 529 | 5.02e-01 | 1.42 |
| PINN (trained) | oracle | 529 | 1.01e-02 | 1.30 |
| QI-Radon | dynamic | 4096 | 2.23e-12 | 1.25 |
| QI-Radon | oracle | 4096 | 3.93e-12 | 1.32 |
| QI-tensor | dynamic | 4096 | 3.21e-12 | 1.11 |
| QI-tensor | oracle | 4096 | 2.69e-12 | 1.28 |
| RBF-IMQ | dynamic | 2025 | 6.76e-04 | 1.08 |
| RBF-IMQ | oracle | 2025 | 7.65e-05 | 1.21 |
| RBF-PHS+p1 | dynamic | 2028 | 8.93e-01 | 1.01 |
| RBF-PHS+p1 | oracle | 2028 | 6.89e-03 | 1.08 |
| spectral (F/C) | dynamic | 256 | 2.73e-14 | 1.45 |
| spectral (F/C) | oracle | 256 | 3.87e-15 | 1.06 |

## convection_c80

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 2024 | 3.74e-13 | 1.19 |
| BWLer (explicit) | oracle | 2024 | 7.71e-15 | 1.01 |
| ELM | dynamic | 4096 | 4.62e-05 | 1.08 |
| ELM | oracle | 4096 | 2.88e-05 | 1.29 |
| PINN (trained) | dynamic | 529 | 9.34e-01 | 1.03 |
| PINN (trained) | oracle | 529 | 6.70e-01 | 1.36 |
| QI-Radon | dynamic | 4096 | 8.67e-08 | 1.26 |
| QI-Radon | oracle | 4096 | 4.25e-09 | 1.27 |
| QI-tensor | dynamic | 4096 | 8.51e-09 | 1.11 |
| QI-tensor | oracle | 4096 | 1.95e-09 | 1.01 |
| RBF-IMQ | dynamic | 2024 | 9.62e-01 | 1.01 |
| RBF-IMQ | oracle | 2024 | 9.13e-03 | 1.19 |
| RBF-PHS+p1 | dynamic | 2028 | 9.87e-01 | 1.00 |
| RBF-PHS+p1 | oracle | 2028 | 4.33e-02 | 1.04 |
| spectral (F/C) | dynamic | 528 | 9.81e-14 | 1.93 |
| spectral (F/C) | oracle | 528 | 6.38e-15 | 1.22 |

## reaction

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 1980 | 4.07e-10 | 1.21 |
| BWLer (explicit) | oracle | 1980 | 3.80e-11 | 1.02 |
| ELM | dynamic | 4096 | 8.28e-07 | 1.35 |
| ELM | oracle | 4096 | 1.53e-09 | 1.42 |
| PINN (trained) | dynamic | 529 | 6.56e-02 | 1.12 |
| PINN (trained) | oracle | 121 | 3.16e-03 | 1.07 |
| QI-Radon | dynamic | 4096 | 2.70e-06 | 3.13 |
| QI-Radon | oracle | 4096 | 1.61e-09 | 1.04 |
| QI-tensor | dynamic | 4096 | 6.77e-07 | 1.54 |
| QI-tensor | oracle | 4096 | 6.22e-11 | 1.16 |
| RBF-IMQ | dynamic | 2025 | 1.71e-04 | 1.48 |
| RBF-IMQ | oracle | 2025 | 1.07e-06 | 1.21 |
| RBF-PHS+p1 | dynamic | 2028 | 1.55e-02 | 2.00 |
| RBF-PHS+p1 | oracle | 2028 | 1.02e-04 | 1.09 |
| spectral (F/C) | dynamic | 1980 | 4.07e-10 | 1.21 |
| spectral (F/C) | oracle | 1980 | 3.80e-11 | 1.02 |

## wave

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 2025 | 3.74e-10 | 1.25 |
| BWLer (explicit) | oracle | 2025 | 6.37e-15 | 1.01 |
| ELM | dynamic | 4096 | 1.10e-10 | 1.17 |
| ELM | oracle | 8281 | 9.40e-12 | 1.13 |
| PINN (trained) | dynamic | 256 | 3.31e-01 | 1.07 |
| PINN (trained) | oracle | 529 | 4.67e-02 | 1.13 |
| QI-Radon | dynamic | 4096 | 1.06e-12 | 1.17 |
| QI-Radon | oracle | 8281 | 2.38e-13 | 1.13 |
| QI-tensor | dynamic | 4096 | 3.40e-12 | 1.81 |
| QI-tensor | oracle | 8281 | 5.52e-13 | 1.30 |
| RBF-IMQ | dynamic | 2025 | 1.56e-03 | 1.18 |
| RBF-IMQ | oracle | 2025 | 1.62e-05 | 1.12 |
| RBF-PHS+p1 | dynamic | 2028 | 3.93e-01 | 1.06 |
| RBF-PHS+p1 | oracle | 2028 | 2.30e-03 | 1.09 |
| spectral (F/C) | dynamic | 2025 | 3.03e-10 | 1.15 |
| spectral (F/C) | oracle | 2025 | 5.99e-15 | 1.09 |

## burgers

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 2160 | 1.14e-01 | 1.06 |
| BWLer (explicit) | oracle | 2160 | 2.56e-02 | 1.03 |
| ELM | dynamic | 4096 | 2.86e-01 | 1.09 |
| ELM | oracle | 4096 | 6.92e-02 | 1.09 |
| PINN (trained) | dynamic | 256 | 2.73e-01 | 1.09 |
| PINN (trained) | oracle | 529 | 3.99e-02 | 1.11 |
| QI-Radon | dynamic | 4096 | 2.78e-01 | 1.02 |
| QI-Radon | oracle | 4096 | 5.96e-02 | 1.01 |
| QI-tensor | dynamic | 4096 | 2.63e-01 | 1.02 |
| QI-tensor | oracle | 4096 | 5.99e-02 | 1.05 |
| RBF-IMQ | dynamic | 1980 | 2.32e-01 | 1.03 |
| RBF-IMQ | oracle | 1980 | 3.59e-02 | 1.01 |
| RBF-PHS+p1 | dynamic | 1983 | 2.86e-01 | 1.04 |
| RBF-PHS+p1 | oracle | 1983 | 5.56e-02 | 1.04 |
| spectral (F/C) | dynamic | 2160 | 1.14e-01 | 1.06 |
| spectral (F/C) | oracle | 2160 | 2.56e-02 | 1.03 |

## poisson_man

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 2025 | 6.39e-06 | 1.79 |
| BWLer (explicit) | oracle | 2025 | 2.38e-07 | 1.01 |
| ELM | dynamic | 4096 | 1.86e-07 | 1.07 |
| ELM | oracle | 4096 | 1.90e-08 | 1.29 |
| PINN (trained) | dynamic | 121 | 3.29e-02 | 1.26 |
| PINN (trained) | oracle | 1024 | 8.06e-03 | 1.00 |
| QI-Radon | dynamic | 4096 | 1.43e-07 | 1.16 |
| QI-Radon | oracle | 4096 | 1.21e-08 | 1.01 |
| QI-tensor | dynamic | 4096 | 1.68e-09 | 1.33 |
| QI-tensor | oracle | 4096 | 2.29e-10 | 1.04 |
| RBF-IMQ | dynamic | 2025 | 6.65e-06 | 1.14 |
| RBF-IMQ | oracle | 2025 | 7.28e-07 | 1.10 |
| RBF-PHS+p1 | dynamic | 2028 | 2.93e-05 | 1.08 |
| RBF-PHS+p1 | oracle | 2028 | 3.71e-06 | 1.04 |
| spectral (F/C) | dynamic | 2025 | 6.38e-06 | 1.77 |
| spectral (F/C) | oracle | 2025 | 2.38e-07 | 1.01 |

## darcy_man

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 1024 | 5.88e-14 | 1.16 |
| BWLer (explicit) | oracle | 1024 | 3.83e-15 | 1.02 |
| ELM | dynamic | 4096 | 3.36e-14 | 1.14 |
| ELM | oracle | 4096 | 6.48e-13 | 1.47 |
| PINN (trained) | dynamic | 1024 | 4.05e-04 | 1.00 |
| PINN (trained) | oracle | 529 | 5.08e-03 | 1.09 |
| QI-Radon | dynamic | 4096 | 7.20e-15 | 1.31 |
| QI-Radon | oracle | 4096 | 4.98e-14 | 1.48 |
| QI-tensor | dynamic | 4096 | 5.89e-15 | 1.31 |
| QI-tensor | oracle | 4096 | 2.33e-14 | 1.22 |
| RBF-IMQ | dynamic | 256 | 4.82e-04 | 1.49 |
| RBF-IMQ | oracle | 2024 | 5.89e-05 | 1.24 |
| RBF-PHS+p1 | dynamic | 2027 | 5.04e-03 | 1.30 |
| RBF-PHS+p1 | oracle | 2027 | 2.18e-04 | 1.18 |
| spectral (F/C) | dynamic | 1024 | 1.35e-12 | 1.20 |
| spectral (F/C) | oracle | 1024 | 4.01e-15 | 1.06 |

## darcy_orig

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|

## dysts_Lorenz

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 384 | 8.56e-14 | 1.00 |
| BWLer (explicit) | oracle | 384 | 9.80e-10 | 29.45 |
| ELM | dynamic | 1024 | 6.53e-10 | 10.65 |
| ELM | oracle | 1024 | 1.44e-13 | 2.94 |
| QI | dynamic | 1024 | 3.84e-14 | 1.15 |
| QI | oracle | 1024 | 5.31e-14 | 2.62 |
| RBF-IMQ | dynamic | 384 | 4.85e-07 | 3.23 |
| RBF-IMQ | oracle | 384 | 4.86e-08 | 1.09 |
| RBF-PHS+p1 | dynamic | 512 | 1.63e-05 | 1.79 |
| RBF-PHS+p1 | oracle | 512 | 6.41e-08 | 1.08 |
| spectral (F/C) | dynamic | 384 | 5.75e-15 | 1.00 |
| spectral (F/C) | oracle | 384 | 9.65e-10 | 27.35 |

## dysts_Rossler

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 512 | 6.42e-12 | 1.00 |
| BWLer (explicit) | oracle | 192 | 1.51e-04 | 1.15 |
| ELM | dynamic | 1024 | 3.33e-10 | 1.78 |
| ELM | oracle | 1024 | 3.56e-11 | 3.62 |
| QI | dynamic | 1024 | 8.03e-15 | 1.63 |
| QI | oracle | 1024 | 3.85e-14 | 1.46 |
| RBF-IMQ | dynamic | 384 | 1.58e-05 | 1.15 |
| RBF-IMQ | oracle | 384 | 4.11e-07 | 1.15 |
| RBF-PHS+p1 | dynamic | 512 | 2.24e-04 | 1.08 |
| RBF-PHS+p1 | oracle | 512 | 8.39e-07 | 1.02 |
| spectral (F/C) | dynamic | 512 | 6.42e-12 | 1.00 |
| spectral (F/C) | oracle | 192 | 1.51e-04 | 1.15 |

## dysts_Thomas

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 512 | 6.56e-13 | 1.00 |
| BWLer (explicit) | oracle | 384 | 1.77e-10 | 31.53 |
| ELM | dynamic | 1024 | 6.84e-09 | 10.10 |
| ELM | oracle | 1024 | 2.98e-14 | 1.36 |
| QI | dynamic | 1024 | 3.03e-13 | 1.29 |
| QI | oracle | 1024 | 4.74e-14 | 2.31 |
| RBF-IMQ | dynamic | 384 | 1.28e-03 | 1.66 |
| RBF-IMQ | oracle | 384 | 2.97e-07 | 1.20 |
| RBF-PHS+p1 | dynamic | 512 | 4.15e-05 | 2.39 |
| RBF-PHS+p1 | oracle | 512 | 1.63e-08 | 1.11 |
| spectral (F/C) | dynamic | 512 | 4.30e-14 | 1.00 |
| spectral (F/C) | oracle | 384 | 8.59e-11 | 42.41 |

## dysts_Halvorsen

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 512 | 4.73e-14 | 1.00 |
| BWLer (explicit) | oracle | 384 | 1.13e-08 | 30.56 |
| ELM | dynamic | 1024 | 2.76e-10 | 5.77 |
| ELM | oracle | 1024 | 2.42e-13 | 1.49 |
| QI | dynamic | 1024 | 1.57e-14 | 2.58 |
| QI | oracle | 1024 | 6.82e-14 | 3.07 |
| RBF-IMQ | dynamic | 384 | 1.11e-04 | 3.01 |
| RBF-IMQ | oracle | 512 | 5.14e-07 | 1.10 |
| RBF-PHS+p1 | dynamic | 512 | 2.84e-05 | 2.32 |
| RBF-PHS+p1 | oracle | 512 | 7.72e-08 | 1.09 |
| spectral (F/C) | dynamic | 512 | 5.71e-15 | 1.00 |
| spectral (F/C) | oracle | 384 | 1.11e-08 | 30.57 |

## dysts_Lorenz96

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 384 | 1.18e-14 | 1.00 |
| BWLer (explicit) | oracle | 384 | 2.15e-10 | 18.37 |
| ELM | dynamic | 1024 | 3.19e-11 | 9.34 |
| ELM | oracle | 1024 | 5.87e-14 | 1.23 |
| QI | dynamic | 1024 | 3.04e-14 | 1.29 |
| QI | oracle | 1024 | 4.96e-14 | 3.10 |
| RBF-IMQ | dynamic | 384 | 1.36e-04 | 1.04 |
| RBF-IMQ | oracle | 384 | 3.46e-07 | 1.01 |
| RBF-PHS+p1 | dynamic | 512 | 6.32e-07 | 4.28 |
| RBF-PHS+p1 | oracle | 512 | 2.07e-08 | 1.04 |
| spectral (F/C) | dynamic | 384 | 2.43e-15 | 1.00 |
| spectral (F/C) | oracle | 384 | 2.08e-10 | 21.80 |

## dysts_InteriorSquirmer

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 512 | 3.08e-03 | 1.00 |
| BWLer (explicit) | oracle | 192 | 1.05e-03 | 1.08 |
| ELM | dynamic | 1024 | 3.27e-03 | 1.22 |
| ELM | oracle | 1024 | 1.30e-05 | 1.16 |
| QI | dynamic | 1024 | 1.80e-07 | 1.43 |
| QI | oracle | 1024 | 1.37e-09 | 1.33 |
| RBF-IMQ | dynamic | 512 | 6.36e-04 | 1.08 |
| RBF-IMQ | oracle | 512 | 9.80e-07 | 1.24 |
| RBF-PHS+p1 | dynamic | 512 | 1.58e-03 | 1.02 |
| RBF-PHS+p1 | oracle | 512 | 3.20e-06 | 1.14 |
| spectral (F/C) | dynamic | 512 | 3.08e-03 | 1.00 |
| spectral (F/C) | oracle | 192 | 1.05e-03 | 1.08 |

## dysts_DoublePendulum

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 256 | 3.20e-15 | 1.00 |
| BWLer (explicit) | oracle | 192 | 4.20e-15 | 1.29 |
| ELM | dynamic | 384 | 2.01e-10 | 2.55 |
| ELM | oracle | 1024 | 2.09e-15 | 1.42 |
| QI | dynamic | 192 | 2.35e-14 | 1.16 |
| QI | oracle | 128 | 6.08e-15 | 1.29 |
| RBF-IMQ | dynamic | 192 | 1.82e-05 | 1.18 |
| RBF-IMQ | oracle | 384 | 2.22e-07 | 1.16 |
| RBF-PHS+p1 | dynamic | 512 | 2.20e-10 | 1.64 |
| RBF-PHS+p1 | oracle | 512 | 3.70e-12 | 1.06 |
| spectral (F/C) | dynamic | 384 | 7.44e-16 | 1.00 |
| spectral (F/C) | oracle | 192 | 1.67e-15 | 1.12 |

## dysts_MacArthur

| method | regime | best W | geomean rel L2 | GSD |
|---|---|---|---|---|
| BWLer (explicit) | dynamic | 256 | 4.30e-05 | 1.00 |
| BWLer (explicit) | oracle | 192 | 9.23e-07 | 1.04 |
| ELM | dynamic | 256 | 5.59e-05 | 1.11 |
| ELM | oracle | 1024 | 7.40e-08 | 1.07 |
| QI | dynamic | 384 | 5.17e-05 | 1.03 |
| QI | oracle | 1024 | 3.97e-09 | 1.02 |
| RBF-IMQ | dynamic | 512 | 4.35e-05 | 1.01 |
| RBF-IMQ | oracle | 384 | 2.29e-07 | 1.15 |
| RBF-PHS+p1 | dynamic | 384 | 4.21e-05 | 1.92 |
| RBF-PHS+p1 | oracle | 512 | 2.78e-08 | 1.11 |
| spectral (F/C) | dynamic | 256 | 4.30e-05 | 1.00 |
| spectral (F/C) | oracle | 192 | 9.23e-07 | 1.04 |
