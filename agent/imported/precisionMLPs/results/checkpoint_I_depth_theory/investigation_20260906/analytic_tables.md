# Analytic confirmation: final errors

Three new seeds (3, 4, 5); medians with seed minima and maxima. All errors are relative L2. Inner test samples use 90% of the training-ball radius; full-ball samples include its outer shell. No held-out errors select a checkpoint. See the main report for protocol and compute limits.

| Target | Arm | Parameters | Inner test median [min, max] | Full ball median [min, max] |
|---|---|---:|---:|---:|
| fast_waves | raw | 539 | 0.0749223 [0.0664787, 0.207992] | 0.0975219 [0.0880729, 0.199197] |
| fast_waves | poly3 | 191 | 0.256934 [0.000393859, 0.320986] | 0.254475 [0.00103938, 0.344685] |
| fast_waves | quadratic | 179 | 0.000399827 [0.000313883, 0.000516633] | 0.00126421 [0.000366267, 0.00160954] |
| fast_waves | quadratic_free | 539 | 0.00097712 [0.000726696, 0.00108341] | 0.00153675 [0.00114336, 0.00200308] |
| fast_waves | mlp | 568 | 0.0174103 [0.0170596, 0.0203765] | 0.0220292 [0.0214355, 0.0250779] |
| composition | raw | 539 | 0.00301969 [0.00288001, 0.00314588] | 0.00401679 [0.00374156, 0.00438572] |
| composition | poly3 | 191 | 0.00364996 [0.00256695, 0.00723253] | 0.0047531 [0.00342815, 0.00862056] |
| composition | quadratic | 179 | 0.00604764 [0.00524283, 0.00746913] | 0.00810362 [0.00681708, 0.00901283] |
| composition | quadratic_free | 539 | 0.00186209 [0.0012841, 0.00197839] | 0.00256228 [0.00170434, 0.00271243] |
| composition | mlp | 568 | 0.00113579 [0.000826686, 0.00128814] | 0.0015139 [0.00109256, 0.00158618] |
| product_peak | raw | 539 | 0.0225472 [0.0154798, 0.024055] | 0.0294229 [0.0194583, 0.0298619] |
| product_peak | poly3 | 191 | 0.0253841 [0.0167857, 0.02614] | 0.0304269 [0.0222524, 0.0309215] |
| product_peak | quadratic | 179 | 0.0149247 [0.0143007, 0.0245555] | 0.0192313 [0.0184914, 0.0313082] |
| product_peak | mlp | 568 | 0.00925129 [0.00808818, 0.0132537] | 0.0108708 [0.0107369, 0.0163802] |
| random_ridges | raw | 539 | 0.254908 [0.219568, 0.268868] | 0.274231 [0.239458, 0.277329] |
| random_ridges | poly3 | 191 | 0.275384 [0.240183, 0.293177] | 0.294847 [0.265773, 0.320503] |
| random_ridges | quadratic | 179 | 0.371988 [0.349406, 0.383176] | 0.410943 [0.373679, 0.426861] |
| random_ridges | quadratic_free | 539 | 0.0759938 [0.0503715, 0.104224] | 0.0963267 [0.0653149, 0.122879] |
| random_ridges | mlp | 568 | 0.0243183 [0.0240959, 0.0301519] | 0.0304206 [0.0285508, 0.0363637] |
