# Complete method catalogue by present use

All **60 historical method IDs** are retained, with their original names, cost formulas, statuses, rates, source pointers and historical ratings in [methods.csv](methods.csv). Current utility and architecture qualifications are additional columns, not retroactive changes to results. Current top-three choices live in [rankings by problem](../rankings.md).

The 12 newer components/extensions are registered separately in [components](../components.md); they are not counted as 12 independent inventions. The original catalogue contains detailed descriptions for each linked ID. Its rating scale allowed some architectures that are now excluded.

This organization covers the Radon/QUILL construction and PDE program plus relevant precursors, not unrelated optimizer or interactive-app projects. Every expF19 result file is indexed in [result_inventory.csv](result_inventory.csv); older precursor evidence remains linked through each historical method.

## Analytical construction

| Method and full description | Current utility | Architecture/scope | Why keep it |
|---|---|---|---|
| [A09 — Nearest-direction Fourier snapping](../../radon_method_catalogue.md#a09-nearest-direction-fourier-snapping) | Failed shortcut | Flat | Nearest-direction snapping missed required angular precision. |
| [A10 — Prescribed Radon profiles + shared scalar LS](../../radon_method_catalogue.md#a10-prescribed-radon-profiles--shared-scalar-ls) | Useful hybrid | Flat | Known profiles reduce discovery to shared scalar fitting. |
| [A11 — Angular interpolation + scalar profile LS](../../radon_method_catalogue.md#a11-angular-interpolation--scalar-profile-ls) | Useful hybrid | Flat | Angular interpolation repairs snapping; scalar profile solve retained. |
| [A12 — Leading derivative-sample readouts](../../radon_method_catalogue.md#a12-leading-derivative-sample-readouts) | Failed floor control | Flat | Leading derivative samples retain smoothing bias. |
| [A13 — Complex-shift readouts with short halo](../../radon_method_catalogue.md#a13-complex-shift-readouts-with-short-halo) | Failed boundary shortcut | Flat | Short finite band lacks adequate endpoint treatment. |
| [A14 — Complex-shift readouts with long halo](../../radon_method_catalogue.md#a14-complex-shift-readouts-with-long-halo) | Successful predecessor | Flat | Direct complex-shift construction with expensive long halos. |
| [A15 — General-d tensor sphere construction](../../radon_method_catalogue.md#a15-general-d-tensor-sphere-construction) | General dimension control | Flat | Dense tensor sphere construction works at modest d; angular curse remains. |
| [A16 — Sobol sphere construction](../../radon_method_catalogue.md#a16-sobol-sphere-construction) | Approximate alternative | Flat | Sobol directions scale in count but tested precision did not reach floor. |
| [A17 — Known-metric adapted sphere construction](../../radon_method_catalogue.md#a17-known-metric-adapted-sphere-construction) | Known-prior specialization | Flat | Adapted metric is supplied; not automatic general discovery. |
| [A18 — Sparse known Fourier-ridge construction](../../radon_method_catalogue.md#a18-sparse-known-fourier-ridge-construction) | Best for supplied sparse ridges | Flat | Excellent high ambient d when exact sparse support is known. |
| [A19 — Naive square-root halo truncation](../../radon_method_catalogue.md#a19-naive-square-root-halo-truncation) | Failed boundary shortcut | Flat | sqrt(N) truncation alone is not corrected QUILL. |
| [A20 — Corrected finite-contour QUILL Radon](../../radon_method_catalogue.md#a20-corrected-finite-contour-quill-radon) | Preferred analytical constructor | Flat | Explicit corrected finite halo; known profiles required. |
| [A21 — Analytic Gegenbauer ball-frame reconstruction](../../radon_method_catalogue.md#a21-analytic-gegenbauer-ball-frame-reconstruction) | Capacity/coordinate tool | Analytical frame | Useful exact structure audit; not by itself an unknown-PDE solve. |

## Clean samples

| Method and full description | Current utility | Architecture/scope | Why keep it |
|---|---|---|---|
| [A01 — Uniform Radon grid + global least squares](../../radon_method_catalogue.md#a01-uniform-radon-grid--global-least-squares) | Baseline/default | Flat | Broad reference; dense global fit remains memory-limited. |
| [A02 — Smooth monitor-adapted ridge centers](../../radon_method_catalogue.md#a02-smooth-monitor-adapted-ridge-centers) | Specialized geometry | Flat | Smooth monitor helps localized targets; rough variant failed. |
| [A03 — Gradient-energy angular-density placement](../../radon_method_catalogue.md#a03-gradient-energy-angular-density-placement) | Failed heuristic | Flat | Gradient-energy directions did not provide useful gain. |
| [A04 — Active-subspace ridge placement](../../radon_method_catalogue.md#a04-active-subspace-ridge-placement) | Specialized geometry | Flat | Requires exploitable low-dimensional structure; estimated subspace results mixed. |
| [A05 — Stagewise projection-pursuit ridge atoms](../../radon_method_catalogue.md#a05-stagewise-projection-pursuit-ridge-atoms) | Failed standalone | Flat | Stagewise atom addition needed joint polishing. |
| [A06 — Joint direction VarPro / Gauss–Newton](../../radon_method_catalogue.md#a06-joint-direction-varpro--gaussnewton) | Specialized geometry | Flat | Strong sparse-direction refinement; repeated global readout fits. |
| [A07 — Greedy hierarchical ridge mesh](../../radon_method_catalogue.md#a07-greedy-hierarchical-ridge-mesh) | Specialized geometry | Flat | Useful sparse-plus-background hierarchy; dense fitting remains. |

## Native PDE solving

| Method and full description | Current utility | Architecture/scope | Why keep it |
|---|---|---|---|
| [B01 — Dense spacetime ridge Gauss–Newton](../../radon_method_catalogue.md#b01-dense-spacetime-ridge-gaussnewton) | Dense baseline | Flat spacetime | Native discovery, but global correction solve hits memory wall. |
| [B12 — Flat-ridge shifted elliptic residual iteration](../../radon_method_catalogue.md#b12-flat-ridge-shifted-elliptic-residual-iteration) | Restricted nonfloor precursor | Flat | Shifted elliptic residual iteration depends on operator. |
| [E02 — Cached analytical disk ridge coordinates](../../radon_method_catalogue.md#e02-cached-analytical-disk-ridge-coordinates) | Useful small cached baseline | Flat | Strong smooth disk precision; dense retained point-coordinate cache. |
| [E03 — Cached enclosing-ball / box ridge coordinates](../../radon_method_catalogue.md#e03-cached-enclosing-ball--box-ridge-coordinates) | Memory-limited predecessor | Flat | Box/boundary extension; dense map/cache replaced in current platform. |
| [E04 — Low-degree sparse angular cubature](../../radon_method_catalogue.md#e04-low-degree-sparse-angular-cubature) | Low-degree specialization | Flat | 20D quadratic/10D cubic do not establish generic high-dimensional precision. |
| [E05 — Declared-active-axis ridge construction](../../radon_method_catalogue.md#e05-declared-active-axis-ridge-construction) | Known-prior specialization | Flat original input | Supplied active axes; not learned manifold compression. |
| [E07 — Direct directional-profile operator](../../radon_method_catalogue.md#e07-direct-directional-profile-operator) | Reusable operator component | Flat | Forward/adjoint primitive, not a complete solve. |
| [E08 — Matched ball-frame residual iteration](../../radon_method_catalogue.md#e08-matched-ball-frame-residual-iteration) | Useful matched-operator solver | Flat | Analytical frame iteration; restricted operator match. |
| [E09 — Streamed general solve with redundant profiles](../../radon_method_catalogue.md#e09-streamed-general-solve-with-redundant-profiles) | Superseded streamed baseline | Flat | Memory improvement with redundant-profile conditioning. |
| [E10 — Streamed analytical disk modes with block preconditioning](../../radon_method_catalogue.md#e10-streamed-analytical-disk-modes-with-block-preconditioning) | Main current ancestor | Flat | Independent modes and block preconditioning; see later U01 extensions. |
| [E11 — Streamed analytical disk modes with diagonal scaling](../../radon_method_catalogue.md#e11-streamed-analytical-disk-modes-with-diagonal-scaling) | Scaling variant/control | Flat | Diagonal alternative to block preconditioning; different speed/memory tradeoff. |

## Inverse methods

| Method and full description | Current utility | Architecture/scope | Why keep it |
|---|---|---|---|
| [D01 — Analytical heat-tensor inverse wrapper](../../radon_method_catalogue.md#d01-analytical-heat-tensor-inverse-wrapper) | Useful analytical-family inverse | Flat analytical forward | Heat-tensor inference; not general nonlinear inverse. |
| [D02 — Native Burgers parameter inverse with tangents](../../radon_method_catalogue.md#d02-native-burgers-parameter-inverse-with-tangents) | Strong restricted native inverse | Flat at fixed time | Burgers parameters/tangents; no single global spacetime export. |
| [D03 — Unknown initial-profile inverse with regularization](../../radon_method_catalogue.md#d03-unknown-initial-profile-inverse-with-regularization) | Restricted functional inverse | Flat at fixed time | Regularized 17-coordinate initial profile; noise/information limited. |
| [D04 — Regularized backward-heat compilation](../../radon_method_catalogue.md#d04-regularized-backward-heat-compilation) | Classical regularized fallback | Flat compiler output | Backward heat solved spectrally before encoding; dependency explicit. |

## Specialized dynamics

| Method and full description | Current utility | Architecture/scope | Why keep it |
|---|---|---|---|
| [B02 — Analytic linear-IVP Radon propagation](../../radon_method_catalogue.md#b02-analytic-linear-ivp-radon-propagation) | Analytical linear IVP | Flat when exported | Supported linear propagation only. |
| [B04 — Fixed spacetime wave construction](../../radon_method_catalogue.md#b04-fixed-spacetime-wave-construction) | Analytical wave IVP | Flat spacetime | Supported exact propagation family. |
| [B05 — Cole–Hopf Burgers construction](../../radon_method_catalogue.md#b05-colehopf-burgers-construction) | Analytical nonlinear shortcut | Flat when exported | Cole-Hopf transform is equation-specific. |
| [B07 — Native compact-spline coefficient dynamics](../../radon_method_catalogue.md#b07-native-compact-spline-coefficient-dynamics) | Weak precision scaling | Native spline evolution | Prototype with poor precision scaling; no general global tanh claim. |
| [B08 — Native full tanh-readout dynamics](../../radon_method_catalogue.md#b08-native-full-tanh-readout-dynamics) | Native snapshot method | Flat at fixed time | Full tanh coordinate evolution; larger storage, no global spacetime export. |
| [B09 — Native reduced tanh-coordinate dynamics](../../radon_method_catalogue.md#b09-native-reduced-tanh-coordinate-dynamics) | Useful restricted evolution | Flat at fixed time | Reduced native dynamics supports specialized forward/inverse tasks. |
| [B10 — Native flat-ridge Navier–Stokes evolution](../../radon_method_catalogue.md#b10-native-flat-ridge-navierstokes-evolution) | Useful restricted evolution | Flat at fixed time | Native NS evolution, but cache and time-dependent-export limits. |
| [B11 — Native NS Taylor-in-time recurrence](../../radon_method_catalogue.md#b11-native-ns-taylor-in-time-recurrence) | Unsuccessful floor attempt | Flat at fixed time | Native Taylor integrator did not establish requested accuracy. |
| [B13 — Adaptive ridge dynamics without replay](../../radon_method_catalogue.md#b13-adaptive-ridge-dynamics-without-replay) | Failed adaptivity variant | Native snapshot | No rejected-window replay; retained as negative control. |
| [B14 — Adaptive ridge dynamics with rejected-window replay](../../radon_method_catalogue.md#b14-adaptive-ridge-dynamics-with-rejected-window-replay) | Partial adaptivity variant | Native snapshot | Replay repairs consistency but tested results not floor. |
| [C03 — Stiff QUILL reaction–diffusion with BDF](../../radon_method_catalogue.md#c03-stiff-quill-reactiondiffusion-with-bdf) | Restricted stiff integrator | Native time evolution | BDF evolution, not one global spacetime MLP. |
| [C09 — Restricted weak/entropy front-family solve](../../radon_method_catalogue.md#c09-restricted-weakentropy-front-family-solve) | Restricted weak-form demonstration | Tiny supplied tanh/ReLU family | Entropy matters; no generic shock discovery. |

## External solves and architecture departures

| Method and full description | Current utility | Architecture/scope | Why keep it |
|---|---|---|---|
| [B03 — FFT-evolved Radon heat profiles](../../radon_method_catalogue.md#b03-fft-evolved-radon-heat-profiles) | External solver hybrid | Flat compiler output | FFT evolution supplies unknown solution; not native discovery. |
| [B06 — External Fourier evolution then Radon compilation](../../radon_method_catalogue.md#b06-external-fourier-evolution-then-radon-compilation) | External solver hybrid | Flat compiler output | External Fourier trajectory then neural encoding; compiler/control only. |
| [C01 — Shared-product elliptic box solver](../../radon_method_catalogue.md#c01-shared-product-elliptic-box-solver) | Outside current architecture | Product gates | Successful PDE work cannot be counted as flat-MLP success. |
| [C02 — Mapped annulus Galerkin / Newton–CG](../../radon_method_catalogue.md#c02-mapped-annulus-galerkin--newtoncg) | Classical geometry control | Mapped Galerkin | Useful annulus comparison; not required architecture. |
| [C04 — Finite-volume Burgers with ReLU primitive](../../radon_method_catalogue.md#c04-finite-volume-burgers-with-relu-primitive) | Classical shock baseline | Finite volume plus representation | External finite-volume solve, useful reference only. |
| [C05 — Shared-product native NS and animation](../../radon_method_catalogue.md#c05-shared-product-native-ns-and-animation) | Outside current architecture | Shared products | Fluid animation is not flat-MLP solver evidence. |
| [C06 — Generic cached QUILL-product residual engine](../../radon_method_catalogue.md#c06-generic-cached-quill-product-residual-engine) | Superseded broad prototype | Shared products | Broad residual interface, architecture noncompliant. |
| [C07 — Sparse-support precision product solver](../../radon_method_catalogue.md#c07-sparse-support-precision-product-solver) | Specialized precision control | Sparse products | Sparse support discovery costs and architecture must remain explicit. |
| [C08 — Exact-polynomial residual control](../../radon_method_catalogue.md#c08-exact-polynomial-residual-control) | Classical diagnostic | Polynomial evaluator | Separates QUILL encoding from residual solving. |
| [C10 — Scaled Gaussian concentrating-core probe](../../radon_method_catalogue.md#c10-scaled-gaussian-concentrating-core-probe) | Known-field representation control | Products and supplied scaling | Not a PDE solve or Navier-Stokes blowup result. |
| [E01 — Deep tanh polarization compiler + native residual solve](../../radon_method_catalogue.md#e01-deep-tanh-polarization-compiler--native-residual-solve) | Outside current architecture | Deep tanh | Depth repair is valid MLP but fails single-hidden-layer requirement. |
| [E06 — Sequential time-slab ridge networks](../../radon_method_catalogue.md#e06-sequential-time-slab-ridge-networks) | Outside global-output requirement | Time-slab collection | Native continuation but deployment requires routing among networks. |

## Related architectures

| Method and full description | Current utility | Architecture/scope | Why keep it |
|---|---|---|---|
| [A08 — Learned latent chart + Radon LS head](../../radon_method_catalogue.md#a08-learned-latent-chart--radon-ls-head) | Outside current architecture | Deep encoder plus head | Learned chart is interesting but not one hidden layer on original inputs. |

