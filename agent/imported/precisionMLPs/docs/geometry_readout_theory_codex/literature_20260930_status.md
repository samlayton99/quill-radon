# QI, learned geometry, and optimization: literature search

Coordinator: Codex `/root`. Date: 2026-09-30.

Scope: synthesize Sam's September 30 intuitions against primary literature, anchored in uniform QI approximation and single-neuron perturbations. Research and explanation only; no new experiments, model edits, or training runs.

Status: complete. Three scoped literature reports retrieved; findings integrated into the chat explanation. No experiments were launched.

Parallel research responsibilities:
- `deep_adaptive_geometry_literature`: deeper-layer feature placement, splines, learned kernels and partitions.
- `dimension_randomness_literature`: ridge/Radon approximation, composition, intrinsic dimension, random features and scaling.
- `local_repair_optimization_literature`: perturbation coupling, conditioning, variable projection and information geometry.

Local context: read the existing theory checkpoint and Sam's September 26 driving notes. Personal-context synchronization was previously transport-limited; using the local reviewed instructions.

Output: self-contained explanation in chat, distinguishing established results, derived identities, and research hypotheses. Source reports are returned to the coordinator through native task messages.

## Principal verified sources and scope

- Chui and Mhaskar (2018), *Deep Nets for Local Manifold Learning*: explicit local coordinate charts followed by spline quasi-interpolation and coefficient combination. A construction under manifold, sampling, and smoothness assumptions; not an SGD-discovery theorem. https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2018.00012/full
- Cyr et al. (2020), *Robust Training and Initialization of Deep Neural Networks: An Adaptive Basis Viewpoint*: basis-oriented initialization and hybrid least-squares/gradient training; numerical regression and PDE evidence. https://proceedings.mlr.press/v107/cyr20a.html
- Wilson et al. (2016), *Deep Kernel Learning*: learned coordinates and kernel regression, with regular inducing grids in latent feature space. Covariance kernels are distinct from individual neuron bumps. https://proceedings.mlr.press/v51/wilson16.html
- Fan et al. (2023), *Probabilistic partition of unity networks for high-dimensional regression problems*: learned low-dimensional coordinates, partitions, and weighted polynomial least squares. https://arxiv.org/html/2210.02694v2
- Trask et al. (2022), *Hierarchical partition of unity networks: fast multilevel training*: localized polynomial solves and a multigrid-inspired training cycle. https://proceedings.mlr.press/v190/trask22a.html
- Parhi and Nowak, *What Kinds of Functions Do Deep Neural Networks Learn? Insights from Variational Spline Theory*: composed Radon-domain spline spaces with regularized representer theorems. https://arxiv.org/abs/2105.03361
- Schmidt-Hieber (2020), *Nonparametric Regression Using Deep Neural Networks with ReLU Activation Function*: approximation/statistical benefits of low-arity composition; retains an optimization-error term. https://arxiv.org/abs/1708.06633
- Nakada and Imaizumi (2020), *Adaptive Approximation and Generalization of Deep Neural Network with Intrinsic Dimensionality*: rates tied to support dimension under stated assumptions. https://jmlr.org/papers/v21/20-002.html
- Bach (2017), *On the Equivalence between Kernel Quadrature Rules and Random Feature Expansions*: adaptive sampling, spectral effective dimension, and quadrature/random-feature equivalence for fixed kernel settings. https://jmlr.org/papers/v18/15-178.html
- Rudi and Rosasco (2017), *Generalization Properties of Learning with Random Features*: balances sample size, number of features, and regularization. https://papers.nips.cc/paper_files/paper/2017/file/61b1fb3f59e28c67f3925f3c79be81a1-Paper.pdf
- Martens (2020), *New Insights and Perspectives on the Natural Gradient Method*: Fisher/Gauss-Newton relations and qualifications concerning parameterization and damping. https://jmlr.org/papers/v21/17-678.html
- Transtrum, Machta, Sethna (2011), *Geometry of nonlinear least squares with applications to sloppy models and optimization*: geometry of prediction vectors as parameters vary; distinct from input-feature manifolds. https://arxiv.org/abs/1010.1449
- Fornberg, Larsson, Flyer (2011), *Stable Computations with Gaussian Radial Basis Functions*: stable coordinates for the same approximation space in the flat-kernel regime. https://doi.org/10.1137/09076756X
- Mallat (2016), *Understanding Deep Convolutional Networks*: multiscale contraction and preservation of distinctions needed for the task; leaves weight optimization unresolved. https://pmc.ncbi.nlm.nih.gov/articles/PMC4792410/

## Synthesis checks

- QI provides a controlled approximation construction, not a proof that raw neuron coordinates are well conditioned or that uniform placement is optimal for every target.
- A single perturbed parameter produces gradients proportional to overlaps of parameter-induced function changes. This happens even in convex fixed-basis least squares; it does not by itself establish bad local minima.
- Under fixed-variance Gaussian observation noise, the same parameter-Jacobian overlap metric is Fisher information, up to the noise scale.
- A uniform grid in a known increasing scalar coordinate map produces linked nonuniform widths and spacings in the original coordinate. This is a controlled, testable continuation of QI, not an explanation already established for arbitrary Adam branches.
- Low-dimensional data support, low-order target composition, and sparse useful ridge directions are distinct ways to avoid an ambient full grid. None removes difficulty for arbitrary high-dimensional targets.
- Approximation floor, optimization gap, and statistical estimation error require separate measurements. More width and more data address different bottlenecks.
