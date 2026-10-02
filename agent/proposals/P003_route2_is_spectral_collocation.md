# P003: What "route 2" (the streamed native residual solver) actually is

Status: **proposed framing** (agent-drafted Oct 2, 2026; awaiting Sam). Serves: decision on the PDE program's direction. INBOX item 4. This is a characterization for Sam to accept or reject, not a positive result.

## Proposed statement
The Oct 1-2 "route 2" PDE solver is a Legendre/Zernike polynomial spectral least-squares collocation solver whose iterate is compiled exactly into a one-hidden-layer tanh MLP. Unknowns are total-degree-$p$ polynomial coefficients ($P=\binom{p+d}{d}$ per field). An exact Funk-Hecke/Gegenbauer identity over a tensor sphere rule with $M=(p+1)^{d-1}$ directions maps the polynomial to ridge profiles $C_n^{(d/2)}(\omega_m\cdot z)$, each QUILL-encoded into $N+2\lceil\sqrt N\rceil$ tanh neurons. The exported network is genuinely `Linear/Tanh/Linear` on the original inputs, but its function space is exactly the degree-$\le p$ polynomials (to ~1e-15), its Krylov Jacobian is the polynomial Jacobian, and accuracy is set by $p$, not by the neuron count (QUILL's convergence in $N$ plays no role; $N$ fixed at 129/257).

## Consequences against `human/REQUIREMENTS.md`
- One hidden layer: passes.
- No dressed-up solver: formally passes (no external trajectory), but the solve is a classical spectral method plus an exact compile. **Sam's call** whether this counts as "dressing up another method".
- Not brittle in $d$: fails. $M=(p+1)^{d-1}$; neurons per unknown grow 14 (2-D) -> 780 (5-D); projected 36 GB at 6-D, p16, four fields.
- Evaluation-bound: storage yes (streamed, no $Q\times P$ matrix), arithmetic no (~$K P^2$, $K$ Krylov products up to 4000).
- Plug-and-play: partial; chart, degree ladder, $N$/$\lambda$, damping class, block normalization, oversampling, block size, tolerances were set per run or family.

## Evidence that it works where it works (agent-side)
- Cold frozen-policy 3-D NS (N001): velocity 1.38e-14, pressure 1.18e-14, 128k neurons, 169 s, 548 MB. Harder variant (frequency 1.5, $\nu=0.03$): 9.2e-10, out of time.
- 2-D variable diffusion pair: 7.6e-16 / 9.7e-16 field error after the Oct 2 conditioning fix (box chart, tight inner solves), developed and tested on the same pair.
- Failures under the frozen policy: Helmholtz $n=4$ 8.6e-4, Allen-Cahn 0.53, Burgers inverse $\nu=0.02$ parameter error 2.07.
- Full audit: `agent/workspace/2026-10-02_import_audit/audit_route2_current.md`.

## Why it matters for paper 2
If accepted, the PDE story cannot rest on route 2 as "a new PINN method": the honest comparison class is spectral collocation (which the pre-Radon audit found beats QI on smooth box problems). What route 2 does contribute: an exact, conditioning-aware compile of polynomial spectral iterates into flat tanh MLPs, a bounded-memory streamed Gauss-Newton/LSMR engine, and a working inverse wrapper.
