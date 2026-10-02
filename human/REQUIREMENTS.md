# Requirements for the Radon / physics program

**Status:** Sam's words, collected verbatim from the Codex sessions of Oct 1-2, 2026 (times UTC) by an agent on Oct 2. The checklist below is the agent's condensation of these quotes; **Sam should confirm or edit it** (INBOX item 1). Only Sam changes this file.

## The checklist (condensed; awaiting Sam's confirmation)

1. **One hidden layer.** The deployed representation is a single-hidden-layer MLP on the original inputs, with one or several linear outputs. No product gates, deep compilers, time slabs, local/windowed networks, or learned input encoders.
2. **No dressed-up solver.** The network must not be the encoding of a solution computed by another numerical method. Equation + domain + boundary/initial conditions + data -> network.
3. **Memory is the bottleneck; time can give.** Cost should be bound asymptotically by evaluation/inference, not by a global readout solve. Small tractable least-squares pieces are fine.
4. **Not brittle in $d$.** Numerical help is allowed only if it does not blow up with dimension more than the representation already does.
5. **General and plug-and-play.** Works across equations without per-equation special casing or thinking through edge cases; may adapt itself. No overfitting to a few examples.
6. **Accuracy and scaling.** Machine-epsilon floor (1e-14 acceptable) with the right scaling law in neurons.
7. **What PINNs promise, honestly.** Forward, inverse (clean and noisy, no cheating or leakage), data anchoring, high dimension within the honest limits of the curse of dimensionality, hard problems (discontinuities, blowup).
8. **Known-function construction is first-class.** The analytic Radon construction (no training, no fit) is its own result, separate from the PDE problem.
9. **Noise.** Judge noisy problems by statistical accuracy and physical consistency, not by demanding precision beyond what the data supply.

## Verbatim quotes

**Oct 1 03:19.** "okay, I LOVE that we were able to do the radon without any training. this is incredible. Think about how to do this with pinns. is there an obvious way to do a full 3d navier stokes using this? ... (we did a least squares construction, but I want to do it without least squares, using our radon intuition). The problem was that solving these networks was too big, so we couldn't get anything more than 3 inputs reliably. but with the construction equation, we can actually make this work." ... "We learned that resolution is limited by resolution within directions and resolution among number of directions, keep that in mind. Also we had some theory about the optimal tradeoff between directions and number of centers."

**03:41.** "we need this to work on nonlinear pdes, things that actually matter."

**03:44.** "look up the halo node stuff. like really look it up. it evolved and we figured out the bets way to do it. also did you do any sweeps on lambda? like really apply the previous QUILLs findings to make sure this is as best as possible (also halo was sqrt(N))"

**04:02.** "in order for this to be an actual viable method or useful thing, it needs to work without any numerical solve. is there not a mathematical way to make this happen? like really, the method is dead in the water if we have to just encode a numerical solver into it. the point is we should be able to use our method with the dynamics alone. and it can't be for just linear pdes, but all types of pdes." ... "I am fine with breaking down small least squares solves or what not, so long as it is tractable. we should only be limited by our ability to do inference, not solve a readout."

**04:26.** "list out all the advantages or hopes of pinns, and then see if we can capture those with our construction. In someways we are way better than traditional pinns with adam + lbfgs. but I want to be better in every category."

**04:47.** "1) we need to get this to work as a general method, where the setup is very routine and applicable. 2) I want to see just how viable this method is up to how many dimensions is realistic? 3) I want to see this work on much harder problems. how does it handle discontinuities, blowups, etc."

**07:14.** "No, I want a general pinns solver. I genuinely think we can be the ones to fully solve this. There is a genuinely general way to set this up, that should work for just about everything, with only pathological and non interesting cases. push it until we get there, and loose all forms of project specific specification."

**08:05.** "I care much less about time and much more about accuracy. If we have enough neurons, we should be able to scale to any arbitrary precision. The right scaling laws is what I really care about." / "I need navier stokes to machine epsilon floor with a general solver"

**09:09.** "1e-14 is pretty good. if you can do better, that's great, but 1e-14 is basically there." ... "I told you this was garbage, that it must be a true PINN, not some other method that we secretly dress up with our network."

**09:47.** "either 1. you go in and utilize depth, and figure out how to get this processes into a deeper mlp ... route 2) in this route I want you to look at the theory more and figure out a more true way to use the actual radon mlp. I know we can do it. if we had the function it works." (Sam then chose route 2.)

**10:01.** "Keep tanh/GELU as the main target; show squared ReLU separately"

**21:08.** "is the current method is better than least squares over all the neurons together? that was the limit, and it was terrible. the radon theory we came up with lets us solve it for a single direction right? so that gets rid of the computation wall when the function is known. the point is we want to be bound (at least in terms of assymptotically) by the evaluation, not the construction."

**23:48.** "is there a way to do it without doing a pde numerical run? ... I am wondering if we can natively use it, in a way that scales the asymptotics in a way that is realistic (without relying on another method's solve). Like I am fine with some fitting, so long as it is asymptotically better."

**23:58.** "obviously our approach would have to let it scale and get to the floor. the point is that we can let it get longer on time, but memory is the bottleneck. obvioulsy we don't want crazy bad temporal complexity asymptotics, but if we have to budge somewhere, time is okay as well. memory is the bottleneck."

**Oct 2 02:46.** "we need this to reasonably and reliable work at higher dimensions and for harder problems, ones that people would actually want to solve in the real world. so keep going until you have met that, while still staying true to the metrics, memory, and other requirements that we have"

**04:07.** "I want to see this work on really hard problems, doing inverse problems extremely well, with no cheating or leakage, where it can do all the things pinns promises. ... don't let it get weird and pathological. this needs to be a general setup that can work for anything. make sure you are not overfitting to a few examples. also the curse of dimensionality will eventually bite us. ... under only general assumptions as a general method, we will fall subject to it. ... I just want to see what our current limits are without exploiting any special structure."

**05:27.** "we need something that would work on a blind pde. it should be a general method. just theoretically saying we found something that expresses it, is not as interesting as an actual method"

**05:54.** "numerical work: remember, numerical work must not be brittle to d. that is the principle to keep in mind. we want to scale up and not be crushed by d more than we have to. so numerical help is fine so long as it doesn't blow up d more than it already is. / one global representation: it must be a 1 layer mlp. that is a hard requirement. / it should be able to work on arbitrary functions. Like I don't want a method that is brittle to one equation dynamic vs another. we can have the method adapt itself, but like it should be plug and play without having to think through edge cases. make sense? / agreed on the noise."

**06:07.** "top 3 for each class of problem it solves. ... (like I really like the original theory. that was the first time we actually were able to do arbitrarily large functions with higher d. that was brilliant. so we should have the methods for general approximation when function is known, that is the theoretical one, one for noisy measurements, one for clean inverse problems, one for noisy inverse problems, etc.)"

**Oct 2 (this repo's setup).** The human/agent split: "the only conclusions that last and live are the ones that the human side claims, the agent side is preliminary support/evidence" ... "if I really need to see something, it must be on the human side."

Source (local only, not in the public repo): `agent/workspace/2026-10-02_import_audit/sessions/sam_messages_radon_sessions.md` (all of Sam's messages from the relevant sessions).
