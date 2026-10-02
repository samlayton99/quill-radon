## Use lstsq over QI
- So basically, I verified the results numerically, the machinery works.
- I found that lstsq gives much better numerical results on all fronts, 
so I tested to see if they were the same solution... they aren't, but if you project the QI solution into the null space of phi, then they are identical. so the phi matrix has super big null space and weirdly conditioned, but that conditioning is not correlated with the ability to get good precision. but it basically chooses the same solution in the null space 
- this null space is consitent across activation functions (tanh and gelu are different, side quest)
- norm of weight also drops down, the norm solution is pretty solid and not a big deal.

## it is robust to noise and it scales well
- noise in x is prety harmless
- noise in y is much harder, follows statistics 1/sqrt(n)
- noise in centers is painful
- as we scale the size and data, there are precision scaling laws for both.

## we have 3 geometry conditions that lead to machine eps precision
main idea is fixing different parts of the ideal geometry and sweeping/interpolating over different less ideal solutions, and see how the least squares is able to resolve. 

### Gamma
TLDR: gamma is the most important thing to get right. 
- gamma in the right regime is really important
- ideal lambda stays pretty constant as you increase frequency and across function types. matters about 8-10 orders of precision
- lambda is pretty consistent across all functions and all widths. Set gamma as N/8 (not W) and you get the right range. (not all activations though, but it is like .25 vs .75 for tanh and gelu respectively)
- so basically tanh and gelu are the same, just shifted lambda

### bias uniformity
Idea is to sweep from random initialization all the way to perfect interval spacing (uniformity)
- correct center placement gives you about 2 orders of magnitude of precision. pretty useful to get this right
- more uniformity is giving wider lambda error valleys, (and often deeper) across all functions
- uniformity monotonically improves precision for almost every lambda and function choice, gradual improvements the entire time, but most dramatic improvements on the last nudge to uniformity


### weight sameness/uniformity
pick a fixed l1 norm ball at radius gamma. interpolate from initialization to the uniform center of the face.
- In the right lambda regimes, identical (uniform) weights is by far the best
- In the wrong bias regimes, uniform weights can be the worst.
- sign of weight doesn't matter (doesn't matter which face you choose)
- tanh and gelu are the same, just shifted lambda like mentioned before
- as we increase uniformity, it gets worse before it gets better. best is always fully uniform weights (true for more convex functions)
- but if we hold the smallest weights, called soft-weights (freeze them) it actually provides the best solution. epecially true for convex functions. it flattens the hump.
- more weight uniformity shifts lambda up just a tad
- the l1 ball seems to be the right, not the l2 ball. If we hold l2 norm constant, it messus up near the corners. 


### Interactions
Interpolating between, we see some interesting things
- at perfect bias uniformity, this has both the best and worst solutions. depends on weight uniformity
- weight uniformity interacts with lambda regimes too. so in the right lambda regime, it is monotonically better, in the wrong regime, it is monotonically worse


### Open questions on geometry
- seems like there is a second ideal lambda regime that emerges across all functions as N gets bigger (it is like lambda = .05)
- why does the last epsilon mile of weight uniformity and bias uniformity have a step change increase in precision?
- a small defect in runge, if we cluster bias points/centers near the center (more dense at high curvature), we could get machine eps at N=32. 
- is it possible that freezing the softweights is even better? (it tentatively seems so)
- how does this softweights idea relate to multi stage fitting? 
- a few different lambda regimes spaced evenly?
- how well does this generalize (tradeoff between precision and generalization in data poor regions)


## Optimization
We now have a 4th barrier to gradient based optimizations, solving least squares. pretty obvious that it doesn't work


- QI basin is generally stable here, running adam on it doesn't ruin the basin. i.e. run adam starting at the QI solution, let it run, lstsq readout. it's fine. 
- initializing at QI solution, then train, this gives better solution even when doing only adam (no lstsq readout)
- if we only scale to right gamma alone (don't do any centered bias or uniformity) it also seems to do better, but this improvement decays as N grows. 
- adam + lstsq consistently finds a better geometry than just its initialization + lstsq. i.e. adam does contribute some geometry benefit over initializations


### Open questions
- are there any cases or variants where these optimizers can do lstsq
- what does the trained geometry look like? (bias uniformity, spread-out-ness, gamma)


## 2D
We test 4 type of 2d functions, and get some pretty good results

- there seems to be an optimal lambda and its in a similar .1 to .25 regime (but it seems to decrease maybe slightly as N increases?)
- we can drive almost everything to the precision floor here using these geometries.
- it seems to work best to concentrate them in the middle around curvature. 


## More experiments to try:
- Generalization vs precision. masking training 
- precsion (and generalization tradeoff) when holding freezing/protecting softweights
- just try it on some 2D real physics task with constructed geometry
- just try the initializations on multiple layers. shoot for it. 
- how does this work with CE/non MSE?
