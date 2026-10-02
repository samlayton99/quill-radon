"""Aggregate the two encoder seeds and preserve the independent PCA audit."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from manifold import OUT, data, embed, targets, plt
runs=[json.loads((OUT/f'metrics_s{s}_large.json').read_text()) for s in (0,1)]
u,x,pca,sv,mu=data()
pair=np.array([[-.8656066786497831,.003151705488562584,.7727957330644131],[-.6304931540941848,.0031728427694085576,.6962785444136483]])
z=(embed(pair)-mu)@pca
collision=dict(intrinsic_points=pair.tolist(),pca_codes=z.tolist(),pca_distance=float(np.linalg.norm(z[0]-z[1])),intrinsic_distance=float(np.linalg.norm(pair[0]-pair[1])),target_values=targets(pair).tolist(),method='Independent input-only root search; supplied pair independently re-evaluated here. Numerical collision, not an interval-certified root.')
(OUT/'pca_collision.json').write_text(json.dumps(collision,indent=2))
fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
colors=['#666666','#db8c20','#147d92','#8958b3']
for j,ax in enumerate(axes):
    values=np.array([[r['best'][j]['test'] for r in run['records']] for run in runs]);means=values.mean(0)
    ax.bar(np.arange(4),means,color=colors)
    ax.scatter(np.arange(4)-.055,values[0],c='black',s=18,zorder=5,label='Encoder seed 0')
    ax.scatter(np.arange(4)+.055,values[1],c='white',edgecolor='black',s=22,zorder=5,label='Encoder seed 1')
    ax.set_yscale('log');ax.set_xticks(range(4),['True 3D\ncoordinates','PCA\n3D','Learned\n3D chart','Original\n24D']);ax.set_title(['Product wave','Composition','Gaussian bump'][j]);ax.set_ylabel('Held-out relative L2');ax.grid(axis='y',alpha=.2)
    for i,v in enumerate(means):ax.text(i,v*1.55,f'{v:.1e}',ha='center',fontsize=10)
    ax.set_ylim(min(means)/2,max(means)*5)
axes[1].legend(loc='lower right',fontsize=9)
fig.suptitle('Compress first, then fit 2,048 ridge features — three tasks, no labels used for compression',fontsize=13)
fig.savefig(OUT/'comparison.png',dpi=180);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
lines=['# Learned 3D chart and Radon-style ridge head on a noiseless 24D manifold','',
'Codex experiment, 2026-09-30. Question 3. Two independent encoder initializations; one shared fixed train/validation/test split.','',
'**Yes, the hybrid works on this controlled manifold, but compression quality controls its precision.** The learned encoder supports three separate target functions, reaching relative L2 errors from 1.51e−4 to 1.48e−3. Using the true coordinates on the identical head budget reaches 3.12e−7 to 1.70e−5. PCA compression folds the manifold and fails badly. None of these results establishes a universal manifold learner.','',
'![Held-out comparisons](comparison.png)','',
'## Protocol and scope','',
'The intrinsic variable u is uniform on [−1,1]^3. The 24D embedding concatenates u, quadratic and product terms, sines/cosines, and mixed cubic/trigonometric terms, then applies a fixed orthogonal mixing. It is an injective graph embedding, and a favorable global linear projection back to u exists. Thus this tests curved-manifold chart recovery, not an arbitrary topology or an embedding with no linear inverse chart. The encoder never receives u or any target labels.','',
'There are4,096 input-training points,1,024 validation points, and4,096 test points, generated with independent scrambled Sobol sequences. The training-input PCA projection initializes a residual nonlinear autoencoder:24→64→64→3 encoder and3→64→64→24 decoder, tanh hidden layers, float64 CPU. Each seed uses6,000 Adam steps followed by40 blocks of up to100 L-BFGS iterations. Validation reconstruction selects the checkpoint. See config_s0.json and config_s1.json for settings.','',
'Each representation receives128 directions and16 evenly spaced tanh centers per direction, with a25% collar around the training projection range. Latent 3D directions use the existing spherical-Fibonacci rule; ambient24D directions use the existing seeded Gaussian-direction rule. The bandwidth gamma*h is selected from{.15,.25,.4}, and the SVD cutoff from{1e−8,1e−10,1e−12,1e−14}, using validation error separately for each downstream target. The head is a frozen finite Radon-style ridge dictionary plus a fitted output bias. It is a least-squares head, not the analytic no-solve Radon construction tested separately.','',
'The controls match2,048 ridge features plus a bias. They do not match total model parameters or end-to-end training cost: learned compression has an additional encoder and its training. The ambient baseline is a fixed ridge bank, not an optimized deep24D network. The encoder is shared across all three targets, but the selected head bandwidth may differ by target. All reported function errors are unsquared test relative L2.','',
'## Results','',
'| Coordinates | Product wave | Composition | Gaussian bump |','|---|---:|---:|---:|']
for k,label in enumerate(['True 3D coordinates','PCA 3D','Learned 3D, seed 0','Learned 3D, seed 1','Ambient 24D']):
    s=1 if k==3 else 0;idx=[0,1,2,2,3][k]
    v=[r['test'] for r in runs[s]['records'][idx]['best']]
    lines.append('| '+label+' | '+' | '.join(f'{a:.6g}' for a in v)+' |')
lines += ['',
'The target functions are sin(pi*u1)*cos(pi*u2), exp(.5*sin(pi*u1)+.3*u2*u3), and exp(−2||u−(.2,−.15,.1)||²). All targets vary in, or are evaluated across, the same3D domain. The product wave happens not to depend on u3; the other targets do. The encoder sees none of these labels during representation learning.','',
'The two learned encoders reconstruct held-out ambient inputs to relative errors 0.00450 and 0.00537. These are reconstruction errors, not proven irreducible downstream error floors: a different decoder or richer downstream basis can improve despite the same encoder. The learned latent coordinate is not equal to u. Its sampled intrinsic Jacobian has minimum singular values 0.885 and 0.880 and median condition numbers 1.304 and 1.276, across256 unseen points. This shows sampled local regularity; it does not prove global injectivity.','',
'## A specific explanation for PCA failure','',
'PCA retains64.99% of input variance, but its3D coordinate map has Jacobian determinants from −1.655 to 6.540. A smooth injective map between connected open subsets of equal dimension cannot reverse orientation between regular points. Thus this projection cannot be a globally injective chart. Low dimension or high retained variance does not establish preservation of the information needed by the target.','',
'The independent audit also located two numerical preimages with almost identical PCA codes. Re-evaluation gives:', '',
'```json',json.dumps(collision,indent=2),'```','',
'The product-wave values differ by about0.507 despite a PCA-code difference near1e−13. This is direct numerical evidence of task-relevant folding, in addition to the orientation argument. An isolated near-collision by itself is not a numerical lower bound on population error.','',
'## What this says about the theory','',
'A learned chart plus a classical approximation family is a workable hybrid. The three stages must be separated: preserve target-relevant distinctions; express the target simply in the retained coordinates; then resolve it with enough directions and centers. A small latent dimension does not establish any of the three. The oracle/learned gap is a measured precision gap, not evidence that Radon approximation fails or that the encoder has necessarily discarded all of that information.','',
'An early24-directions×24-centers pilot was angular-resolution limited even on the true coordinates. The final128×16 allocation was chosen after this pilot, and all four representations receive the same final allocation. Results are exploratory, not a preregistered confirmation. Per-target bandwidth/cutoff selection is validation-only; test values never select those settings.','',
'Reproduction: run manifold.py train --steps 6000 --seed 0 (and seed 1), then manifold.py evaluate --large --seed 0 (and seed 1), then manifold_summary.py. Explicit shell commands are in the root experiment README. Saved checkpoints, coordinates, full sweeps, histories, and metrics are beside this report. Independent read-only audit found no target leakage or error-metric bug.']
text='\n'.join(lines)+'\n'
# Restore natural spacing after compact numeric expressions in prose.
for a,b in [('are4,096','are 4,096'),(',1,024',', 1,024'),('and4,096','and 4,096'),('with24','with 24'),('encoder:24','encoder: 24'),('and3→','and 3→'),('uses6,000','uses 6,000'),('by40','by 40'),('to100','to 100'),('receives128','receives 128'),('and16','and 16'),('a25%','a 25%'),('from{','from {'),('match2,048','match 2,048'),('across256','across 256'),('retains64.99','retains 64.99'),('its3D','its 3D'),('about0.507','about 0.507'),('near1e','near 1e'),('early24','early 24'),('final128','final 128'),('same3D','same 3D'),('deep24D','deep 24D'),('latent3D','latent 3D'),('ambient24D','ambient 24D')]:text=text.replace(a,b)
(OUT/'report.md').write_text(text)
print(json.dumps(collision,indent=2))
