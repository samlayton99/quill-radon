"""Compact measured Q1/Q2/Q5 summary from saved final states; no further training."""
import importlib.util,json
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('deep',Path(__file__).with_name('deep.py'));d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
plt=d.plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(16,8.6),layout='constrained')
for row,seed in enumerate([0,1]):
 m=json.loads((d.OUT/f'seed{seed}_metrics.json').read_text());a=np.load(d.OUT/f'seed{seed}_analysis.npz');q=m['qi_readout_prediction'];x=a['x'];pc=a['pc_second'];p=dict(np.load(d.OUT/f'seed{seed}.npz'))
 ax=axes[row,0];sc=ax.scatter(pc[:,0],pc[:,1],c=x,cmap='viridis',s=3);ax.set(title=f'Seed {seed}: learned second-layer curve',xlabel='Principal component 1',ylabel='Principal component 2');fig.colorbar(sc,ax=ax,label='Input x');ax.text(.04,.95,f'{m["pca"]["second"]["dims_99"]} PCs capture 99% variance\ntrained relative error {m["live"]["test"]:.2e}',transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
 ax=axes[row,1];ax.scatter(q['centers'],q['trained']['actual'],c='#ba4339',s=23,label='Learned, orientation-canonical');ax.plot(q['centers'],q['predicted'],'o-',color='#277ca3',markersize=3,lw=1,label=r'Independent QI guess: $h_j f\prime(c_j)/2$');ax.axhline(0,color='.7',lw=.8);ax.set(title=f'Derivative guess misses the learned weights',xlabel='Neuron transition center x',ylabel='Readout coefficient');ax.legend(fontsize=8,loc='lower left');ax.text(.04,.94,f'{q["eligible"]}/48 eligible neurons\ncoefficient error {100*q["trained"]["relative_coefficient_error"]:.1f}%',transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
 ax=axes[row,2];names=['Refit\noriginal','Uniform\nall','Uniform\ncrossings','Random\ncrossings','Tangent\nQI'];vals=[m['interventions']['refit_only']['test'],m['interventions']['uniform_arc_a1']['test'],m['interventions']['uniform_arc_a1_single_zero_only']['test'],m['random_single_zero_test_median'],m['selected']['tangent_qi']['test']];ax.bar(np.arange(5),vals,color=['#333333','#dd9b51','#277ca3','#aaaaaa','#287d58']);ax.set_yscale('log');ax.set_xticks(np.arange(5),names);ax.set_ylim(1e-7,3e-5);ax.set(title='Reposition features, then refit readout',ylabel='Independent test relative L2')
 for i,v in enumerate(vals):ax.text(i,v*1.15,f'{v:.1e}',ha='center',fontsize=9)
fig.suptitle('An accurate learned network has geometric structure, but its coefficients are not automatically QI samples',fontsize=15)
fig.supxlabel('Uniform = equal first-layer arclength; random = median of 10 matched placements.\nRefitted coefficient norms are about 1e7–1e8 and gains depend on the solve cutoff; improved precision does not imply improved conditioning.',fontsize=10)
fig.savefig(d.OUT/'deep_summary.png',dpi=190);fig.savefig(d.OUT/'deep_summary.pdf');plt.close(fig)
