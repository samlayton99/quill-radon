"""Six-panel views: radial spoke wires, target surface, and four exact directions.

Reads the cached endpoint-direction fits in spoke_profiles/endpoint_direction_check.
The plotted G_j(t)=M*(g_j(t)-g_j(0))+f_hat(x0) fixes the constant ambiguity and
removes the 1/M amplitude scaling. Exactly: f_hat(x0+u)=mean_j G_j(v_j dot u).
The radial wires visualize profile parameters, not restrictions of f to those lines.
"""
from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

import spoke_profiles as analysis

OUT = analysis.OUT / "six_panel_views"
COLORS = ["#2166ac", "#d95f02", "#219653", "#8e44ad"]
SELECTED = [0,16,8,24]
LABELS = ["x axis · 0°", "y axis · 90°", "diagonal y = x · 45°", "diagonal y = −x · 135°"]
FORMULAS = {
    "gauss_bump":r"$f(x,y)=\exp[-\|(x,y)-(0.2,0.1)\|^2/0.5^2]$",
    "fast_waves":r"$f(x,y)=\cos[6\pi\|(x,y)-(0.3,-0.2)\|/\sqrt{2}]$",
    "composition":r"$f(x,y)=\exp[\sin(\pi x)\cos(\pi y)]$",
}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=analysis.OUT/"endpoint_direction_check"/"solution_M32_N128.npz"
    sol=dict(np.load(source))
    M=int(sol["M"])
    radius=.36
    t=np.linspace(-radius,radius,601)
    g,dg,g0,centers=analysis.profiles(sol,t)
    center_value=sol["bias"]+g0.sum(axis=0)
    G=M*(g-g0[:,None,:])+center_value[None,None,:]
    # Verify that normalization preserves the model, independently of the plots.
    xx=analysis.original.ball(100,radius,analysis.original.X0,np.random.default_rng(718))
    flat=np.zeros((len(xx),len(analysis.KEYS)))+sol["bias"]
    normalized=np.zeros_like(flat)
    for j,v in enumerate(sol["directions"]):
        one=np.tanh(float(sol["gamma"])*(xx@v[:,None]-sol["centers"][j]))@sol["w"][j]
        flat+=one
        normalized+=(M*(one-g0[j])+center_value)/M
    discrepancy=float(np.max(np.abs(flat-normalized)))
    assert discrepancy<1e-12
    for j,angle in zip(SELECTED,[0,90,45,135]):
        v=sol["directions"][j]
        expected=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
        assert np.linalg.norm(v-expected)<1e-14
    phi=np.linspace(0,2*np.pi,161)
    rho=np.linspace(0,radius,65)
    xp=rho[:,None]*np.cos(phi)[None,:]
    yp=rho[:,None]*np.sin(phi)[None,:]
    xy=analysis.original.X0+np.column_stack([xp.ravel(),yp.ravel()])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False,
                         "axes.titleweight":"medium"})
    metadata={"source":str(source.relative_to(analysis.original.REPO_ROOT)),
              "directions":M,"centers_per_direction":int(sol["N"]),
              "data_center":analysis.original.X0.tolist(),"display_radius":radius,
              "angle_rule":"endpoint directions (axes and diagonals included)",
              "profile_normalization":"G_j(t) = M*(g_j(t)-g_j(0)) + f_hat(x0)",
              "reconstruction":"f_hat(x0+u) = mean_j G_j(v_j dot u)",
              "normalization_check_max_abs":discrepancy,"targets":{}}
    with PdfPages(OUT/"spoke_gallery.pdf") as pdf:
        for key in FORMULAS:
            k=analysis.KEYS.index(key)
            name=analysis.TARGETS[k][1]
            fn=analysis.TARGETS[k][3]
            target=fn(xy).reshape(xp.shape)
            slices=np.stack([fn(analysis.original.X0+t[:,None]*sol["directions"][j]) for j in SELECTED])
            zlo=min(G[:,:,k].min(),target.min(),slices.min())
            zhi=max(G[:,:,k].max(),target.max(),slices.max())
            pad=.08*(zhi-zlo)
            zlim=(zlo-pad,zhi+pad)
            fig=plt.figure(figsize=(13.5,16.5),layout="constrained")
            grid=fig.add_gridspec(3,2,height_ratios=[1.35,1,1])
            wire=fig.add_subplot(grid[0,0],projection="3d",computed_zorder=False)
            surf=fig.add_subplot(grid[0,1],projection="3d",computed_zorder=False)
            for j,v in enumerate(sol["directions"]):
                if j not in SELECTED:
                    wire.plot(t*v[0],t*v[1],G[j,:,k],color="#748396",alpha=.48,lw=.9,zorder=2)
                wire.plot(t*v[0],t*v[1],np.full_like(t,zlim[0]),color="#c7cdd4",alpha=.3,lw=.5,zorder=1)
            for j,color in zip(SELECTED,COLORS):
                v=sol["directions"][j]
                wire.plot(t*v[0],t*v[1],G[j,:,k],color=color,lw=2.2,zorder=3)
                wire.plot(t*v[0],t*v[1],np.full_like(t,zlim[0]),color=color,alpha=.35,lw=.9,zorder=1)
            wire.scatter([0],[0],[center_value[k]],color="#222222",s=20,zorder=5)
            wire.set_title("All 32 spoke functions\nEach curve lies above its radial spoke",pad=12)
            wire.set_zlabel("normalized spoke value",labelpad=9)
            surf.plot_surface(xp,yp,target,cmap="viridis",rstride=1,cstride=2,
                              linewidth=0,antialiased=True,alpha=.9,zorder=1)
            # Colored cuts locate the four snapshots on the actual surface.
            for (j,color),slice_y in zip(zip(SELECTED,COLORS),slices):
                v=sol["directions"][j]
                surf.plot(t*v[0],t*v[1],slice_y,color=color,lw=1.8,zorder=3)
            surf.set_title("Actual target surface\nColored lines locate the four spatial slices",pad=12)
            surf.set_zlabel("target value",labelpad=9)
            for ax in [wire,surf]:
                ax.set_xlabel("x offset",labelpad=8);ax.set_ylabel("y offset",labelpad=8)
                ax.set_xlim(-radius,radius);ax.set_ylim(-radius,radius);ax.set_zlim(*zlim)
                ax.set_xticks([-.3,0,.3]);ax.set_yticks([-.3,0,.3])
                ax.view_init(elev=27,azim=-59)
                ax.set_box_aspect((1,1,.85))
                ax.xaxis.pane.fill=False;ax.yaxis.pane.fill=False;ax.zaxis.pane.fill=False
                ax.tick_params(labelsize=9)
            for index,(j,color,label) in enumerate(zip(SELECTED,COLORS,LABELS)):
                ax=fig.add_subplot(grid[1+index//2,index%2])
                ax.plot(t,G[j,:,k],color=color,lw=2.5)
                ax.plot(t,slices[index],color="#343a40",ls="--",lw=1.7)
                ax.scatter([0],[center_value[k]],color="#222222",s=18,zorder=4)
                ax.axvline(0,color=".8",lw=.7);ax.axhline(0,color=".85",lw=.7)
                ax.set_title(label,pad=10)
                ax.set_xlabel("signed distance t from the data center")
                ax.set_ylabel("function value")
                ax.set_xlim(-radius,radius);ax.set_ylim(*zlim)
                ax.grid(alpha=.18)
                ax.legend(handles=[Line2D([0],[0],color=color,lw=2.5,label="1D spoke function"),
                                   Line2D([0],[0],color="#343a40",ls="--",lw=1.7,label="actual target along this line")],
                          fontsize=9,loc="best",framealpha=.9)
            fig.suptitle(f"{name.capitalize()} — 4096 neurons\n"+FORMULAS[key]+"\n"
                         +f"Center (0.35, −0.25), displayed radius 0.36 · relative fit error {sol['rel_l2'][k]:.2e}\n"
                         +"Spokes scaled by 32 and aligned at the center; their projected average reconstructs the fit.",
                         fontsize=15)
            fig.savefig(OUT/f"{key}_six_panel.png",dpi=165)
            pdf.savefig(fig)
            plt.close(fig)
            metadata["targets"][key]={"relative_l2":float(sol["rel_l2"][k]),
                                       "fit_at_center":float(center_value[k]),
                                       "max_abs_center_error":float(abs(center_value[k]-fn(analysis.original.X0[None,:])[0]))}
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    np.savez_compressed(OUT/"plotted_profiles.npz",t=t,G=G,directions=sol["directions"],
                        keys=np.array(analysis.KEYS),center_value=center_value)
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":
    main()
