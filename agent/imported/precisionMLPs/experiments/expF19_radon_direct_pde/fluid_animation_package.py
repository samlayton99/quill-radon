"""Pack measured positions into a self-contained in-conversation 3D player."""
from pathlib import Path
import argparse, base64, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/fluid_animation'


def main():
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);args=p.parse_args()
    a=np.load(OUT/'trajectory.npz')
    positions=a['positions'];speeds=a['speeds'];maximum=float(np.max(speeds))
    quantized=np.round(positions/np.pi*32760).astype('<i2')
    qs=np.round(speeds/maximum*255).astype(np.uint8)
    data=dict(frames=len(positions),points=positions.shape[1],finalTime=float(a['times'][-1]),
              maxSpeed=maximum,positions=base64.b64encode(quantized.tobytes()).decode(),
              speeds=base64.b64encode(qs.tobytes()).decode())
    template=(Path(__file__).with_name('fluid_animation_view.html')).read_text()
    html=template.replace('__QUILL_DATA__',json.dumps(data,separators=(',',':')))
    if len(html.encode())>=1_000_000:raise RuntimeError('Visualization exceeds fragment size limit')
    args.destination.parent.mkdir(parents=True,exist_ok=True);args.destination.write_text(html)
    assert '__QUILL_DATA__' not in html
    main=np.load(OUT/'trajectory.npz');ref=np.load(OUT/'reference.npz');time=np.load(OUT/'time_check.npz')
    rel=lambda x,y:float(np.linalg.norm(x-y)/np.linalg.norm(y))
    meta=json.loads((OUT/'trajectory_metrics.json').read_text())
    report=dict(spatial_relative_differences=[rel(x,y) for x,y in zip(main['samples'],ref['samples'])],
                time_relative_differences=[rel(x,y) for x,y in zip(main['samples'],time['samples'])],
                check_times=[0,1,2,3,4],spatial_reference_cutoff=17,production_cutoff=13,
                production_dt=4/480,time_check_dt=4/960,
                maximum_sampled_physical_residual=meta['max_physical_residual'],
                maximum_sampled_divergence=meta['max_divergence'],
                maximum_relative_energy_defect=meta['max_relative_energy_defect'],
                particle_velocity_interpolation_max=max(meta['particle_interpolation_relative_checks']),
                position_quantization_max_absolute=float(np.max(abs(quantized.astype(float)*np.pi/32760-positions))),
                frames=len(positions),particles=positions.shape[1],fragment_bytes=len(html.encode()),
                trajectory_sha256=hashlib.sha256((OUT/'trajectory.npz').read_bytes()).hexdigest(),
                interpretation='Native QUILL PDE dynamics, with passive material sheets/tracers. Numerical time integration; no neural training/readout fit. Visualization interpolation and quantization do not affect PDE dynamics.')
    if max(report['spatial_relative_differences'])>1e-3 or max(report['time_relative_differences'])>1e-5:
        raise RuntimeError('Resolution validation failed')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
