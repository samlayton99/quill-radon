"""Render the recorded QUILL 3D trajectory, without advancing any dynamics.

Input: trajectory.npz with positions[F,738,3], speeds[F,738], times[F].
Particles 0:289 and 289:578 are two 17x17 material sheets; 578: are tracers.
Periodic crossings are broken, never connected by a spurious cube-wide chord.
Matplotlib/Pillow supply the scientific rendering. An optional macOS-native
AVFoundation encoder supplies H.264 without ffmpeg or additional packages.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/codex-fluid-preview-mpl')

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
import time
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/checkpoint_F_applications/expF19_radon_direct_pde/fluid_animation'

SWIFT_ENCODER = r'''
import Foundation
import AVFoundation
import CoreVideo
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum EncodeError: Error { case failed(String) }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw EncodeError.failed(message) }
}
func encode() throws {
    let args = CommandLine.arguments
    try require(args.count == 8, "usage: encoder frames output width height fps count decodedFirst")
    let folder = args[1], output = URL(fileURLWithPath: args[2])
    let width = Int(args[3])!, height = Int(args[4])!, fps = Int32(args[5])!, count = Int(args[6])!
    try? FileManager.default.removeItem(at: output)
    let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
    writer.shouldOptimizeForNetworkUse = true
    let settings: [String: Any] = [
        AVVideoCodecKey: AVVideoCodecType.h264,
        AVVideoWidthKey: width, AVVideoHeightKey: height,
        AVVideoCompressionPropertiesKey: [
            AVVideoAverageBitRateKey: 5_000_000,
            AVVideoExpectedSourceFrameRateKey: Int(fps),
            AVVideoMaxKeyFrameIntervalKey: Int(fps),
            AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
        ]
    ]
    let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
    input.expectsMediaDataInRealTime = false
    let attrs: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferCGImageCompatibilityKey as String: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
    ]
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attrs)
    try require(writer.canAdd(input), "Cannot add video writer input")
    writer.add(input)
    try require(writer.startWriting(), "startWriting failed: \(String(describing: writer.error))")
    writer.startSession(atSourceTime: .zero)
    for index in 0..<count {
        let deadline = Date().addingTimeInterval(30)
        while !input.isReadyForMoreMediaData {
            try require(writer.status != .failed, "Writer failed: \(String(describing: writer.error))")
            try require(Date() < deadline, "Encoder did not become ready")
            Thread.sleep(forTimeInterval: 0.001)
        }
        try autoreleasepool {
            let url = URL(fileURLWithPath: folder).appendingPathComponent(String(format: "%05d.png", index))
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(src, 0, nil),
                  let pool = adaptor.pixelBufferPool else { throw EncodeError.failed("Cannot load frame or buffer pool") }
            var optional: CVPixelBuffer?
            try require(CVPixelBufferPoolCreatePixelBuffer(nil, pool, &optional) == kCVReturnSuccess,
                        "Pixel buffer allocation failed")
            let buffer = optional!
            CVPixelBufferLockBaseAddress(buffer, [])
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: width,
                                          height: height, bitsPerComponent: 8,
                                          bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                          space: CGColorSpaceCreateDeviceRGB(),
                                          bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue |
                                                      CGBitmapInfo.byteOrder32Little.rawValue)
            else { throw EncodeError.failed("Cannot create drawing context") }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            try require(adaptor.append(buffer, withPresentationTime: CMTime(value: Int64(index), timescale: fps)),
                        "Appending frame failed: \(String(describing: writer.error))")
        }
    }
    input.markAsFinished()
    let sem = DispatchSemaphore(value: 0)
    writer.finishWriting { sem.signal() }
    try require(sem.wait(timeout: .now()+60) == .success, "finishWriting timed out")
    try require(writer.status == .completed, "Writer did not complete: \(String(describing: writer.error))")
    let asset = AVURLAsset(url: output)
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    let first = try generator.copyCGImage(at: .zero, actualTime: nil)
    let decoded = URL(fileURLWithPath: args[7])
    guard let dest = CGImageDestinationCreateWithURL(decoded as CFURL, UTType.png.identifier as CFString, 1, nil)
    else { throw EncodeError.failed("Cannot create verification PNG") }
    CGImageDestinationAddImage(dest, first, nil)
    try require(CGImageDestinationFinalize(dest), "Cannot write verification PNG")
    let report: [String: Any] = ["encoder": "macOS AVFoundation H.264", "frames": count,
                                "width": width, "height": height, "fps": Int(fps),
                                "duration_seconds": Double(count)/Double(fps),
                                "decoded_first_width": first.width, "decoded_first_height": first.height]
    let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}
do { try encode() } catch { fputs("\(error)\n", stderr); exit(1) }
'''


def encoder_binary():
    if platform.system() != 'Darwin' or not Path('/usr/bin/swiftc').exists():
        return None
    digest = hashlib.sha256(SWIFT_ENCODER.encode()).hexdigest()[:12]
    folder = Path('/private/tmp')/f'codex-quill-fluid-encoder-{digest}'
    folder.mkdir(exist_ok=True)
    binary = folder/'encoder'
    if not binary.exists():
        source = folder/'encode.swift'
        source.write_text(SWIFT_ENCODER)
        command = ['/usr/bin/swiftc', '-O', '-module-cache-path', str(folder/'module_cache'),
                   str(source), '-o', str(binary)]
        result = subprocess.run(command, text=True, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError('Native encoder compilation failed: '+result.stderr[-5000:])
    return binary


def encode_mp4(binary, frames, output, width, height, fps, count, verify_png):
    command = [str(binary), str(frames), str(output), str(width), str(height), str(fps), str(count), str(verify_png)]
    process = subprocess.run(command, text=True, capture_output=True, timeout=180)
    if process.returncode:
        raise RuntimeError('Native encoding failed: '+process.stderr[-3000:])
    result = json.loads(process.stdout.strip().splitlines()[-1])
    source = np.asarray(Image.open(frames/'00000.png').convert('RGB'), float)
    decoded = np.asarray(Image.open(verify_png).convert('RGB'), float)
    result['decoded_first_mean_absolute_rgb_error'] = float(np.mean(abs(source-decoded)))
    result['decoded_first_flipped_mean_absolute_rgb_error'] = float(np.mean(abs(source[::-1]-decoded)))
    if result['decoded_first_mean_absolute_rgb_error'] > 8:
        raise RuntimeError('Encoded first frame differs excessively from the scientific rendering')
    return result


def prepare():
    """Check the native codec using a local non-simulation orientation fixture."""
    began = time.perf_counter()
    binary = encoder_binary()
    if binary is None:
        return {'encoder': None, 'gif_available': True}
    with tempfile.TemporaryDirectory(prefix='codex-fluid-encoder-check-') as name:
        folder = Path(name)
        for i in range(3):
            im = Image.new('RGB', (320, 240), '#102535')
            draw = ImageDraw.Draw(im)
            draw.rectangle((0, 0, 159, 80), fill='#fb6a53')
            draw.rectangle((160, 160, 319, 239), fill='#59deed')
            draw.text((20, 105), 'ENCODER CHECK '+str(i), fill='white')
            im.save(folder/f'{i:05d}.png')
        result = encode_mp4(binary, folder, folder/'check.mp4', 320, 240, 15, 3, folder/'decoded.png')
        result.update(binary=str(binary), preparation_seconds=time.perf_counter()-began)
    return result


def sheet_triangles():
    faces = []
    for i in range(16):
        for j in range(16):
            a = 17*i+j
            faces.extend([(a, a+1, a+17), (a+1, a+18, a+17)])
    return np.asarray(faces)


def unwrapped_faces(points, triangles):
    faces = points[triangles]
    edges = np.stack([faces[:, 0]-faces[:, 1], faces[:, 1]-faces[:, 2], faces[:, 2]-faces[:, 0]])
    valid = np.max(abs(edges), axis=(0, 2)) <= np.pi
    return faces[valid], int(np.sum(~valid))


def render(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from matplotlib.patches import Rectangle

    started = time.perf_counter()
    path = Path(args.input)
    with np.load(path, allow_pickle=False) as archive:
        positions, speeds, times = (np.array(archive[k]) for k in ('positions', 'speeds', 'times'))
    if positions.ndim != 3 or positions.shape[1:] != (738, 3):
        raise ValueError('Expected positions[F,738,3]')
    if speeds.shape != positions.shape[:2] or times.shape != (len(positions),):
        raise ValueError('Inconsistent speed/time dimensions')
    if not all(np.all(np.isfinite(v)) for v in (positions, speeds, times)):
        raise ValueError('Trajectory contains nonfinite values')
    if np.any(np.diff(times) <= 0) or np.any(abs(positions) > np.pi+1e-5) or np.any(speeds < 0):
        raise ValueError('Expected increasing times, nonnegative speed, coordinates in [-pi,pi]')
    OUT.mkdir(parents=True, exist_ok=True)
    frames = len(times)
    bg, foreground, muted = '#08121e', '#eaf2fa', '#8497ac'
    sheet_colors = ['#38c8d5', '#faab58']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': foreground, 'axes.labelcolor': muted,
                         'figure.facecolor': bg, 'savefig.facecolor': bg})
    fig = plt.figure(figsize=(args.width/100, args.height/100), dpi=100, facecolor=bg)
    ax = fig.add_axes([.005, .06, .76, .84], projection='3d', computed_zorder=False)
    ax.set_facecolor(bg)
    ax.set_box_aspect((1, 1, 1), zoom=1.08)
    ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), zlim=(-np.pi, np.pi))
    ax.view_init(elev=22, azim=-54)
    ax.set_axis_off()
    fig.text(.045, .952, '3D flow from the constructed solver', fontsize=24, weight='bold', va='center')
    fig.text(.046, .915, 'NAVIER–STOKES  /  PERIODIC DOMAIN  /  PASSIVE MATERIAL TRACKING',
             fontsize=9, color=muted, va='center')

    # A thin wire box and floor grid make depth and periodic crossings legible.
    corners = np.asarray([[a, b, c] for a in (-np.pi, np.pi) for b in (-np.pi, np.pi) for c in (-np.pi, np.pi)])
    edges = [np.array([a, b]) for i, a in enumerate(corners) for b in corners[i+1:] if np.sum(a != b) == 1]
    ax.add_collection3d(Line3DCollection(edges, colors='#324459', linewidths=.8, alpha=.65, zorder=1))
    grid = []
    for s in np.linspace(-np.pi, np.pi, 5):
        grid.extend([np.array([[-np.pi, s, -np.pi], [np.pi, s, -np.pi]]),
                     np.array([[s, -np.pi, -np.pi], [s, np.pi, -np.pi]])])
    ax.add_collection3d(Line3DCollection(grid, colors='#213044', linewidths=.55, alpha=.65, zorder=1))
    for label, point in [('x', (np.pi+.35, -np.pi, -np.pi)),
                         ('y', (-np.pi, np.pi+.35, -np.pi)), ('z', (-np.pi, -np.pi, np.pi+.35))]:
        ax.text(*point, label, color=muted, fontsize=12)

    fig.text(.775, .79, 'SIMULATION TIME', fontsize=9, color=muted)
    time_label = fig.text(.775, .727, '', fontsize=37, family='DejaVu Sans Mono', weight='bold')
    fig.text(.778, .682, f'of {times[-1]:.2f}', fontsize=10, color=muted)
    progress_bg = Rectangle((.777, .648), .17, .005, transform=fig.transFigure, color='#203249')
    progress = Rectangle((.777, .648), 0, .005, transform=fig.transFigure, color='#38c8d5')
    fig.add_artist(progress_bg); fig.add_artist(progress)
    fig.text(.777, .584, 'TWO MATERIAL SHEETS', fontsize=9, color=muted)
    for i, color in enumerate(sheet_colors):
        fig.add_artist(Rectangle((.778, .542-i*.045), .012, .012, transform=fig.transFigure,
                                 facecolor=color, edgecolor='none', alpha=.9))
        fig.text(.801, .542-i*.045, f'Sheet {i+1}   ·   17 × 17 markers', fontsize=9)
    fig.text(.777, .422, '160 TRACER PARTICLES', fontsize=9, color=muted)
    fig.text(.777, .389, 'Color indicates particle speed', fontsize=9, color=foreground)
    cmap = matplotlib.colormaps['plasma']
    vmax = max(float(np.quantile(speeds[:, 578:], .995)), 1e-12)
    norm = colors.Normalize(0, vmax, clip=True)
    cax = fig.add_axes([.778, .346, .17, .014])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=cmap, extent=(0, vmax, 0, 1))
    cax.set_yticks([]); cax.set_xticks([0, vmax], ['0', f'{vmax:.2f}'])
    cax.tick_params(axis='x', colors=muted, labelsize=8, length=0, pad=5)
    for spine in cax.spines.values():
        spine.set_visible(False)
    speed_label = fig.text(.777, .279, '', color=foreground, fontsize=11)
    fig.text(.777, .214, f'Viscosity  ν = {args.viscosity:g}', color=muted, fontsize=10)
    fig.text(.777, .178, 'Coordinates: [−π, π]³', color=muted, fontsize=10)
    fig.text(.045, .037, 'Surfaces and dots are passive markers. Gaps show crossings of the periodic boundary.',
             fontsize=9, color=muted)
    fig.text(.948, .037, 'QUILL', fontsize=10, color='#38c8d5', ha='right', weight='bold')

    triangles = sheet_triangles()
    artists, gif_frames = [], []
    reports, palette = [], None
    poster_index = min(frames-1, int(round(.72*(frames-1))))
    compiler_report = None
    try:
        binary = encoder_binary()
    except Exception as exc:
        binary = None
        compiler_report = str(exc)
    with tempfile.TemporaryDirectory(prefix='codex-fluid-preview-frames-') as folder_name:
        folder = Path(folder_name)
        first_path = folder/'00000.png'
        for frame in range(frames):
            for artist in artists:
                artist.remove()
            artists = []
            dropped = 0
            for i, color in enumerate(sheet_colors):
                faces, excluded = unwrapped_faces(positions[frame, i*289:(i+1)*289], triangles)
                dropped += excluded
                base = np.array(colors.to_rgb(color))
                rgba = np.tile(np.r_[base, .19], (len(faces), 1))
                if len(faces):
                    normal = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
                    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-15)
                    shade = .65+.35*abs(normal@np.array([.3, -.4, .866]))
                    rgba[:, :3] *= shade[:, None]
                surface = Poly3DCollection(faces, facecolors=rgba,
                                           edgecolors=(*base, .13), linewidths=.28,
                                           zsort='average', zorder=2+i*.01)
                ax.add_collection3d(surface); artists.append(surface)
            begin = max(0, frame-args.trail_frames)
            trail = positions[begin:frame+1, 578:]
            segment_count = 0
            if len(trail) > 1:
                all_segments = np.stack([trail[:-1], trail[1:]], axis=2)
                valid = np.max(abs(trail[1:]-trail[:-1]), axis=2) <= np.pi
                segment_colors = cmap(norm(speeds[begin+1:frame+1, 578:]))
                segment_colors[:, :, 3] = np.linspace(.04, .72, len(trail)-1)[:, None]
                line = Line3DCollection(all_segments[valid], colors=segment_colors[valid], linewidths=.85, zorder=4)
                ax.add_collection3d(line); artists.append(line)
                segment_count = int(np.sum(valid))
            current = positions[frame, 578:]
            particle_colors = cmap(norm(speeds[frame, 578:]))
            for size, alpha in [(55, .055), (22, .15), (5.2, .95)]:
                color = particle_colors.copy(); color[:, 3] = alpha
                dots = ax.scatter(*current.T, s=size, c=color, depthshade=False, linewidths=0, zorder=5)
                artists.append(dots)
            time_label.set_text(f'{times[frame]:.2f}')
            fraction = (times[frame]-times[0])/max(times[-1]-times[0], 1e-15)
            progress.set_width(.17*fraction)
            speed_label.set_text(f'Median speed  {np.median(speeds[frame, 578:]):.3f}')
            fig.canvas.draw()
            image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3]).copy()
            image.save(folder/f'{frame:05d}.png')
            if frame == poster_index:
                image.save(OUT/'poster.png')
            small = image.resize((args.gif_width, int(round(args.height*args.gif_width/args.width))), Image.Resampling.LANCZOS)
            if palette is None:
                palette = small.quantize(colors=256)
            gif_frames.append(small.quantize(palette=palette, dither=Image.Dither.NONE))
            reports.append(dict(time=float(times[frame]), dropped_periodic_triangles=dropped,
                                visible_triangles=2*len(triangles)-dropped, trail_segments=segment_count))
            if frame % 20 == 0 or frame == frames-1:
                print(json.dumps({'rendered': frame+1, 'frames': frames, 'seconds': round(time.perf_counter()-started, 2)}), flush=True)
        plt.close(fig)
        durations = np.diff(np.round(np.arange(frames+1)*100/args.fps)*10).astype(int).tolist()
        gif_frames[0].save(OUT/'preview.gif', save_all=True, append_images=gif_frames[1:],
                           duration=durations, loop=0, optimize=False, disposal=2)
        encoded = None
        if binary is not None:
            try:
                encoded = encode_mp4(binary, folder, OUT/'preview.mp4', args.width, args.height,
                                     args.fps, frames, folder/'decoded-first.png')
            except Exception as exc:
                compiler_report = str(exc)
                (OUT/'preview.mp4').unlink(missing_ok=True)
    with Image.open(OUT/'preview.gif') as gif:
        gif_count = gif.n_frames
        gif_milliseconds = 0
        for i in range(gif_count):
            gif.seek(i); gif_milliseconds += gif.info.get('duration', 0)
    if gif_count != frames:
        raise AssertionError('GIF dropped or merged unexpected source frames')
    result = dict(source=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                  source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_shape=list(positions.shape), source_time_interval=[float(times[0]), float(times[-1])],
                  source_frames=frames, material_sheet_markers=2*289, passive_tracers=160,
                  scientific_rendering='Fixed camera; piecewise-linear sampled material sheets; actual saved trajectories only',
                  periodic_rule='Exclude each triangle/trail segment if any edge coordinate jump exceeds pi',
                  color_scale=dict(quantity='particle speed', maximum=vmax, clips_above_quantile=.995),
                  viscosity_label=args.viscosity, fps=args.fps, movie=encoded,
                  gif=dict(frames=gif_count, milliseconds=gif_milliseconds, width=args.gif_width),
                  encoder_error=compiler_report, poster_frame=poster_index, per_frame=reports,
                  render_seconds=time.perf_counter()-started,
                  peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if platform.system()=='Darwin' else 1024),
                  artifact_bytes={name:(OUT/name).stat().st_size for name in ('preview.mp4','preview.gif','poster.png') if (OUT/name).exists()},
                  scope='This script performs visualization only; it neither substitutes a fluid solver nor changes the supplied particles.')
    (OUT/'render_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'per_frame'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(OUT/'trajectory.npz'))
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=800)
    parser.add_argument('--gif-width', type=int, default=768)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--trail-frames', type=int, default=22)
    parser.add_argument('--viscosity', type=float, default=.075)
    args = parser.parse_args()
    if args.width % 2 or args.height % 2 or args.fps < 1:
        parser.error('Even dimensions and positive fps required')
    if args.prepare:
        print(json.dumps(prepare(), indent=2))
    else:
        render(args)
