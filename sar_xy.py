"""
SAR X-Y Maximum Intensity Projection — streamlined reconstruction script.
Sweeps Z, applies matched-filter focusing at each depth, then outputs
a single X-Y max-projection heatmap.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fft2, ifft2, fftshift


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_row(filepath, samples, frames):
    """Load one .bin row file → complex channel-1 data (samples, frames)."""
    try:
        data_int = np.fromfile(filepath, dtype=np.int16)
    except FileNotFoundError:
        print(f"  Warning: {os.path.basename(filepath)} missing — zeros")
        return np.zeros((samples, frames), dtype=np.complex128)

    expected_2rx = frames * samples * 4
    if len(data_int) == expected_2rx:
        ch = data_int[0::4] + 1j * data_int[2::4]
    else:
        ch = data_int[0::8] + 1j * data_int[4::8]

    out = np.zeros((samples, frames), dtype=np.complex128)
    n = len(ch)
    for x in range(frames):
        s, e = x * samples, (x + 1) * samples
        if e <= n:
            out[:, x] = ch[s:e]
    return out


def load_all(folder, pattern, samples, frames, rows):
    """Load all rows → (samples, rows, frames)."""
    data = np.zeros((samples, rows, frames), dtype=np.complex128)
    for y in range(rows):
        data[:, y, :] = load_row(
            os.path.join(folder, pattern.format(y=y + 1)), samples, frames)
    return data


# ---------------------------------------------------------------------------
# Spatial taper — reduces sidelobes from rectangular aperture edges
# ---------------------------------------------------------------------------
def apply_spatial_taper(sar_slice, taper_x=0.0, taper_y=0.15):
    """Apply independent Tukey cosine tapers to each axis of (rows, frames)."""
    ny, nx = sar_slice.shape

    def _tukey_1d(n, frac):
        w = np.ones(n)
        t = int(n * frac)
        if t < 2:
            return w
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(t) / t))
        w[:t] = ramp
        w[-t:] = ramp[::-1]
        return w

    out = sar_slice
    if taper_y > 0:
        out = out * _tukey_1d(ny, taper_y)[:, None]
    if taper_x > 0:
        out = out * _tukey_1d(nx, taper_x)[None, :]
    return out


# ---------------------------------------------------------------------------
# Matched-filter focus (single Z slice → cropped 2-D image)
# ---------------------------------------------------------------------------
def matched_filter_focus(sar_slice, dx, dy, z_mm, display_w, display_h, n_fft=1024):
    """
    Focus sar_slice at depth z_mm, return cropped image and axes.
    display_w, display_h: visible area in mm (centered on scan).
    """
    f0 = 77e9
    c = 299792458.0
    k = 2 * np.pi * f0 / c
    z0 = z_mm * 1e-3

    x_vec = dx * np.arange(-(n_fft - 1) / 2, (n_fft - 1) / 2 + 1) * 1e-3
    y_vec = dy * np.arange(-(n_fft - 1) / 2, (n_fft - 1) / 2 + 1) * 1e-3
    Xg, Yg = np.meshgrid(x_vec, y_vec)
    mf = np.exp(-1j * 2 * k * np.sqrt(Xg**2 + Yg**2 + z0**2))

    ny, nx = sar_slice.shape
    fy, fx = mf.shape

    if fx > nx:
        p = (fx - nx) // 2
        sar_slice = np.pad(sar_slice, ((0, 0), (p, fx - nx - p)), 'constant')
    elif nx > fx:
        p = (nx - fx) // 2
        mf = np.pad(mf, ((0, 0), (p, nx - fx - p)), 'constant')

    if fy > ny:
        p = (fy - ny) // 2
        sar_slice = np.pad(sar_slice, ((p, fy - ny - p), (0, 0)), 'constant')
    elif ny > fy:
        p = (ny - fy) // 2
        mf = np.pad(mf, ((p, ny - fy - p), (0, 0)), 'constant')

    img = fftshift(ifft2(fft2(sar_slice) * fft2(mf)))

    # Build full-size axes then crop to display area
    ry, rx = img.shape
    x_full = dx * np.arange(-(rx - 1) / 2, (rx - 1) / 2 + 1)
    y_full = dy * np.arange(-(ry - 1) / 2, (ry - 1) / 2 + 1)

    ix = (x_full > -display_w / 2) & (x_full < display_w / 2)
    iy = (y_full > -display_h / 2) & (y_full < display_h / 2)
    img = img[np.ix_(iy, ix)]
    return img, x_full[ix], y_full[iy]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="SAR X-Y Maximum Intensity Projection (streamlined)")

    p.add_argument('--folder', required=True)
    p.add_argument('--pattern', default='row_{y}_Raw_0.bin')

    # Radar / scan
    p.add_argument('--frames', type=int, required=True)
    p.add_argument('--rows', type=int, required=True)
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--speed', type=float, default=30)
    p.add_argument('--periodicity', type=float, default=18)
    p.add_argument('--y-step', type=float, default=2)
    p.add_argument('--slope', type=float, default=29.982,
                   help='Chirp slope MHz/us (default 29.982)')
    p.add_argument('--sample-rate', type=float, default=10000,
                   help='ADC sample rate ksps (default 10000)')
    p.add_argument('--scan-width', type=float, default=250)
    p.add_argument('--scan-height', type=float, default=250)

    # Z sweep
    p.add_argument('--zstart', type=float, default=250)
    p.add_argument('--zend', type=float, default=350)
    p.add_argument('--zstep', type=float, default=5)

    # Display / processing
    p.add_argument('--db', action='store_true')
    p.add_argument('--db-range', type=float, default=30)
    p.add_argument('--window', default='none', choices=['none', 'hanning', 'hamming'])
    p.add_argument('--taper-x', type=float, default=0.0,
                   help='X (horizontal) Tukey taper fraction (default 0 = off)')
    p.add_argument('--taper-y', type=float, default=0.15,
                   help='Y (vertical) Tukey taper fraction (default 0.15)')
    p.add_argument('-o', '--output', default='sar_xy.png')

    args = p.parse_args()

    dx = args.speed * (args.periodicity / 1000.0)
    dy = args.y_step
    K = args.slope * 1e12
    fS = args.sample_rate * 1e3
    Ts = 1.0 / fS
    c = 299792458.0
    n_fft_time = 1024
    tI = 4.5225e-10

    display_w = args.scan_width * 1.2
    display_h = args.scan_height * 1.2

    # Load
    print(f"Loading {args.rows} rows x {args.frames} frames "
          f"({args.samples} samp) from '{args.folder}'...")
    raw = load_all(args.folder, args.pattern, args.samples, args.frames, args.rows)

    # Range FFT
    print(f"Range FFT (window={args.window})...")
    if args.window != 'none':
        wfn = np.hanning if args.window == 'hanning' else np.hamming
        raw = raw * wfn(args.samples).reshape(-1, 1, 1)
    raw_fft = fft(raw, n=n_fft_time, axis=0)

    # Z sweep
    z_vals = np.arange(args.zstart, args.zend + args.zstep / 2, args.zstep)
    print(f"Z sweep {args.zstart}–{args.zend} mm, step {args.zstep} "
          f"({len(z_vals)} slices)")

    stack = []
    x_ax = y_ax = None
    for z_mm in z_vals:
        z0 = z_mm * 1e-3
        k_idx = int(round(K * Ts * (2 * z0 / c + tI) * n_fft_time))
        if k_idx < 0 or k_idx >= raw_fft.shape[0]:
            print(f"  Z={z_mm:.0f}mm  bin {k_idx} OUT OF RANGE — skipping")
            continue
        print(f"  Z={z_mm:.0f}mm  bin {k_idx}")

        sar_slice = raw_fft[k_idx, :, :]
        if args.taper_x > 0 or args.taper_y > 0:
            sar_slice = apply_spatial_taper(sar_slice, args.taper_x, args.taper_y)

        img, x_ax, y_ax = matched_filter_focus(
            sar_slice, dx, dy, z_mm, display_w, display_h)

        mag = np.abs(img)
        mag = (mag + np.fliplr(mag)) / 2.0
        stack.append(mag)

    if not stack:
        print("ERROR: No valid Z slices. Adjust --zstart / --zend.")
        return

    # Shift axes so (0,0) = bottom-left of physical scan
    x_ax = x_ax + args.scan_width / 2
    y_ax = y_ax + args.scan_height / 2

    mip = np.max(np.array(stack), axis=0)

    if args.db:
        peak = np.max(mip)
        if peak > 0:
            mip = np.clip(20 * np.log10(mip / peak + 1e-12), -args.db_range, 0)
        clabel = 'dB'
    else:
        clabel = 'Intensity'

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(x_ax, y_ax, mip, cmap='jet', shading='gouraud')
    plt.xlabel('Horizontal (mm)')
    plt.ylabel('Vertical (mm)')
    plt.title('SAR X-Y Maximum Intensity Projection')
    plt.colorbar(label=clabel)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved → {args.output}")
    plt.show()


if __name__ == '__main__':
    main()
