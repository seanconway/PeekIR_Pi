#!/usr/bin/env python3
"""
SAR Complete — run a capture with ``sar_coordinator`` and immediately
reconstruct a visualization with ``sar_reconstruct``.

Reconstruction parameters are derived from the capture settings so the same
CLI that drives the scan also drives the image reconstruction. Visualization
artifacts (PNG heatmaps and the Plotly HTML) are written into the same
``scan_N`` folder that holds the .bin row files.

Usage:
    python sar_complete.py
        # Same defaults as ``sar_coordinator.py`` plus --zstart 200 --zend 400
        # and the Plotly interactive slice viewer as the default output.
    python sar_complete.py -W 280 -H 40 --y-step 1 --snake
    python sar_complete.py --viz matplotlib         # fall back to the matplotlib window
    python sar_complete.py --viz gif --gif-fps 15   # render a GIF instead of HTML

All coordinator flags (``-W``, ``-H``, ``-s``, ``--snake``, ``--num-frames``,
``--frame-periodicity``, ``--y-step``, ``-o`` …) keep their original meaning.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import sar_coordinator as sc
import socket

# sar_reconstruct is imported lazily in run_reconstruction so the capture
# half of this script stays usable on machines without numpy/scipy/matplotlib.


def build_parser():
    """Coordinator parser + reconstruction-only flags."""
    parser = argparse.ArgumentParser(
        description="Run a SAR scan and reconstruct a visualization in one step."
    )
    sc.build_parser(parser)

    recon = parser.add_argument_group("reconstruction")
    recon.add_argument("--zstart", "--z_start", dest="zstart", type=str, default="200",
                       help="Z sweep start (e.g. '200', '200mm', '0.2m'; default 200)")
    recon.add_argument("--zend", "--z_end", dest="zend", type=str, default="400",
                       help="Z sweep end (e.g. '400', '400mm', '0.4m'; default 400)")
    recon.add_argument("--zstep", type=str, default=None,
                       help="Z sweep step (default: sar_reconstruct default of 3 mm)")
    recon.add_argument("--zindex", type=str, default=None,
                       help="Reconstruct a single Z slice instead of sweeping")
    recon.add_argument("--samples", type=int, default=512,
                       help="ADC samples per chirp (default 512)")
    recon.add_argument("--algo", type=str, default="mf", choices=["mf", "fista", "bpa"],
                       help="Reconstruction algorithm (default 'mf')")
    recon.add_argument("--fista_iters", type=int, default=20)
    recon.add_argument("--fista_lambda", type=float, default=0.05)
    recon.add_argument("--window", type=str, default="none",
                       choices=["none", "hanning", "hamming", "kaiser"],
                       help="Range FFT window (default 'none')")
    recon.add_argument("--kaiser-beta", type=float, default=6.0)
    recon.add_argument("--db", action="store_true", help="Display heatmaps in dB scale")
    recon.add_argument("--db-range", type=float, default=40.0,
                       help="Dynamic range in dB when --db is used (default 40)")
    recon.add_argument("--xyonly", action="store_true",
                       help="Skip X-Z and Y-Z heatmaps, only produce X-Y")
    recon.add_argument("--filename-pattern", type=str, default="row_{y}_Raw_0.bin",
                       help="Filename pattern for row data (default 'row_{y}_Raw_0.bin')")
    recon.add_argument("--3d_scatter", dest="scatter3d", action="store_true",
                       help="Also generate an interactive 3D scatter plot")
    recon.add_argument("--3d_scatter_intensity", dest="scatter3d_intensity",
                       type=float, default=95.0)
    recon.add_argument("--gif-fps", type=int, default=10)

    recon.add_argument(
        "--viz",
        choices=["plotly", "matplotlib", "gif"],
        default="plotly",
        help="Primary visualization output (default: plotly). "
             "'plotly' writes sar_interactive_slices.html into the scan folder, "
             "'matplotlib' opens the interactive slider window, "
             "'gif' renders a Z-slice animation to sar_slices.gif.",
    )

    pi = parser.add_argument_group("pi automation (SSH)")
    pi.add_argument("--pi-host", default="10.244.13.117",
                    help="Raspberry Pi hostname or IP (default 10.244.13.117)")
    pi.add_argument("--pi-user", default="peekir",
                    help="SSH user on the Pi (default 'peekir')")
    pi.add_argument("--pi-project-dir", default="PeekIR_Pi",
                    help="Project path on the Pi, relative to $HOME (default 'PeekIR_Pi')")
    pi.add_argument("--pi-venv", default="venv",
                    help="Virtualenv directory inside the project (default 'venv')")
    pi.add_argument("--coordinator-host", default=None,
                    help="IP the Pi should dial back to. Auto-detected if omitted.")
    pi.add_argument("--skip-pi-ssh", action="store_true",
                    help="Skip all Pi SSH automation (homing + tcp client launch).")
    pi.add_argument("--skip-pi-origin", action="store_true",
                    help="Skip the pre-scan 'move.py origin' homing step.")
    return parser


def build_reconstruct_argv(args, scan_folder):
    """Translate combined args into a ``sar_reconstruct.py``-style argv."""
    num_rows = sc.num_rows_from_height_step(args.height, args.y_step)
    num_frames = sc.compute_num_frames(args)

    argv = [
        "sar_reconstruct.py",
        "--folder", str(scan_folder),
        "--frames", str(num_frames),
        "--rows", str(num_rows),
        "--samples", str(args.samples),
        "--speed", str(args.speed),
        "--periodicity", str(args.frame_periodicity),
        "--y-step", str(args.y_step),
        "--scan-width", str(args.width),
        "--scan-height", str(args.height),
        "--filename-pattern", args.filename_pattern,
        "--zstart", str(args.zstart),
        "--zend", str(args.zend),
        "--algo", args.algo,
        "--fista_iters", str(args.fista_iters),
        "--fista_lambda", str(args.fista_lambda),
        "--window", args.window,
        "--kaiser-beta", str(args.kaiser_beta),
        "--db-range", str(args.db_range),
        "--3d_scatter_intensity", str(args.scatter3d_intensity),
    ]

    if args.zstep is not None:
        argv += ["--zstep", str(args.zstep)]
    if args.zindex is not None:
        argv += ["--zindex", str(args.zindex)]
    if args.snake:
        argv.append("--snake")
    if args.db:
        argv.append("--db")
    if args.xyonly:
        argv.append("--xyonly")
    if args.scatter3d:
        argv.append("--3d_scatter")

    if args.viz == "plotly":
        argv.append("--plotly")
    elif args.viz == "gif":
        argv += ["--gif", "sar_slices.gif", "--gif-fps", str(args.gif_fps)]
    # "matplotlib" -> pass nothing; sar_reconstruct opens its slider window

    return argv


def _detect_local_ip():
    """Best-effort local IP (the one the Pi should dial back to)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _build_ssh_cmd(args, remote_cmd):
    """Wrap a remote shell command in `bash -lc` and an ssh invocation.

    Using a login shell (-lc) ensures `source venv/bin/activate` resolves
    the same way it does in an interactive session.
    """
    target = f"{args.pi_user}@{args.pi_host}"
    return [
        "ssh",
        "-T",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=15",
        target,
        f'bash -lc "{remote_cmd}"',
    ]


def _remote_cd_and_activate(args):
    """Shell snippet that cds into the project and activates the venv."""
    return (f"cd $HOME/{args.pi_project_dir} && "
            f"source {args.pi_venv}/bin/activate")


def pi_home_gantry(args):
    """Blocking: SSH to the Pi and run `python move.py origin` to home the gantry."""
    remote = f"{_remote_cd_and_activate(args)} && python move.py origin"
    print(f"\n[Pi SSH] Homing gantry via {args.pi_user}@{args.pi_host} ...")
    result = subprocess.run(_build_ssh_cmd(args, remote), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Pi 'move.py origin' exited with code {result.returncode}. "
            f"Aborting scan."
        )
    print("[Pi SSH] Gantry homed.\n")


def pi_launch_tcp_client(args, coordinator_host, port):
    """Non-blocking: SSH to the Pi and run the gantry TCP client.

    Returns a Popen handle. The client exits by itself when the coordinator
    sends SHUTDOWN, which closes the SSH session and reaps this process.
    """
    remote = (f"{_remote_cd_and_activate(args)} && "
              f"python move.py tcp host={coordinator_host} port={port}")
    print(f"[Pi SSH] Launching gantry TCP client "
          f"(coordinator {coordinator_host}:{port}) ...")
    return subprocess.Popen(_build_ssh_cmd(args, remote))


def run_capture(args):
    """Run the TCP-coordinated capture. Returns the created scan folder path."""
    out_path = sc.next_scan_output_dir(Path(args.output_dir))
    args.output_dir = str(out_path)

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    except Exception:
        print(f"Note: could not create '{args.output_dir}' locally.")
        print("      Ensure it exists on the machine running mmWave Studio.")

    # 1. Pre-scan: home the gantry on the Pi (blocking over SSH).
    if not args.skip_pi_ssh and not args.skip_pi_origin:
        try:
            pi_home_gantry(args)
        except Exception as e:
            print(f"Homing failed: {e}")
            return None

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(2)

    coordinator_host = args.coordinator_host or _detect_local_ip()

    print(f"\nSAR Coordinator listening on port {args.port}")
    print("Start the clients:")
    print(f"  Radar  — run sar_scan_tcp.lua in mmWave Studio")
    if args.skip_pi_ssh:
        print(f"  Gantry — run: python3 move.py tcp "
              f"host={coordinator_host} port={args.port}")
    else:
        print(f"  Gantry — auto-launching via SSH to "
              f"{args.pi_user}@{args.pi_host}")
    print()

    # 2. Launch the gantry TCP client on the Pi in the background.
    gantry_ssh_proc = None
    if not args.skip_pi_ssh:
        gantry_ssh_proc = pi_launch_tcp_client(
            args, coordinator_host, args.port
        )

    gantry = None
    radar = None
    scan_ok = False
    try:
        gantry, radar = sc.accept_clients(server)
        print()
        sc.run_scan(gantry, radar, args)
        scan_ok = True
    except KeyboardInterrupt:
        print("\n\nScan interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for client in (gantry, radar):
            if client:
                try:
                    client.send("SHUTDOWN")
                except Exception:
                    pass
                client.close()
        server.close()

        if gantry_ssh_proc is not None:
            try:
                gantry_ssh_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[Pi SSH] Gantry client still running; terminating...")
                gantry_ssh_proc.terminate()
                try:
                    gantry_ssh_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    gantry_ssh_proc.kill()

        print("Coordinator shut down.")

    return out_path if scan_ok else None


def run_reconstruction(args, scan_folder):
    """Invoke ``sar_reconstruct.main`` with outputs landing in ``scan_folder``."""
    argv = build_reconstruct_argv(args, scan_folder)

    print("\n" + "=" * 60)
    print("Reconstruction")
    print(f"  Folder: {scan_folder}")
    print(f"  argv:   {' '.join(argv[1:])}")
    print("=" * 60 + "\n")

    import sar_reconstruct as sr

    # sar_reconstruct.py writes PNG/HTML/GIF artifacts with relative paths,
    # so run it with the scan folder as CWD to keep everything co-located.
    saved_cwd = os.getcwd()
    saved_argv = sys.argv
    try:
        os.chdir(str(scan_folder))
        sys.argv = argv
        sr.main()
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)


def main():
    parser = build_parser()
    args = parser.parse_args()

    scan_folder = run_capture(args)
    if scan_folder is None:
        print("Scan did not complete; skipping reconstruction.")
        return

    run_reconstruction(args, scan_folder)


if __name__ == "__main__":
    main()
