#!/usr/bin/env python3
"""
SAR Scan Coordinator — TCP server that synchronizes radar data capture
with gantry movement for SAR imaging.

Recommended setup:
  - Run this script on the Windows machine (same as mmWave Studio)
  - Lua radar client connects via localhost (near-zero latency)
  - Pi gantry client connects over the local network (~1-5ms latency)

Usage:
    python sar_coordinator.py -W 280 -H 40 -r 40 -o "C:\\path\\to\\data"
    python sar_coordinator.py -W 280 -H 40 -x 0 -y 0 -r 40 -s 18 --return-speed 36 -o "C:\\data"

Protocol (newline-delimited text, two persistent TCP connections):

  RADAR client (mmWave Studio Lua):
    Server sends: CONFIGURE <frames> <period> | ARM <filepath> | CAPTURE | PING | SHUTDOWN
    Client sends: RADAR | CONFIGURED | ARMED | CAPTURE_DONE | PONG | ERROR <msg>

  GANTRY client (Raspberry Pi move.py):
    Server sends: MOVE <args> | STOP | PING | SHUTDOWN
    Client sends: GANTRY | MOTOR_STARTED | MOVE_COMPLETE | PONG | ERROR <msg>
"""

import socket
import threading
import argparse
import os
import sys
import time


class ClientConnection:
    """Wrapper around a TCP socket for line-based protocol communication."""

    def __init__(self, sock, addr, name):
        self.sock = sock
        self.addr = addr
        self.name = name
        self._buf = ""

    def send(self, msg):
        self.sock.sendall((msg + "\n").encode())
        print(f"  [{self.name}] -> {msg}")

    def recv(self, timeout=None):
        prev = self.sock.gettimeout()
        if timeout is not None:
            self.sock.settimeout(timeout)
        try:
            while "\n" not in self._buf:
                data = self.sock.recv(4096)
                if not data:
                    raise ConnectionError(f"{self.name} disconnected")
                self._buf += data.decode()
        finally:
            self.sock.settimeout(prev)
        line, self._buf = self._buf.split("\n", 1)
        line = line.strip()
        print(f"  [{self.name}] <- {line}")
        return line

    def expect(self, expected, timeout=60):
        """Read one message and raise if it doesn't match *expected*."""
        msg = self.recv(timeout)
        if msg != expected:
            raise RuntimeError(f"{self.name}: expected '{expected}', got '{msg}'")
        return msg

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Connection acceptance
# ---------------------------------------------------------------------------
def accept_clients(server_sock, timeout=120):
    """Wait for both RADAR and GANTRY clients to connect and identify themselves."""
    server_sock.settimeout(timeout)
    clients = {}

    while len(clients) < 2:
        remaining = {"RADAR", "GANTRY"} - set(clients.keys())
        print(f"  Waiting for: {', '.join(sorted(remaining))}")

        conn, addr = server_sock.accept()
        print(f"  Connection from {addr}")

        conn.settimeout(10)
        buf = ""
        while "\n" not in buf:
            data = conn.recv(1024)
            if not data:
                conn.close()
                continue
            buf += data.decode()

        ident = buf.split("\n")[0].strip()
        remaining_buf = buf.split("\n", 1)[1] if "\n" in buf else ""

        if ident in ("RADAR", "GANTRY") and ident not in clients:
            client = ClientConnection(conn, addr, ident)
            client._buf = remaining_buf
            clients[ident] = client
            print(f"  {ident} registered from {addr}")
        else:
            label = "duplicate" if ident in clients else "unknown"
            print(f"  Rejected {label} client '{ident}' from {addr}")
            conn.close()

    return clients["GANTRY"], clients["RADAR"]


# ---------------------------------------------------------------------------
# Scan execution
# ---------------------------------------------------------------------------
def run_scan(gantry, radar, args):
    """Execute the full SAR scan sequence."""
    width = args.width
    height = args.height
    x_start = args.x_start
    y_start = args.y_start
    num_rows = args.rows
    speed = args.speed
    return_speed = args.return_speed
    frame_periodicity = args.frame_periodicity
    stabilization_ms = args.stabilization
    output_dir = args.output_dir

    y_step = args.y_step if args.y_step is not None else (
        height / (num_rows - 1) if num_rows > 1 else 0
    )

    if args.num_frames is not None:
        num_frames = args.num_frames
    else:
        scan_time_ms = (width / speed) * 1000
        num_frames = int(scan_time_ms / frame_periodicity) + 10

    # Detect Windows vs POSIX path separator from user-supplied output dir
    sep = "\\" if "\\" in output_dir else "/"
    output_dir = output_dir.rstrip("\\/")

    print(f"\n{'=' * 60}")
    print("SAR Scan Configuration")
    print(f"  Area:         {width} x {height} mm")
    print(f"  Start:        ({x_start}, {y_start})")
    print(f"  Rows:         {num_rows}   Y-step: {y_step:.2f} mm")
    print(f"  Speed:        {speed} mm/s   Return: {return_speed} mm/s")
    print(f"  Frames/row:   {num_frames} @ {frame_periodicity} ms "
          f"= {num_frames * frame_periodicity:.0f} ms capture")
    print(f"  Stabilize:    {stabilization_ms} ms")
    print(f"  Output:       {output_dir}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # 1. Configure radar and warm up DCA1000
    # ------------------------------------------------------------------
    print("[1/4] Configuring radar...")
    radar.send(f"CONFIGURE {num_frames} {int(frame_periodicity)}")
    radar.expect("CONFIGURED")
    print("      Radar configured.")

    # Run a dummy ARM+CAPTURE cycle so the DCA1000 is fully warmed up.
    # The first real StartRecord after initialization often fails to write.
    print("      Warming up DCA1000 (dummy capture)...")
    dummy_path = f"{output_dir}{sep}_warmup_dummy.bin"
    radar.send(f"ARM {dummy_path}")
    resp = radar.recv(timeout=30)
    if resp == "ARMED":
        radar.send("CAPTURE")
        radar.recv(timeout=60)  # wait for CAPTURE_DONE
    # Clean up the dummy file
    try:
        import glob as _glob
        for f in _glob.glob(dummy_path.replace(".bin", "*")):
            os.remove(f)
    except Exception:
        pass
    # DCA1000 needs time between StopRecord and the next StartRecord
    # to finalize the previous file. Without this, row_1 gets silently dropped.
    time.sleep(3)
    print("      DCA1000 ready.\n")

    # ------------------------------------------------------------------
    # 2. Move gantry to start position (if not at origin)
    # ------------------------------------------------------------------
    if x_start > 0 or y_start > 0:
        print(f"[2/4] Moving to start ({x_start}, {y_start})...")
        parts = []
        if x_start > 0:
            parts.append(f"right={x_start}mm")
        if y_start > 0:
            parts.append(f"up={y_start}mm")
        gantry.send(f"MOVE {' '.join(parts)} speed={speed}mms")
        gantry.expect("MOTOR_STARTED", timeout=30)
        gantry.expect("MOVE_COMPLETE", timeout=120)
        print("      At start position.\n")
    else:
        print("[2/4] Start at current position.\n")

    # ------------------------------------------------------------------
    # 3. Verify DCA1000 with a real test row (optional sanity check)
    # ------------------------------------------------------------------
    print("[3/4] Ready to scan.\n")

    # ------------------------------------------------------------------
    # 4. Row-by-row scan
    # ------------------------------------------------------------------
    print(f"[4/4] Scanning {num_rows} rows...\n")
    scan_start = time.time()

    for row in range(1, num_rows + 1):
        filepath = f"{output_dir}{sep}row_{row}.bin"
        row_start = time.time()

        print(f"--- Row {row}/{num_rows} ---")

        # ARM DCA1000
        radar.send(f"ARM {filepath}")
        resp = radar.recv(timeout=30)
        if resp != "ARMED":
            print(f"  ARM failed: {resp}")
            return

        # Start gantry scan movement
        gantry.send(f"MOVE right={width}mm speed={speed}mms")
        gantry.expect("MOTOR_STARTED", timeout=30)

        # Wait for motor to reach constant speed
        if stabilization_ms > 0:
            time.sleep(stabilization_ms / 1000.0)

        # Trigger radar capture (blocks on Lua side until frame completes)
        radar.send("CAPTURE")

        # Wait for both gantry and radar to finish (concurrent)
        results = {}
        errors = []

        def _wait(client, key):
            try:
                results[key] = client.recv(timeout=180)
            except Exception as e:
                errors.append(f"{client.name}: {e}")

        t_g = threading.Thread(target=_wait, args=(gantry, "gantry"))
        t_r = threading.Thread(target=_wait, args=(radar, "radar"))
        t_g.start()
        t_r.start()
        t_g.join(timeout=180)
        t_r.join(timeout=180)

        if errors:
            print(f"  ERRORS: {errors}")
            return

        if results.get("gantry") != "MOVE_COMPLETE":
            print(f"  Gantry unexpected: {results.get('gantry')}")
            return
        if results.get("radar") != "CAPTURE_DONE":
            print(f"  Radar unexpected: {results.get('radar')}")
            return

        # Return gantry to start X
        gantry.send(f"MOVE left={width}mm speed={return_speed}mms")
        gantry.expect("MOTOR_STARTED", timeout=30)
        gantry.expect("MOVE_COMPLETE", timeout=120)

        # Step up for next row
        if row < num_rows and y_step > 0:
            gantry.send(f"MOVE up={y_step:.2f}mm speed={speed}mms")
            gantry.expect("MOTOR_STARTED", timeout=30)
            gantry.expect("MOVE_COMPLETE", timeout=120)

        elapsed = time.time() - row_start
        print(f"  Row {row} done ({elapsed:.1f}s)\n")

    total = time.time() - scan_start
    print(f"{'=' * 60}")
    print(f"Scan complete — {num_rows} rows in {total:.1f}s")
    print(f"Data saved to {output_dir}")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SAR Scan Coordinator — synchronize radar capture with gantry movement"
    )
    parser.add_argument("-W", "--width", type=float, required=True,
                        help="Horizontal scan distance per row (mm)")
    parser.add_argument("-H", "--height", type=float, required=True,
                        help="Total vertical scan distance (mm)")
    parser.add_argument("-x", "--x-start", type=float, default=0,
                        help="X start position (mm, default 0)")
    parser.add_argument("-y", "--y-start", type=float, default=0,
                        help="Y start position (mm, default 0)")
    parser.add_argument("-r", "--rows", type=int, required=True,
                        help="Number of scan rows")
    parser.add_argument("-s", "--speed", type=float, default=18,
                        help="Scan speed (mm/s, default 18)")
    parser.add_argument("--return-speed", type=float, default=36,
                        help="Gantry return speed (mm/s, default 36)")
    parser.add_argument("--y-step", type=float, default=None,
                        help="Y step between rows (mm, overrides height-based calc)")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Frames per capture (default: auto from scan time)")
    parser.add_argument("--frame-periodicity", type=float, default=18,
                        help="Radar frame periodicity (ms, default 18)")
    parser.add_argument("--stabilization", type=float, default=500,
                        help="Motor stabilization delay before capture (ms, default 500)")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Output directory for .bin files (Windows path)")
    parser.add_argument("-p", "--port", type=int, default=5555,
                        help="TCP server port (default 5555)")

    args = parser.parse_args()

    # Try to create output directory (works when coordinator runs on Windows)
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    except Exception:
        print(f"Note: could not create '{args.output_dir}' locally.")
        print("      Ensure it exists on the machine running mmWave Studio.")

    # Start TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(2)

    print(f"\nSAR Coordinator listening on port {args.port}")
    print("Start the clients:")
    print(f"  Radar  — run sar_scan_tcp.lua in mmWave Studio")
    print(f"  Gantry — run: python3 move.py tcp host=<this-ip> port={args.port}")
    print()

    gantry = None
    radar = None
    try:
        gantry, radar = accept_clients(server)
        print()
        run_scan(gantry, radar, args)
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
        print("Coordinator shut down.")


if __name__ == "__main__":
    main()
