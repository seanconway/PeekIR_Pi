#!/usr/bin/env python3
"""
ESP32 BLE monitor for metal detection + battery status.

Requires:
    pip install bleak

Usage:
    python metal_detector.py
    python metal_detector.py --name ESP32-BLE-MON
    python metal_detector.py --address 0C:4E:A0:4D:B1:2E
"""

import argparse
import asyncio
import contextlib
from datetime import datetime

from bleak import BleakClient, BleakScanner

SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHARACTERISTIC_UUID_TX = "12345678-1234-5678-1234-56789abcdef2"

DEFAULT_DEVICE_NAME = "ESP32-BLE-MON"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_message(message: str) -> str:
    message = message.strip()

    if message.startswith("PIN: HIGH"):
        return f"[{ts()}] Metal detect: HIGH | {message}"
    if message.startswith("PIN: LOW"):
        return f"[{ts()}] Metal detect: LOW  | {message}"
    if message.startswith("BAT:"):
        return f"[{ts()}] Battery: {message[4:].strip()}"
    if message.startswith("BOOT:"):
        return f"[{ts()}] Device: {message}"
    return f"[{ts()}] Notify: {message}"


def notification_handler(_sender, data: bytearray) -> None:
    try:
        message = data.decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[{ts()}] Decode error: {exc}", flush=True)
        return

    print(parse_message(message), flush=True)


async def find_device(name: str | None, address: str | None, timeout: float = 10.0):
    if address:
        device = await BleakScanner.find_device_by_address(address, timeout=timeout)
        if device is None:
            raise RuntimeError(f"Could not find device with address {address}")
        return device

    print(f"[{ts()}] Scanning for BLE device...", flush=True)
    devices = await BleakScanner.discover(timeout=timeout)

    # Exact name first
    if name:
        for d in devices:
            if d.name == name:
                return d

    # Then service UUID if present in advertisement metadata
    for d in devices:
        uuids = [u.lower() for u in (d.metadata.get("uuids") or [])]
        if SERVICE_UUID.lower() in uuids:
            return d

    # Then partial name
    if name:
        for d in devices:
            if d.name and name.lower() in d.name.lower():
                return d

    found = "\n".join(
        f"  - {d.name or '<unknown>'} [{d.address}]"
        for d in devices
    ) or "  <no devices found>"

    raise RuntimeError(f"Could not find matching ESP32 device.\nFound:\n{found}")


async def wait_for_stop(stop_event: asyncio.Event):
    try:
        while not stop_event.is_set():
            await asyncio.sleep(0.2)
    except asyncio.CancelledError:
        pass


async def monitor(device_name: str | None, address: str | None, reconnect_delay: float):
    stop_event = asyncio.Event()

    while not stop_event.is_set():
        try:
            device = await find_device(device_name, address)
            print(
                f"[{ts()}] Connecting to {device.name or '<unknown>'} [{device.address}]...",
                flush=True,
            )

            disconnected_event = asyncio.Event()

            def handle_disconnect(_client):
                print(f"[{ts()}] Disconnected", flush=True)
                disconnected_event.set()

            async with BleakClient(device, disconnected_callback=handle_disconnect) as client:
                if not client.is_connected:
                    raise RuntimeError("Connection failed")

                print(f"[{ts()}] Connected", flush=True)

                # Subscribe directly. Bleak resolves the characteristic by UUID.
                await client.start_notify(CHARACTERISTIC_UUID_TX, notification_handler)
                print(f"[{ts()}] Listening for notifications...", flush=True)

                stop_task = asyncio.create_task(wait_for_stop(stop_event))
                disconnect_task = asyncio.create_task(disconnected_event.wait())

                done, pending = await asyncio.wait(
                    [stop_task, disconnect_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                with contextlib.suppress(Exception):
                    await client.stop_notify(CHARACTERISTIC_UUID_TX)

        except KeyboardInterrupt:
            stop_event.set()
            break
        except Exception as exc:
            if stop_event.is_set():
                break
            print(f"[{ts()}] Error: {exc}", flush=True)
            print(f"[{ts()}] Retrying in {reconnect_delay:.1f} seconds...", flush=True)
            try:
                await asyncio.sleep(reconnect_delay)
            except KeyboardInterrupt:
                stop_event.set()
                break


def main():
    parser = argparse.ArgumentParser(
        description="Monitor ESP32 BLE metal detector and battery status"
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_DEVICE_NAME,
        help=f"BLE device name to search for (default: {DEFAULT_DEVICE_NAME})",
    )
    parser.add_argument(
        "--address",
        default=None,
        help="BLE address to connect to directly",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Seconds to wait before reconnecting",
    )
    args = parser.parse_args()

    try:
        asyncio.run(monitor(args.name, args.address, args.reconnect_delay))
    except KeyboardInterrupt:
        print(f"\n[{ts()}] Stopped", flush=True)


if __name__ == "__main__":
    main()