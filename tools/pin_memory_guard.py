#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import errno
import mmap
import os
import resource
import signal
import subprocess
import sys
import time


MiB = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reserve unswappable RAM with mlock(2), then optionally run a command. "
            "This is useful for reducing MemAvailable while testing memory budgets."
        )
    )
    parser.add_argument(
        "-m",
        "--mib",
        type=int,
        default=2048,
        help="amount of RAM to pin in MiB (default: 2048)",
    )
    parser.add_argument(
        "--no-touch",
        action="store_true",
        help="do not pre-fault every page before mlock; usually not recommended",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="print page-touch progress",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run after '--'; if omitted, hold memory until Ctrl+C",
    )
    args = parser.parse_args()
    if args.mib <= 0:
        parser.error("--mib must be positive")
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def format_limit(value: int) -> str:
    if value == resource.RLIM_INFINITY:
        return "unlimited"
    return f"{value // MiB} MiB"


def touch_pages(buf: mmap.mmap, size: int, progress: bool) -> None:
    page = mmap.PAGESIZE
    last_report = time.monotonic()
    for offset in range(0, size, page):
        buf[offset : offset + 1] = b"\0"
        if progress:
            now = time.monotonic()
            if now - last_report >= 1.0:
                print(f"pin_memory_guard: touched {offset // MiB}/{size // MiB} MiB", file=sys.stderr)
                last_report = now
    if progress:
        print(f"pin_memory_guard: touched {size // MiB}/{size // MiB} MiB", file=sys.stderr)


def mlock_buffer(buf: mmap.mmap, size: int) -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int

    address = ctypes.addressof((ctypes.c_char * 1).from_buffer(buf))
    ret = libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(size))
    if ret != 0:
        err = ctypes.get_errno()
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        hint = ""
        if err in (errno.ENOMEM, errno.EPERM):
            hint = (
                f"; RLIMIT_MEMLOCK soft={format_limit(soft)}, hard={format_limit(hard)}. "
                "Try running as root, `ulimit -l unlimited`, or granting CAP_IPC_LOCK."
            )
        raise OSError(err, f"mlock({size // MiB} MiB) failed: {os.strerror(err)}{hint}")


def munlock_buffer(buf: mmap.mmap, size: int) -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munlock.restype = ctypes.c_int
    address = ctypes.addressof((ctypes.c_char * 1).from_buffer(buf))
    libc.munlock(ctypes.c_void_p(address), ctypes.c_size_t(size))


def run_command(command: list[str]) -> int:
    proc = subprocess.Popen(command)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                return proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait()


def main() -> int:
    args = parse_args()
    size = args.mib * MiB

    print(f"pin_memory_guard: allocating {args.mib} MiB anonymous RAM", file=sys.stderr)
    buf = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)

    try:
        if not args.no_touch:
            print("pin_memory_guard: touching pages to commit physical memory", file=sys.stderr)
            touch_pages(buf, size, args.progress)

        print(f"pin_memory_guard: locking {args.mib} MiB with mlock", file=sys.stderr)
        mlock_buffer(buf, size)
        print("pin_memory_guard: memory is pinned; it will be released when this process exits", file=sys.stderr)

        if args.command:
            return run_command(args.command)

        print("pin_memory_guard: no command provided; press Ctrl+C to release", file=sys.stderr)
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 130
    finally:
        try:
            munlock_buffer(buf, size)
        finally:
            buf.close()


if __name__ == "__main__":
    raise SystemExit(main())
