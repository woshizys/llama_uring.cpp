#!/usr/bin/env bash
set -euo pipefail

interval=1
device=""
csv=""
pid=""
quiet=0

usage() {
    cat <<'EOF'
Usage:
  io_bw_monitor.sh [options] --pid PID
  io_bw_monitor.sh [options] -- COMMAND [ARGS...]

Options:
  -i SEC        sample interval in seconds, default: 1
  -d DEVICE     block device name for /proc/diskstats, e.g. nvme1n1p1
  -o FILE       write CSV samples to FILE
  -q            quiet monitor output; implied when -o is set
  --print       print monitor samples to terminal even when -o is set
  -h            show this help

Columns:
  proc_read_MB_s / proc_write_MB_s:
      Physical storage I/O attributed to the process from /proc/PID/io
      read_bytes/write_bytes. If mmap hits page cache, these may be near zero.

  proc_rchar_MB_s / proc_wchar_MB_s:
      Logical bytes passed through read/write-like syscalls from /proc/PID/io.
      mmap page faults are not well represented by rchar.

  dev_read_MB_s / dev_write_MB_s:
      Whole-device throughput from /proc/diskstats sectors, only shown with -d.

Examples:
  ./tools/io_bw_monitor.sh -d nvme1n1p1 --pid 12345

  ./tools/io_bw_monitor.sh -i 0.5 -d nvme1n1p1 -o uring.csv -- \
    ./build-expert-cache/bin/llama-cli \
      -m /data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.gguf \
      -p "hello" -n 64 --cpu-moe --expert-cache-capacity 1000
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i)
            interval="${2:?missing value for -i}"
            shift 2
            ;;
        -d)
            device="${2:?missing value for -d}"
            device="${device#/dev/}"
            shift 2
            ;;
        -o)
            csv="${2:?missing value for -o}"
            quiet=1
            shift 2
            ;;
        -q|--quiet)
            quiet=1
            shift
            ;;
        --print)
            quiet=0
            shift
            ;;
        --pid)
            pid="${2:?missing value for --pid}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cmd_pid=""
if [[ -z "$pid" ]]; then
    if [[ $# -eq 0 ]]; then
        usage >&2
        exit 2
    fi
    "$@" &
    cmd_pid=$!
    pid=$cmd_pid
fi

proc_io_available=1
deadline=$((SECONDS + 5))
while [[ ! -r "/proc/$pid/io" ]]; do
    if ! kill -0 "$pid" 2>/dev/null; then
        [[ -n "$cmd_pid" ]] && wait "$cmd_pid" || true
        echo "process $pid exited before /proc/$pid/io became readable" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        proc_io_available=0
        if [[ -n "$device" ]]; then
            if [[ "$quiet" -eq 0 ]]; then
                echo "warning: cannot read /proc/$pid/io; process columns will be zero, device bandwidth will still be sampled" >&2
            fi
            break
        fi
        echo "cannot read /proc/$pid/io and no device was specified; try running with -d DEVICE" >&2
        exit 1
    fi
    sleep 0.05
done

read_proc_io() {
    local p=$1
    if [[ ! -r "/proc/$p/io" ]]; then
        printf "0 0 0 0\n"
        return
    fi
    awk '
        $1 == "rchar:"       { rchar = $2 }
        $1 == "wchar:"       { wchar = $2 }
        $1 == "read_bytes:"  { read_bytes = $2 }
        $1 == "write_bytes:" { write_bytes = $2 }
        END {
            printf "%s %s %s %s\n", read_bytes + 0, write_bytes + 0, rchar + 0, wchar + 0
        }
    ' "/proc/$p/io"
}

read_dev_io() {
    local dev=$1
    if [[ -z "$dev" ]]; then
        printf "0 0\n"
        return
    fi
    awk -v dev="$dev" '
        $3 == dev {
            # Linux diskstats sectors are 512-byte sectors.
            printf "%s %s\n", $6 * 512, $10 * 512
            found = 1
        }
        END {
            if (!found) {
                exit 1
            }
        }
    ' /proc/diskstats
}

if [[ -n "$device" ]] && ! read_dev_io "$device" >/dev/null; then
    echo "cannot find device '$device' in /proc/diskstats" >&2
    echo "hint: use lsblk -o NAME,MOUNTPOINTS,SIZE,TYPE" >&2
    [[ -n "$cmd_pid" ]] && kill "$cmd_pid" 2>/dev/null || true
    exit 1
fi

if [[ -n "$csv" ]]; then
    printf "timestamp,pid,proc_read_MB_s,proc_write_MB_s,proc_rchar_MB_s,proc_wchar_MB_s" > "$csv"
    if [[ -n "$device" ]]; then
        printf ",device,dev_read_MB_s,dev_write_MB_s" >> "$csv"
    fi
    printf "\n" >> "$csv"
fi

if [[ "$quiet" -eq 0 ]]; then
    printf "%-20s %-8s %12s %13s %13s %13s" "time" "pid" "p_read_MB/s" "p_write_MB/s" "p_rchar_MB/s" "p_wchar_MB/s"
    if [[ -n "$device" ]]; then
        printf " %12s %13s" "d_read_MB/s" "d_write_MB/s"
    fi
    printf "\n"
fi

read -r prev_pr prev_pw prev_rchar prev_wchar < <(read_proc_io "$pid")
read -r prev_dr prev_dw < <(read_dev_io "$device")
prev_t=$(date +%s.%N)

while kill -0 "$pid" 2>/dev/null; do
    sleep "$interval"
    if [[ "$proc_io_available" -eq 1 && ! -r "/proc/$pid/io" ]]; then
        break
    fi

    read -r cur_pr cur_pw cur_rchar cur_wchar < <(read_proc_io "$pid")
    read -r cur_dr cur_dw < <(read_dev_io "$device")
    cur_t=$(date +%s.%N)
    ts=$(date '+%Y-%m-%d %H:%M:%S')

    values=$(awk -v t0="$prev_t" -v t1="$cur_t" \
        -v pr0="$prev_pr" -v pr1="$cur_pr" \
        -v pw0="$prev_pw" -v pw1="$cur_pw" \
        -v rc0="$prev_rchar" -v rc1="$cur_rchar" \
        -v wc0="$prev_wchar" -v wc1="$cur_wchar" \
        -v dr0="$prev_dr" -v dr1="$cur_dr" \
        -v dw0="$prev_dw" -v dw1="$cur_dw" '
        BEGIN {
            dt = t1 - t0
            if (dt <= 0) {
                dt = 1
            }
            mib = 1024 * 1024
            printf "%.2f %.2f %.2f %.2f %.2f %.2f",
                (pr1 - pr0) / dt / mib,
                (pw1 - pw0) / dt / mib,
                (rc1 - rc0) / dt / mib,
                (wc1 - wc0) / dt / mib,
                (dr1 - dr0) / dt / mib,
                (dw1 - dw0) / dt / mib
        }
    ')

    read -r proc_read proc_write proc_rchar proc_wchar dev_read dev_write <<< "$values"

    if [[ "$quiet" -eq 0 ]]; then
        printf "%-20s %-8s %12s %13s %13s %13s" "$ts" "$pid" "$proc_read" "$proc_write" "$proc_rchar" "$proc_wchar"
        if [[ -n "$device" ]]; then
            printf " %12s %13s" "$dev_read" "$dev_write"
        fi
        printf "\n"
    fi

    if [[ -n "$csv" ]]; then
        printf "%s,%s,%s,%s,%s,%s" "$ts" "$pid" "$proc_read" "$proc_write" "$proc_rchar" "$proc_wchar" >> "$csv"
        if [[ -n "$device" ]]; then
            printf ",%s,%s,%s" "$device" "$dev_read" "$dev_write" >> "$csv"
        fi
        printf "\n" >> "$csv"
    fi

    prev_pr=$cur_pr
    prev_pw=$cur_pw
    prev_rchar=$cur_rchar
    prev_wchar=$cur_wchar
    prev_dr=$cur_dr
    prev_dw=$cur_dw
    prev_t=$cur_t
done

if [[ -n "$cmd_pid" ]]; then
    wait "$cmd_pid" || exit $?
fi
