# PDCat experiment environment

- Collected at (UTC): `2026-08-11T16:37:02.811269+00:00`
- Device: NVIDIA Jetson Orin NX Engineering Reference Developer Kit
- Architecture/kernel: `aarch64 / 5.15.148-tegra`
- L4T: `# R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64, DATE: Mon Jun 16 16:07:13 UTC 2025`
- Memory total/available: 15655.6 MiB / 12484.5 MiB

## Software

| Tool | Version |
| --- | --- |
| nvcc | nvcc: NVIDIA (R) Cuda compiler driver |
| gcc | gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 |
| gxx | g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 |
| cmake | cmake version 3.31.6 |
| rustc | rustc 1.92.0 (ded5c06cf 2025-12-08) |
| cargo | cargo 1.92.0 (344c4567c 2025-10-21) |

## Repositories

| Repository | Branch | Commit | Dirty |
| --- | --- | --- | --- |
| `/workspace/llama_uring.cpp` | `feat/pdcat` | `3005b9b92bd054043d5435ed015a0c9a3fa2ad93` | yes |
| `/workspace/InterfaceIO` | `feat/pdcat` | `f0e00c5cfdddba662f4a31f84d8c7bf13e49e8fe` | yes |
| `/workspace/EK-Edge` | `feat/pdcat` | `b50ef59f3c4f0c988af27df57788f5c68dd16d87` | yes |

## NVMe

| Controller | Model | Firmware | Link |
| --- | --- | --- | --- |
| nvme0 | aigo NVMe SSD DP35 256GB | SN13321 | 8.0 GT/s PCIe x4 |
| nvme1 | Acer SSD N5000M 512GB | X0430L | 16.0 GT/s PCIe x2 |

## Candidate models

- `/data/models/DeepSeek-V2-Lite-GGUF/DeepSeek-V2-Lite-Q8_0.gguf`: exists=True, size=15928.8 MiB, sha256=`7b131e7fdddd10eeca4d1716832de9aaa60ff2544a94bec13cb34946bec514b0`

## Outstanding controls

- Power/clock commands are unavailable in this container; record nvpmodel and jetson_clocks on the host before final runs.
- Record GGUF architecture, quantization, layers, experts, top-k, and expert byte layout in the run configuration.
