# PDCat experiment environment

- Collected at (UTC): `2026-08-11T14:35:48.428591+00:00`
- Device: NVIDIA Jetson Orin NX Engineering Reference Developer Kit
- Architecture/kernel: `aarch64 / 5.15.148-tegra`
- L4T: `# R36 (release), REVISION: 4.4, GCID: 41062509, BOARD: generic, EABI: aarch64, DATE: Mon Jun 16 16:07:13 UTC 2025`
- Memory total/available: 15655.6 MiB / 13663.2 MiB

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
| `/workspace/llama_uring.cpp` | `feat/pdcat` | `413e0c3cd35273cc69d4f5cbd0bbd0918d0f0d21` | yes |
| `/workspace/InterfaceIO` | `feat/pdcat` | `8229dc1b4a6a26b70f3f8a4ee7f83e0e68b6179a` | yes |
| `/workspace/EK-Edge` | `feat/pdcat` | `77ee8265ad39b1c89ca3b9c0a8f52e9f6766f796` | yes |

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
