# Isaac Sim 6.2.0: requirements and status on this machine

Status: **NOT INSTALLED, NOT RUN.** This document is a requirements audit, not a
claim that Isaac Sim works here. The adapter in
`src/sim2real/sim/isaac_backend.py` is a guarded stub that refuses to construct.

Checked on 2026-10-31 against the official Isaac Sim 6.2.0 documentation.

## Sources (fetched 2026-10-31)

- Requirements:
  <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html>
- Cross-product compatibility:
  <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/compatibility.html>
- Python environment installation:
  <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_python.html>
- ROS 2 installation:
  <https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/install_ros.html>

## Requirements versus this machine

| Element | Isaac Sim 6.2.0 minimum | This machine | Result |
| --- | --- | --- | --- |
| OS | Ubuntu 22.04/24.04 or Windows 11 | Ubuntu 24.04 under WSL2 | OS version meets; WSL2 support not established for the simulator (see below) |
| CPU cores | 4 | 2 vCPU visible in WSL (host: 8 cores / 16 logical) | **below** in the current WSL configuration |
| RAM | 32 GB | 5.8 GiB visible in WSL (host: 15.3 GB) | **below** |
| Storage | 50 GB SSD | about 930 GB free | meets |
| GPU | GeForce RTX 4080 | RTX 4050 Laptop | **below** |
| VRAM | 16 GB | 6 GB | **below** |
| Driver | Linux 595.58.03 | 551.76 | **below the tested version** |
| Python | 3.12 | 3.12.3 | meets |
| ROS 2 | Jazzy recommended on Ubuntu 24.04 | Jazzy installed | meets |

The docs also state that GPUs without RT cores are unsupported and that GPUs
with less than 16 GB VRAM may be insufficient for complex scenes. The RTX 4050
Laptop has RT cores but only 6 GB VRAM.

**Isaac Sim tier: UNAVAILABLE on this machine** based on published
requirements, not on a local run. No install has been authorized or attempted.

## Does Isaac Sim run inside WSL2?

**NOT ESTABLISHED.** The 6.2.0 ROS 2 installation page documents WSL2 only for
**external ROS nodes**. Its WSL workflow says: "The simulator runs on Windows.
This WSL workflow is deprecated; custom interfaces are not supported." The
cross-product page does not claim WSL2 simulator support either.

Therefore this repository records: Isaac Sim itself is **not** documented as
running inside WSL2 for 6.2.0. Only ROS 2 nodes are shown running there.

## Possible topologies

1. **Different machine or cloud + ROS bridge.** Run Isaac Sim on a machine that
   meets the requirements (Windows 11 or Ubuntu 24.04 with an RTX 4080-class GPU
   and 16 GB VRAM), and bridge it over DDS to ROS 2 nodes in this WSL2 instance.
   The docs' Windows + WSL networking section describes explicit Fast DDS peers,
   NAT networking, and firewall rules for this topology. This is not tested here.
2. **MuJoCo only (the plan for this repository).** Use the headless MuJoCo
   backend for data generation, filtering, mapping, and VLA training. Isaac Sim
   remains an optional, guarded backend for hardware that can run it.

## What would change this verdict

- A machine meeting the published minimums, or an explicitly supported cloud
  image, and a successful local or remote smoke test whose output is recorded.
- New official documentation that establishes simulator support inside WSL2.
  Until then, do not claim WSL2 Isaac Sim support.
