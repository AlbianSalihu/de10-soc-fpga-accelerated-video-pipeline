# DE10-SoC FPGA-Accelerated Video Pipeline

FPGA-accelerated frame processing demo on **Intel/Terasic DE10-SoC (Cyclone V SoC)**.  
A **Linux application on the HPS** generates frames in DDR, the **FPGA fabric applies a filter using DMA** (DDR → FPGA → DDR), and the system logs **latency/throughput metrics** and **FPGA resource utilization** for benchmarking and portfolio/demo purposes.

---

## Highlights
- **Zero-copy style pipeline** using shared DDR between HPS and FPGA
- **FPGA DMA-based processing** (frame read → filter → frame write)
- **Instrumentation-ready**: frame IDs, timestamps, counters (latency / throughput / drops)
- **Reproducible results**: scripts to run tests and export CSV/plots *(WIP)*

---

## Architecture (High Level)

**HPS (Linux)**
- Generates or loads frames (synthetic patterns, image sequence, or video decode)
- Writes frames into **RAW buffer(s)** in DDR
- Starts/controls the FPGA pipeline via memory-mapped registers
- Collects metrics and optionally saves output frames / logs

**FPGA (Fabric)**
- DMA reads from **RAW buffer(s)** in DDR
- Applies an image filter (configurable)
- DMA writes to **FILTERED buffer(s)** in DDR
- Updates counters / “frame done” flags / timestamps for profiling

**DDR Memory (Shared)**
- RAW buffers: produced by HPS
- FILTERED buffers: produced by FPGA
- Optional double-buffering (ping-pong) to avoid tearing

> Target demo output: latency/throughput graphs + Quartus resource reports (LUT/FF/BRAM/DSP).

---

## Repository Layout (Planned)
