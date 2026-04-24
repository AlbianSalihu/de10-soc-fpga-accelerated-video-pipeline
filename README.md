# DE1-SoC FPGA-Accelerated Video Pipeline

## Overview
This repository is a hardware-oriented edge inference project targeting the Intel DE1-SoC (ARM + FPGA). The end goal is to run a compact CNN using a fully integer-only inference datapath on FPGA, with a verification workflow that ensures the hardware implementation matches a trusted Python reference.

What’s already implemented today:
- A minimal but complete ML + quantization + export pipeline in PyTorch (binary parameter export to .bin format pending)
- Generation of FPGA-ready integer parameters (int8 weights, int32 biases, fixed-point requantization parameters)
- A quantized inference verifier in Python that models integer accumulation + requantization

What’s planned next (not implemented yet):
- FPGA inference blocks (streaming conv, MAC, pooling, requantization)
- End-to-end verification harness (Python-driven stimuli, RTL simulation, output comparison)

The system is built around hardware-aware quantization, inspired by:
> Jacob et al., Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (2017)\
> [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)

## Repository Layout
- ml/ - training, calibration, PTQ quantization, and parameter export
- hardware/ - FPGA datapath implementation (planned / WIP)
- software/ - HPS-side glue / drivers / runtime (planned / WIP)
- verification/ - Python-based verification flow (planned / WIP)
- docs/ - design notes, diagrams, and documentation (planned / WIP)

## Machine Learning Pipeline

The ML pipeline is located in:
```bash
ml/
```
It performs:
- Float model training
- Post-training activation calibration
- Per-channel weight quantization
- Integer-only parameter export
- Quantized inference verification

Detailed documentation: [README.md](ml/README.md)

### Model

The implemented network:
AlexNet64Gray:
- Input: 1×64×64 grayscale
- Convolution + ReLU + MaxPool blocks
- 3 fully-connected layers
- Output: 10 classes

Defined in: [ml/src/models/alexnet64gray.py](ml/src/models/alexnet64gray.py)

The architecture was selected to:
- Be large enough to stress hardware
- Remain tractable for FPGA implementation
- Avoid BatchNorm (simplifies quantization)

---

## End-to-End Flow

### Option A — Run the full pipeline with one command
```bash
python -m ml.scripts.run_pipeline --config ml/config/default.yaml --device cuda
```
Edit `ml/config/default.yaml` to change hyperparameters. Individual steps can be skipped:
```bash
python -m ml.scripts.run_pipeline --run-id 0 --skip-train --skip-calibrate
```

---

### Option B — Run each step manually

### 1. Download dataset
```bash
python -m ml.scripts.download_mnist
```

### 2. Train float model
```bash
python -m ml.src.train.train \
    --epochs 10 \
    --batch-size 128 \
    --device cuda
```
Outputs: `ml/checkpoints/runN/{best,last}.pth`, `ml/runs/runN/{run_meta,final_report}.json`

### 3. Calibrate activation scales (Post-Training Quantization)
```bash
python -m ml.src.export.find_scales \
    --hook relu \
    --percentile 0.999
```
Auto-detects latest run. Outputs: `ml/outputs/runN/act_scales_sy.json`

### 4. Quantize weights + compute FPGA parameters
```bash
python -m ml.src.export.quantize_weights \
    --s0 0.02
```
Auto-detects latest run. Outputs: `ml/outputs/runN/fpgaqparms.{npz,json}`

### 5. Verify integer inference accuracy
```bash
python -m ml.src.export.test_quantized_model \
    --s0 0.02
```
Auto-detects latest run. Compares float vs quantized accuracy.

### 6. Export FPGA binary
```bash
python -m ml.src.export.export_weights
```
Auto-detects latest run. Outputs: `ml/outputs/runN/fpgaqparms.bin`

## What This Enables
This project is not about training MNIST.

It is about enabling:
- FPGA-accelerated neural inference
- Edge AI deployment
- Deterministic low-latency video processing
- Integer-only compute pipelines
- Custom hardware acceleration blocks

The same architecture applies to:
- CNN-based video filters
- Object detection accelerators
- Embedded computer vision
- Industrial image processing
- Real-time sensor pipelines


