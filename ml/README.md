# Machine learning:
This directory implements the full machine learning and quantization pipeline used to generate FPGA-ready integer parameters.

The design follows a Post-Training Quantization (PTQ) workflow.

The quantization design is inspired by:
> Jacob et al., 2017 - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"\
> [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)


## Folder structure:
```
ml
├─ Readme.md                    : The current file
├─ requirements.txt             : Python dependencies
├─ checkpoints
│  └─ run0                      : Auto-incremented per training run (not tracked by git)
│     ├─ best.pth               : Best validation checkpoint
│     └─ last.pth               : Latest epoch checkpoint
├─ runs
│  └─ run0                      : Mirrors checkpoints/runN (tracked by git)
│     ├─ run_meta.json          : Training config and args
│     └─ final_report.json      : Test metrics from best checkpoint
├─ outputs
│  └─ run0                      : Export artifacts for runN
│     ├─ act_scales_sy.json     : Per-layer activation scales (tracked)
│     ├─ fpgaqparms.json        : Layer metadata + quantization params (tracked)
│     ├─ fpgaqparms.npz         : Integer weight arrays (not tracked)
│     └─ fpgaqparms.bin         : FPGA-ready binary (not tracked)
├─ data
│  └─ MNIST
│     └─ mnist images for training
├─ scripts
│  └─ download_mnist.py
└─ src
   ├─ data
   │  └─ mnist64.py
   ├─ export
   │  ├─ export_weights.py
   │  ├─ find_scales.py
   │  ├─ quantize_weights.py
   │  └─ test_quantized_model.py
   ├─ models
   │  └─ alexnet64gray.py
   └─ train
      └─ train.py
```

## Pipeline Runner
Script: [run_pipeline.py](scripts/run_pipeline.py)

Runs the full pipeline end-to-end from a single config file:
```bash
python -m ml.scripts.run_pipeline --config ml/config/default.yaml --device cuda
```
Individual steps can be skipped with `--skip-train`, `--skip-calibrate`, `--skip-quantize`, `--skip-test`, `--skip-export`.
Pin to a specific run with `--run-id N`.

Config files live in [ml/config/](config/):
- `default.yaml` — training and calibration hyperparameters
- `fpga.yaml` — hardware constants (SDRAM threshold, binary format version)

## Dataset
MNIST is resized from 28×28 to 64×64 grayscale.

Download using:
```bash
python -m ml.scripts.download_mnist
```
Source: [download_mnist.py](scripts/download_mnist.py)

## Data Pipeline
Implemented in: [mnist64.py](src/data/mnist64.py)

### Features
- Resize to 64×64
- Optional normalization
- Optional augmentation
- Deterministic train/val split
- Configurable DataLoader

Config class:
```bash
@dataclass
class MNIST64Config:
```

This ensures reproducibility across:
- Training
- Calibration
- Quantized evaluation

## Model Architecture
Defined in: [alexnet64gray.py](src/models/alexnet64gray.py)

### Architecture Summary
Input: 1×64×64
```bash
Conv → ReLU → Pool
Conv → ReLU → Pool
Conv → ReLU
Conv → ReLU
Conv → ReLU → Pool
Flatten
Linear → ReLU
Linear → ReLU
Linear (logits)
```
Key design choices:
- No BatchNorm (simplifies integer export)
- ReLU activations only
- Clean layer separation for FPGA mapping

## Training
Script: [train.py](src/train/train.py)

### Training Setup
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Validation split
- Best-model tracking
- Automatic checkpointing
- Final test evaluation

### Outputs
Written to `ml/checkpoints/runN/`:
- `best.pth`: Best validation checkpoint
- `last.pth`: Latest epoch checkpoint

Written to `ml/runs/runN/`:
- `run_meta.json`: Training configuration and CLI args
- `final_report.json`: Final test metrics from best checkpoint

## Activation Calibration (PTQ)
Script: [find_scales.py](src/export/find_scales.py)

This step computes per-layer activation scales.

We hook post-ReLU activations.

### Why post-ReLU?
Following Jacob et al. (2017)
- Activations are non-negative
- Can use uint8
- Zero-point = 0
- Simplifies hardware
- Matches integer-only inference equation: $r = S(q-Z)$
- In this design: $Z = 0$

### Scale Computation
For each hooked layer:
$s_y = \dfrac{P_{99.9}(y)}{255}$

Where:
- $P_{99.9}$ = robust percentile
- 255 = max uint8 value

This reduces outlier impact while maintaining dynamic range.

## Weight Quantization + FPGA Export
Script: [quantize_weights.py](src/export/quantize_weights.py) 

This step produces:
- int8 weights (symmetric, per output channel)
- int32 bias (in accumulator units)
- integer requant parameters (m, r)

### Weight Quantization
Per-output-channel symmetric:

$$
s_y[c] = \dfrac{max|w[c]|}{127}
$$
$$
q_w[c] = round(\dfrac{w[c]}{s_w[c]})
$$

Weights are stored as int8

### Bias Quantization
$$
q_b[c] = \dfrac{b[c]}{s_x s_w[c]}
$$

Biases are stored as int32.

### Requantization
We compute:

$$
M_c = \dfrac{s_x s_w[c]}{s_y}
$$

Then approximate:
$$
M_c \approx \dfrac{m_c}{2^{r_c}}
$$

Enabling pure integer inference:

$$
acc = \sum(q_x \times q_w) + q_b
$$

$$
q_y = (acc \times m) >> r
$$

No float operations required on FPGA.

## Quantized Inference Verification
Script: [test_quantized_model.py](src/export/test_quantized_model.py)

This runs: 
- Full integer accumulator
- Integer ReLU
- Integer requant
- Final float reconstruction for logits comparison

It compares:
- Float accuracy
- Quantized PTQ accuracy

This represents the theoretical FPGA upper bound.

## Files Produced

| File | Location | Tracked | Description |
|------|----------|---------|-------------|
| `best.pth` | `checkpoints/runN/` | No | Best float checkpoint |
| `last.pth` | `checkpoints/runN/` | No | Latest epoch checkpoint |
| `run_meta.json` | `runs/runN/` | Yes | Training config |
| `final_report.json` | `runs/runN/` | Yes | Test metrics |
| `act_scales_sy.json` | `outputs/runN/` | Yes | Per-layer activation scales |
| `fpgaqparms.json` | `outputs/runN/` | Yes | Layer metadata + quant params |
| `fpgaqparms.npz` | `outputs/runN/` | No | Integer weight arrays |
| `fpgaqparms.bin` | `outputs/runN/` | No | FPGA-ready binary |

## FPGA Compatibility Assumptions
- Post-ReLU activations -> uint8 (post-ReLU)
- Weights -> int8
- Bias -> int32
- Accumulator -> int32 or int64
- Requant -> multiply + shift
- Zero-point = 0

This matches the hardware:
- Line buffer
- Sliding window MAC
- Per-channel requant block

## Full Pipeline Summary
Train float model ->
Calibrate activations (P99.9) ->
Quantize weights + compute -> 
Export integers arrays ->
Verify quantized accuracy ->
Deploy on FPGA

## Why This Design Is Hardware Friendly
- No per-tensor zero-points
- No asymmetric quantization
- No float division
- No runtime scaling memory
- Per-channel scaling improves accuracy
- Requant uses only shift + multiply
