# Machine learning:
This directory implements the full machine learning and quantization pipeline used to generate FPGA-ready integer parameters.

The design follows a Post-Training Quantization (PTQ) workflow.

The quantization design is inspired by:
> Jacob et al., 2017 - "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"\
> [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)


## Folder structure:
```
ml
├─ Readme.md : The current file
├─ requirements.txt : All requirements to run the machine learning parts
├─ checkpoints
│  ├─ act_scales_sy.json:
│  ├─ best.pth:
│  ├─ final_report.json:
│  ├─ fpgaqparms.json:
│  ├─ fpgaqparms.npz: 
│  └─ last.pth:
├─ config
├─ data
│  └─MNIST
│    └─ mnist images for training
├─ outputs
├─ runs
├─ scripts
│  └─ download_mnist.py: 
└─ src
   ├─ data:
   │  └─ mnist64.py
   ├─ export:
   │  ├─ export_weights.py:
   │  ├─ find_scales.py:
   │  ├─ quantize_weights.py: 
   │  └─ test_quantized_model.py: 
   ├─ models:
   │  └─ alexnet64gray.py
   └─ train:
      └─ train.py 
```

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

### Outputs:
- best.pth: Best validation model
- last.pth: Last epoch model
- run_meta.json: Training configuration
- final_report.json: Final test metrics

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
acc = \sum(q_x \times q_x) + q_b
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
- best.pth : Best float model
- act_scales_sy.json : Activation scales
- fpga_qparams.npz : All integer arrays
- fpga_qparams.json : Metadata + layer mapping

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
