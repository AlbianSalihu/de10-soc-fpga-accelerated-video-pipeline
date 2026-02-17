# Machine learning:
This part describe how the machine learning part works.

## Description of the folder structure:
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