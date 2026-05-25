# 🚀 PyTorch Performance Modelling

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Comprehensive Benchmarking Suite for Deep Learning Model Performance Analysis**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Results](#results) • [Contributing](#contributing)

</div>

---

## 📋 Overview

This project provides a **systematic framework for benchmarking deep learning models** across different hardware accelerators and batch sizes. It measures critical performance metrics including throughput, memory consumption, FLOPs, and arithmetic intensity—essential for understanding model deployment characteristics.

Perfect for researchers, engineers, and ML practitioners who need to:
- ✅ Profile models on multiple hardware backends
- ✅ Compare performance across different batch sizes
- ✅ Estimate computational complexity and memory footprint
- ✅ Analyze hardware utilization and arithmetic intensity
- ✅ Generate reproducible, JSON-formatted benchmark reports

---

## ✨ Features

### 🎯 Core Capabilities
- **Multi-Device Support**: Benchmark on CPU, NVIDIA CUDA GPUs, and Apple Metal Performance Shaders (MPS)
- **Flexible Model Selection**: Support for 70+ PyTorch Vision models (ResNet, MobileNet, EfficientNet, Vision Transformers, etc.)
- **Variable Batch Sizes**: Test multiple batch configurations to understand scaling behavior
- **Detailed Performance Metrics**:
  - Throughput (images/second)
  - FLOPs estimation (forward, backward, total)
  - Memory analysis (parameters, activations, DRAM)
  - Arithmetic intensity (FLOPs per byte)
  - Latency per batch

### 🔧 Technical Excellence
- **Comprehensive Profiling**: Track computational complexity, memory bandwidth, and hardware efficiency
- **Statistical Rigor**: Averages over thousands of batches for reliable benchmarks
- **Thread Management**: Configurable CPU threading (OMP, MKL) for fair comparisons
- **DRAM Estimation**: Approximate memory bandwidth requirements during training
- **JSON Export**: Structured benchmark results for downstream analysis

---

## 📊 Supported Models & Hardware

### Models
- **ResNet Family**: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
- **MobileNet Family**: MobileNetV2, MobileNetV3
- **VGG Family**: VGG11, VGG13, VGG16, VGG19
- And 60+ more from `torchvision.models`

### Hardware Devices
| Device | Notation | Use Case |
|--------|----------|----------|
| **CPU** | `cpu` | Baseline, edge deployment analysis |
| **NVIDIA GPU** | `cuda` / `t4` | High-performance training, inference |
| **Apple GPU** | `mps` | Mac deployment, edge inference |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.6+
- Tiny ImageNet dataset (included setup script)

### Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Performance-Modelling

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Download Tiny ImageNet
bash extract_ILSVRC.sh

# 5. Start benchmarking!
python main.py --arch resnet50 --device cuda --batch-size 128
```

### Dependencies

```
torch>=2.6
torchvision
```

Optional (for detailed FLOPs estimation):
```bash
pip install ptflops
```

---

## 📖 Usage

### Basic Benchmarking

```bash
# Benchmark ResNet50 on CUDA GPU with batch size 128
python main.py --arch resnet50 --device cuda --batch-size 128

# Benchmark MobileNetV2 on CPU with batch size 4
python main.py --arch mobilenet_v2 --device cpu --batch-size 4

# Benchmark on Apple Silicon (MPS)
python main.py --arch resnet18 --device mps --batch-size 8
```

### Advanced Options

```bash
# Control iterations
python main.py --arch resnet50 --device cuda --batch-size 64 --max-iters 50000

# Pin memory optimization (CUDA only)
python main.py --arch resnet50 --device cuda --batch-size 128 --pin-memory

# Distributed training simulation
python main.py --arch resnet50 --device cuda --world-size 4 --rank 0
```

### Output

Each benchmark run generates a JSON file with complete metrics:

```json
{
  "arch": "resnet50",
  "device": "cuda",
  "batch_size": 128,
  "avg_batch_time_s": 0.0456,
  "throughput_imgs_per_s": 2807.0,
  "measured_batches": 25000,
  "flops_forward_per_image": 8264873936.0,
  "flops_forward_per_batch": 1057903863808.0,
  "flops_train_per_batch_est": 3173711591424.0,
  "dram_bytes_per_batch_est": 78899509760.0,
  "params_bytes": 102228128,
  "activation_bytes_one_sample": 128161696,
  "GFLOP_per_s_est": 69650.12,
  "arithmetic_intensity_Flop_per_byte": 40.17
}
```

---

## 📈 Results & Benchmarks

### Key Metrics Explained

| Metric | Unit | Interpretation |
|--------|------|-----------------|
| **throughput** | images/sec | Higher = faster |
| **avg_batch_time_s** | seconds | Lower = faster |
| **GFLOP/s_est** | Giga FLOPs/sec | GPU computational throughput |
| **arithmetic_intensity** | FLOPs/byte | Compute-to-memory ratio; higher = better GPU utilization |

### Sample Benchmark Results

**ResNet50 Performance Across Devices:**

```
Device    | Batch | Throughput (img/s) | GFLOP/s | Arithmetic Intensity
----------|-------|-------------------|---------|-------------------
CPU       | 4     | 6.06              | 150.31  | 160.67
T4 GPU    | 128   | 2,807.0           | 69,650  | 40.17
MPS       | 8     | 185.5             | 3,200   | 85.42
```

See `benchmark_*.json` files for detailed runs across different configurations.

---

## 📁 Project Structure

```
.
├── README.md                          # This file
├── main.py                            # Main benchmarking script
├── requirements.txt                   # Python dependencies
├── extract_ILSVRC.sh                 # Dataset preparation script
│
├── benchmark_results/                 # Generated benchmark outputs
│   ├── benchmark_resnet50_cpu_b4.json
│   ├── benchmark_resnet50_mps_b4.json
│   ├── benchmark_resnet50_t4_b128.json
│   └── ...
│
└── tiny-imagenet-200/                # Dataset directory
    ├── train/                         # Training images (100k)
    ├── val/                          # Validation images (10k)
    ├── test/                         # Test images (10k)
    ├── wnids.txt                     # WordNet IDs mapping
    └── words.txt                     # Class labels
```

---

## 🔬 Technical Details

### Benchmarking Methodology

1. **Warmup Phase**: Initial iterations to stabilize GPU/system state
2. **Measurement Phase**: Configurable number of iterations (default: 25,000)
3. **Averaging**: All metrics averaged across measurement phase
4. **Statistical Validity**: Large sample size ensures reliable results

### Memory Estimation

- **Activation bytes**: Tracked via forward hooks
- **DRAM bandwidth**: Estimated from model parameters and activation sizes
- **Parameter memory**: Direct PyTorch calculation

### Computational Complexity

- **Forward FLOPs**: Estimated per image and batch
- **Training FLOPs**: ~3x forward (forward + 2× backward) per batch
- **Optional ptflops integration**: For detailed layer-wise FLOPs

### Device-Specific Optimizations

- **CUDA**: Pin memory support, cuDNN auto-tuning
- **CPU**: OMP/MKL threading configuration
- **MPS**: Apple Silicon metal optimization

---

## 🚀 Performance Tips

### For CPU Benchmarking
```python
# Set thread counts for reproducibility
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### For GPU Benchmarking
```python
# Enable cuDNN auto-tuning (if using CUDA)
torch.backends.cudnn.benchmark = True
```

### For Fair Comparisons
- Use same batch size across devices
- Run multiple measurements (>1,000 batches)
- Disable background processes
- Ensure thermal stability

---

## 📚 Citation & References

If you use this benchmarking suite in research, please cite:

```bibtex
@misc{pytorch-performance-modelling,
  title={PyTorch Performance Modelling Suite},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/performance-modelling}
}
```

### Related Resources
- [PyTorch Official Benchmarks](https://github.com/pytorch/pytorch/tree/master/caffe2/utils/perf_tester)
- [MLPerf Benchmark Suite](https://mlperf.org/)
- [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-improvement`)
3. **Commit** your changes (`git commit -m 'Add amazing improvement'`)
4. **Push** to the branch (`git push origin feature/amazing-improvement`)
5. **Open** a Pull Request

### Contribution Ideas
- [ ] Add support for new models
- [ ] Implement additional performance metrics
- [ ] Create visualization scripts for results
- [ ] Add distributed training benchmarks
- [ ] Optimize benchmarking code
- [ ] Improve documentation

---

## ⚠️ Known Limitations

- **Memory estimation** for activations is approximate
- **FLOPs calculation** depends on model architecture (best estimates for CNNs)
- **MPS device** availability limited to Apple Silicon Macs
- **CUDA support** requires compatible NVIDIA GPU and CUDA toolkit

---

## 📝 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 💬 Questions & Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [your-email@example.com](mailto:sb10449@nyu.edu)

---

<div align="center">

**Made with ❤️ for the ML Community**

⭐ If you found this useful, please consider starring the repository!

</div>
