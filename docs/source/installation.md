# Installation and Setup

This document guides you through the process of installing ONTraC (Ordered Niche Trajectory Construction) and setting up your environment for spatial omics data analysis. For information about key concepts and terminology used in ONTraC, see Key Concepts and Terminology.

## System Requirements

ONTraC is compatible with following systems:

- Operating Systems: Linux, macOS, and Windows
- Python Versions: 3.10, 3.11, and 3.12
- Optional: CUDA-capable GPU for accelerated processing (recommended for large datasets)

```{mermaid}
flowchart LR
  subgraph A["System Requirements"]
    B("Operating Systems:

    • Linux
    • macOS
    • Windows")

    C("Python Versions:

    • 3.10
    • 3.11
    • 3.12")

    D("Hardware Recommendations:

    • CUDA-capable GPU (optional but recommended)
    • Sufficient RAM for large datasets")

  end
```

## GPU Configuration

ONTraC can utilize GPU acceleration via CUDA for faster processing. If a CUDA-capable GPU is not available, ONTraC will run on CPU.

The following PyTorch CUDA versions are supported:

- cu118 (CUDA 11.8)
- cu124 (CUDA 12.4)
- cu126 (CUDA 12.6)

Please refer to the [official website](https://docs.nvidia.com/cuda/) for CUDA installation instructions.

## Installation Methods

ONTraC can be installed using pip. Choose the method that best fits your workflow.

### Step1: Create and Activat a Conda Environment (Optional but Recommended)

```bash
conda create -y -n ONTraC python=3.11
conda activate ONTraC
```

### Step2: Install ONTraC

#### Option1: Install Stable Version using Pip

For basic functionality:

```bash
pip install ONTraC
```

For visualization capabilities:

```bash
pip install "ONTraC[analysis]"
```

For test capabilities:

```bash
pip install "ONTraC[test]"
```

For develop capabilities:

```bash
pip install "ONTraC[dev]"
```

For all capabilities:

```bash
pip install "ONTraC[all]"
```

#### Option2: Install Developing Version from GitHub

For the latest developing version:

```bash
git clone git@github.com:gyuanlab/ONTraC.git .
cd ONTraC
pip install .
# Or with visualization capabilities:
pip install ".[analysis]"
# Or with test capabilities:
pip install ".[test]"
# Or with develop capabilities:
pip install ".[dev]"
# Or with all capabilities:
pip install ".[all]"
```

### Step3 Setting Up Jupyter (Optional but Recommended)

If you plan to use ONTraC with Jupyter notebooks, add the ONTraC environment as a new kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name ONTraC --display-name "Python 3.11 (ONTraC)"
```

## Dependencies and Component Structure

ONTraC has several core dependencies that are automatically installed during the installation process:

```{mermaid}
flowchart LR
subgraph A["ONTraC Dependencies Structure"]
  subgraph B1["Core Dependencies"]
    B11[Pytorch]
    B12["PyTroch Geometric (PYG)"]
    B13(Pandas)
    B14(pyyaml)
    B15(scipy)
  end

  subgraph B2["Analysis Dependencies (Optional)"]
    B21[matplotlib]
    B22[seaborn]
  end

  C1[ONTraC]
  C2["ONTraC analysis"]

  subgraph D["ONTraC Components"]
    subgraph E["Running Components"]
      E1("ONTraC_NN
      (Niche Network)")
      E2("ONTraC_GNN
      (Graph Neural Network)")
      E3("ONTraC_NT
      (Niche Trajectory)")
    end

    subgraph F["Analysis Components"]
      F1("ONTraC_analysis
      (Visualization)")
    end
  end
end
B11 --> C1
B12 --> C1
B13 --> C1
B14 --> C1
B15 --> C1
B12 --> C1
B21 --> C2
B22 --> C2
C1 --> E1
C1 --> E2
C1 --> E3
C1 --> F1
C2 --> F1
```
