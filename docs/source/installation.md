# Installation and Setup

This tutorial walks you through installing ONTraC (Ordered Niche Trajectory Construction) and confuguring your environment for spatial omics data analysis.

## System Requirements

ONTraC supports following systems:

- Operating Systems: Linux, macOS, and Windows
- Python Versions: 3.10, 3.11, and 3.12
- Optional: CUDA-capable GPU for faster processing (recommended for large datasets)

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

ONTraC can utilize GPU acceleration via CUDA for faster processing. If a CUDA-capable GPU is not available, ONTraC will run on the CPU.

The following PyTorch CUDA versions are supported:

- cu118 (CUDA 11.8)
- cu124 (CUDA 12.4)
- cu126 (CUDA 12.6)

Please refer to the [official CUDA website](https://docs.nvidia.com/cuda/) for CUDA installation instructions.

```{note}
Please use `nvidia-smi` to check CUDA installation status.
```

## Installation

ONTraC can be installed using pip. Choose the installation method that best suits your workflow.

### Step 0: Clear Cache (Optional but Recommended)

```sh
pip cache purge
conda clean -a -y
```

### Step 1: Create and Activate a Conda Environment (Optional but Recommended)

```bash
conda create -y -n ONTraC python=3.11
conda activate ONTraC
```

### Step 2: Install ONTraC

#### Option 1: Install Stable Version using pip

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

#### Option 2: Install the Development Version from GitHub

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

### Step 3: Set Up Jupyter (Optional but Recommended)

If you plan to use ONTraC with Jupyter notebooks, add the ONTraC environment as a new kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name ONTraC --display-name "Python 3.11 (ONTraC)"
```

### Step 4: Test

#### CUDA Availability Test (Optional)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

```{note}
Please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for PyTorch installation instructions.
```

#### ONTraC Installation Test

```{code-cell}
:class: input-cell

python -c "import ONTraC; ONTraC.utils.write_version_info()"
```

```{code-block}
:class: output-cell

##################################################################################

         ▄▄█▀▀██   ▀█▄   ▀█▀ █▀▀██▀▀█                   ▄▄█▀▀▀▄█
        ▄█▀    ██   █▀█   █     ██    ▄▄▄ ▄▄   ▄▄▄▄   ▄█▀     ▀
        ██      ██  █ ▀█▄ █     ██     ██▀ ▀▀ ▀▀ ▄██  ██
        ▀█▄     ██  █   ███     ██     ██     ▄█▀ ██  ▀█▄      ▄
         ▀▀█▄▄▄█▀  ▄█▄   ▀█    ▄██▄   ▄██▄    ▀█▄▄▀█▀  ▀▀█▄▄▄▄▀

                        version: 1.2.0

##################################################################################
```
