# Daphnia Detection and Analysis: General

Our program is an automated pipeline for scientific analysis of animals of the genus *Daphnia*. Detailed information about development and functionality in PAPERLINK.
We provide conda environments with package versions we used, but functionality is dependend on the cuda version you use.

![image](https://github.com/Fipsii/DaphniaDetector/blob/main/Zeichnung4.png?raw=true)

# Installation and Usage

Recommended installation is via [PDM](https://pdm-project.org/latest/), but instructions for standard Pip are also provided below.


## Method 1: PDM (Recommended)

This project uses [PDM](https://pdm-project.org/) for dependency management to ensure exact reproducibility.

### 1. Install PDM
If you do not have PDM installed, install it via pip (or see the [official docs](https://pdm-project.org/latest/#installation)):

```bash
pip install --user pdm
```

### 2. Setup the Environment
Navigate to the project root directory.

```bash
/path/to/daphniadetectv2
```
You can either let PDM automatically manage the virtual environment or link it to an existing one.

Option A: Let PDM manage it (Automatic) This will create a virtual environment specifically for this project.

```bash
pdm sync
```

Option B: Use your own python/venv If you prefer to use a specific Python interpreter (e.g., from a specific global install or another venv):

```bash
# Tell PDM which python interpreter to use
pdm use /path/to/your/venv/bin/python

# Install dependencies into that environment
pdm sync
```

3. Run the Application
Use pdm run to execute scripts within the environment:

```bash
pdm run python DaphniaDetectv2/src/daphniadetectv2/DaphnidDetector.py
```

## Method 2: Standard Pip (venv)

1. Create Virtual Environment

```bash
python -m venv venv
```
2. Activate Environment

Linux/macOS:

```bash
source venv/bin/activate
```

Windows: 

```bash
.\venv\Scripts\activate
```
3. Install the Project

```bash
pip install .
```

4. Run the Application

```bash
python DaphniaDetectv2/src/daphniadetectv2/DaphnidDetector.py
```
