# Daphnia Detection and Analysis: General

Our program is an automated pipeline for scientific analysis of animals of the genus *Daphnia*. Detailed information about development and functionality in PAPERLINK.
We provide conda environments with package versions we used, but functionality is dependend on the cuda version you use.

![image](https://github.com/Fipsii/DaphniaDetector/blob/main/Zeichnung4.png?raw=true)


## Installation

Recommended installation with [PDM](https://pdm-project.org/latest/).


Make a python environment of your own and tell pdm to use it with `pdm use /path/to/your/venv` or let pdm handle the venv for you and simply:

```
pdm sync --group dev
```
