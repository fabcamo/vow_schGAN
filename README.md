# VOW_SCHGAN

This repository contains code and experiments related to **SchemaGAN**, a generative model for geotechnical soil schematization developed within the **VOW project**.

---

## Overview

The repository provides a full workflow to prepare CPT (Cone Penetration Test) data for use with *SchemaGAN*.  
The main scripts handle coordinate extraction, data interpretation and compression, and spatial preparation of input matrices for GAN training.

---

## Setup

Create and activate a virtual environment (Python 3.10):

```bash
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Script Descriptions

### 1. `extract_coords.py`
Generates a CSV file containing the **coordinates** of all CPTs found in the specified project folder.

### 2. `extract_data.py`
Reads each `.gef` file, performs **CPT interpretation**, and compresses the resulting *Ic* (soil behaviour index) profiles from all CPTs into **32-pixel depth representations**.

### 3. `schGAN_input_file.py`
Reads the compressed *Ic* data, performs **geospatial sectioning** of the CPTs, and generates the **input matrices** for *SchemaGAN*.  
Each matrix represents a subsurface section where:
- `0` indicates areas without data, and  
- *Ic* values are placed at the corresponding CPT positions.

---

## Notes

- Ensure that all CPT files are stored in the designated data folder before running the scripts.  
- The generated input matrices can be directly used to train or test *SchemaGAN*.  
- File paths and parameters can be configured within each script as needed.

---

## Repository Structure

```
VOW_SCHGAN/
│
├── src/
│   ├── extract_coords.py
│   ├── extract_data.py
│   ├── schGAN_input_file.py
│   └── ...
│
├── data/
│   ├── raw/          # Raw .gef files
│   ├── processed/    # Interpreted and compressed CPT data
│   └── inputs/       # SchemaGAN-ready matrices
│
├── requirements.txt
└── README.md
```

---
