# Copilot Instructions for VOW_SCHGAN

## Project Purpose
This repository implements a workflow for preparing Cone Penetration Test (CPT) data for use with SchemaGAN, a generative model for geotechnical soil schematization in the VOW project.

## Architecture & Data Flow
- **Raw CPT data** (`.gef` files) is placed in `data/raw/`.
- `src/extract_coords.py`: Extracts CPT coordinates to CSV.
- `src/extract_data.py`: Interprets CPTs, compresses *Ic* (soil behavior index) profiles to 32-pixel depth, outputs to `data/processed/`.
- `src/create_schGAN_input_file.py`: Reads compressed *Ic* data, sections CPTs spatially, and generates input matrices in `data/inputs/` for SchemaGAN. Matrices use `0` for no data, *Ic* values at CPT locations.

## Key Conventions
- All scripts are in `src/`. Data folders are at the project root.
- File paths and parameters are set within each script (no central config).
- Matrices for GAN training/testing are always generated in `data/inputs/`.
- *Ic* profile compression is always to 32 pixels (see `extract_data.py`).

## Developer Workflows
- **Setup:** Use Python 3.10, create a venv, install from `requirements.txt`.
- **Run scripts in order:**
  1. `extract_coords.py` → 2. `extract_data.py` → 3. `create_schGAN_input_file.py`
- **Data folder hygiene:** Place new `.gef` files in `data/raw/` before running scripts. Remove or archive old processed data if re-running.
- **No test suite or build system**: Manual script execution is the norm.

## Integration & Dependencies
- No external APIs or services; all processing is local.
- Scripts expect standard Python scientific stack (see `requirements.txt`).
- No custom CLI or entrypoint; run scripts directly.

## Example Usage
```bash
# Activate environment
.\.venv\Scripts\activate
# Extract coordinates
ython src/extract_coords.py
# Interpret and compress CPTs
python src/extract_data.py
# Generate SchemaGAN input matrices
python src/create_schGAN_input_file.py
```

## Reference Files
- `src/extract_coords.py`, `src/extract_data.py`, `src/create_schGAN_input_file.py`: Main workflow scripts
- `data/raw/`, `data/processed/`, `data/inputs/`: Data flow directories

---
For questions about workflow or data conventions, see `README.md` for details.
