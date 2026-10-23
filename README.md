# Malaria Decision Layer

**Patient-level screening is not determined by detector weights alone: a secondary reanalysis of the downstream inference configuration in automated malaria microscopy.**

This repository contains the complete codebase, analysis engine (`mag`), and experiment suite for the manuscript above. It provides the computational reproducibility layer for the study, ensuring that every number, table, and figure in the manuscript is generated directly from the source data via a strictly audited pipeline.

---

## 1. The Core Premise

Automated malaria microscopy is reported as a patient-level diagnostic, but what is trained and released is a field-level object detector. Converting its detections into one call per patient requires a **downstream inference configuration**: a confidence floor, a suppression rule, an aggregation rule, a threshold, and a calibration map. 

This repository treats that inference configuration as the object of study. It holds the released detector's cached output fixed and varies only the downstream decision layer. 

### Key Findings Demonstrated by This Code
1. **The configuration changes the diagnosis:** On the same cached detections, changing only the confidence floor and suppression rule moved a single aggregation rule from sensitivity 99.1% / specificity 74.1% to a specificity of **0.0%** (calling every patient infected).
2. **The published result was not reconstructed:** The released artefacts did not rebuild the source study's published patient-level total in either class ontology. The documented rule scored 105/170 against the published 116/170, with the divergence located in species assignment rather than screening.
3. **Checkpoints are not comparable out-of-the-box:** The released checkpoints (RT-DETR, YOLOv5x, YOLOv8x) differ in output convention. RT-DETR emits a fixed query budget per field regardless of content, meaning a raw box count measures the decoder, not the parasites.

---

## 2. Repository Structure

*   **`mag3.0.0/`** — The core analysis engine. It implements the 14 aggregation rules, statistical gates (McNemar, Tango, GEE), and the strict "no fabrication" architecture.
*   **`malaria_experiment_suite/`** — The orchestration pipeline. It handles GPU inference, cache management, checkpoint verification, cap-sensitivity testing, and LaTeX manuscript compilation.

> **Note on Data & Outputs:** 
> Due to GitHub file size limits, large datasets, the detection cache (`data/`), and generated results (`results/`, `verify/`, etc.) are **excluded** via `.gitignore`. They are generated deterministically when you run the pipeline.

---

## 3. The Archive & Structural Constraints

The analysis runs against the public **Plasmodium-v1 archive** (Guemas et al.), which carries 170 test patients from 6 French centres over 4,508 fields. 

Reviewers should note two structural constraints of this archive that are **built into the estimands** of the code:
*   **Specificity is not identifiable:** All 54 uninfected patients come from a single centre (Montpellier). Specificity is perfectly collinear with centre, meaning no estimator can separate them.
*   **Sensitivity is conditioned on enriched sampling:** 96.7% of the fields from an infected patient contain an annotated parasite. The pipeline evaluates sensitivity under this enriched field sampling, which is far more favourable than routine screening.

---

## 4. Prerequisites & Installation

### Environment
*   **Python**: 3.9.13 (or newer)
*   **CUDA**: PyTorch `2.7.1+cu118` with CUDA available (tested on RTX 4050, 6.4 GB)
*   **Dependencies**: `yolov5` (7.0.14), `torch`, `pandas`, `statsmodels`, `matplotlib`, etc.
*   **LaTeX**: `pdflatex` and `bibtex` must be on your system PATH.

### Setup
```bash
git clone https://github.com/briceket23/malaria-decision-layer.git
cd malaria-decision-layer/mag3.0.0

# Install the core engine (CPU only)
pip install -e .

# If you are running the detection stage (requires GPU)
pip install -e ".[gpu]"
