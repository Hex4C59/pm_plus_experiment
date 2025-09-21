# PM+ Experiment (Speech Emotion Recognition)

This repository is a lightweight, reproducible template for Speech Emotion
Recognition (SER) on the PM+ dataset. It standardizes project structure,
data handling, experiment tracking, and model development using PyTorch.

Key features
- Clean layout for data, configs, scripts, and source code.
- Reproducible runs and split manifests stored under data/splits and runs/.
- Clear extension points for datasets, models, losses, and metrics.

Quick start
1) Environment (Python 3.10)
   - Option A (uv)
     - uv venv
     - uv sync
   - Option B (venv + pip)
     - python -m venv .venv && source .venv/bin/activate
     - pip install -U pip && pip install -e .

2) Data layout
   - Place PM+ audio under data/raw/PM+/
   - Put annotations (e.g., label.xlsx) under data/annotations/
   - Put split definitions under data/splits/

3) Build train/test split
   - Use scripts in scripts/split/ to generate manifests and/or symlink trees.
   - Keep speaker-level integrity and stratify by gender if needed.

4) Train / Infer
   - python train.py -h
   - python infer.py -h

Repository layout
- configs/       Experiment/default config examples
- data/          Data root (raw, processed, annotations, splits)
- pretrained/    External weights and download helpers
- scripts/       Data preparation, split, and utility scripts
- src/           Library code (data, models, losses, metrics, utils)
- runs/          Outputs of training/evaluation runs

License
- MIT. See pyproject.toml for project metadata.