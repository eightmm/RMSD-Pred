<div align="center">

# RMSD-Pred

**Protein-Ligand Binding Pose RMSD Prediction using Graph Neural Networks**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.7+-green.svg)](https://pyg.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## Overview

RMSD-Pred scores the quality of a docked protein-ligand pose by predicting its
RMSD to the (unknown) native binding mode — without needing the crystal pose. It
builds three graphs per complex — protein pocket, ligand, and the protein-ligand
interface — and couples them at every layer.

**How it works**
1. **Pocket extraction** — protein atoms with a ligand atom within 10 Å.
2. **Graph construction** — protein graph (atom pairs within 4 Å, distance RBF
   edges; nodes encoded by residue + atom-type embeddings), ligand atom graph
   (covalent bonds), and a complex interface graph (atom pairs within 5 Å,
   distance RBF). Each node carries a 20-step random-walk positional encoding (LSPE).
3. **Message passing** — 3 parallel stacks (protein / ligand / complex) of 4
   `GatedGCNLSPE` layers (`emb=256`). Protein and ligand states are stitched into
   the complex graph and fed back each layer.
4. **Readout** — sum-pool complex nodes → MLP. Two heads share the architecture:
   a **regression** model (predicted RMSD in Å) and a **classification** model
   (probability that RMSD > 2 Å).

> Companion model: **[BA-Pred](https://github.com/eightmm/BA-Pred)** predicts
> binding affinity (pKd). Typical pipeline: dock → filter poses with RMSD-Pred →
> rank affinity with BA-Pred.

## Installation

```bash
git clone https://github.com/eightmm/RMSD-Pred.git
cd RMSD-Pred
uv sync   # installs PyTorch (CUDA 12.8) + PyTorch Geometric
```

> **GPU (incl. NVIDIA Blackwell / sm_120):** `uv sync` uses the bundled CUDA 12.8
> index. With pip: `pip install -e .` then
> `pip install torch --index-url https://download.pytorch.org/whl/cu128`

## Usage

### Command Line

```bash
rmsdpred -r example/prot.pdb -l example/ligs.sdf -o results.tsv --device cuda
```

### Python API

```python
from rmsdpred.inference import inference

# Uses packaged seed0 reg + cls weights by default
inference(
    protein_pdb="example/prot.pdb",
    ligand_file="example/ligs.sdf",
    output="results.tsv",
    batch_size=128,
    device="cuda",
)

# Or specify custom weights
inference(
    protein_pdb="example/prot.pdb",
    ligand_file="example/ligs.sdf",
    output="results.tsv",
    batch_size=128,
    reg_weight="/path/to/reg.pth",
    cls_weight="/path/to/cls.pth",
    device="cuda",
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-r, --protein_pdb` | Receptor protein PDB file | required |
| `-l, --ligand_file` | Ligand file (`.sdf` / `.mol2` / `.dlg` / `.pdbqt` / `.txt`) | required |
| `-o, --output` | Output TSV file | `result.tsv` |
| `--reg_weight` | Regression model weight file | packaged `random_reg_seed0.pth` |
| `--cls_weight` | Classification model weight file | packaged `random_cls_seed0.pth` |
| `--batch_size` | Batch size | `128` |
| `--ncpu` | CPU threads / DataLoader workers | `4` |
| `--device` | `cuda` or `cpu` | `cuda` |

## Input/Output Formats

### Input
- **Protein**: PDB format (`.pdb`)
- **Ligands**: one of
  - `.sdf` — SD file (multi-molecule / multi-pose supported)
  - `.mol2` — Tripos MOL2 (multi-molecule supported)
  - `.dlg` / `.pdbqt` — AutoDock poses (parsed via Meeko); `ADG_Score` is taken
    from the docking log when available
  - `.txt` — a list of any of the above file paths, one per line

### Output
Tab-separated file with columns:
- `Name` — pose identifier (`_Name` property if present, else `<file>_<index>`)
- `pRMSD` — predicted RMSD to native pose (Å)
- `Is_Above_2A` — probability that RMSD > 2 Å (0–1)
- `ADG_Score` — AutoDock score (`NaN` if unavailable)

Poses that fail to parse/build yield `NaN`.

## Weights

Packaged checkpoints live in `src/rmsdpred/weight/random/`: regression
(`random_reg_seed{0,1,2}.pth`) and classification (`random_cls_seed{0,1,2}.pth`).
Defaults are the seed0 pair. Average across seeds for ensemble predictions.

## Project Structure

```
RMSD-Pred/
├── src/
│   └── rmsdpred/          # Main package
│       ├── data/          # Graph construction, atom/residue features, loaders
│       ├── model/         # GatedGCNLSPE, PredictionRMSD (PyG)
│       ├── weight/        # Packaged reg + cls weights
│       └── inference.py   # Inference engine + CLI
├── example/               # Sample protein + ligand poses
├── tests/                 # Smoke tests (dataset build + forward)
├── pyproject.toml
└── README.md
```

## License

This project is licensed under the **Apache License 2.0**.

## Citation

```bibtex
@article{Sim_2026,
  title={BA-Pred and RMSD-Pred: Integrated Graph Neural Network Models for Accurate Protein-Ligand Binding Affinity and Binding Pose Prediction},
  author={Sim, Jaemin and Lee, Juyong},
  journal={Journal of Chemical Information and Modeling},
  year={2026},
  month={apr},
  doi={10.1021/acs.jcim.5c02591},
  url={https://doi.org/10.1021/acs.jcim.5c02591}
}
```
