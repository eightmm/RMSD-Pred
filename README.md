# RMSD-Pred

Protein-ligand binding pose RMSD prediction using Graph Neural Networks.

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
rmsdpred \
    -r example/prot.pdb \
    -l example/ligs.sdf \
    -o results.tsv \
    --device cuda
```

### Python API

```python
from rmsdpred.inference import inference

# Uses packaged random_seed0 weights by default
inference(
    protein_pdb="example/prot.pdb",
    ligand_file="example/ligs.sdf",
    output="results.tsv",
    batch_size=128,
    device="cuda"
)

# Or specify custom weights
inference(
    protein_pdb="example/prot.pdb",
    ligand_file="example/ligs.sdf",
    output="results.tsv",
    batch_size=128,
    reg_weight="/path/to/reg.pth",
    cls_weight="/path/to/cls.pth",
    device="cuda"
)
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-r, --protein_pdb` | Receptor protein PDB file | required |
| `-l, --ligand_file` | Ligand file (.sdf/.mol2/.dlg/.pdbqt/.txt) | required |
| `-o, --output` | Output TSV file | `result.tsv` |
| `--reg_weight` | Regression model weight file | packaged `random_reg_seed0.pth` |
| `--cls_weight` | Classification model weight file | packaged `random_cls_seed0.pth` |
| `--batch_size` | Batch size | `128` |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--ncpu` | Number of CPU workers | `4` |

### Output

Tab-separated file with columns:

- **Name**: Ligand pose identifier
- **pRMSD**: Predicted RMSD (Angstrom)
- **Is_Above_2A**: Confidence score (0-1, probability of RMSD > 2A)
- **ADG_Score**: AutoDock score (NaN if unavailable)

## Project Structure

```
RMSD-Pred/
├── src/rmsdpred/
│   ├── data/
│   │   ├── data.py
│   │   ├── ligand_atom_feature.py
│   │   ├── protein_atom_feature.py
│   │   └── utils.py
│   ├── model/
│   │   ├── GatedGCNLSPE.py
│   │   └── model.py
│   ├── weight/
│   │   └── random/
│   └── inference.py
├── example/
│   ├── prot.pdb
│   ├── ligs.sdf
│   └── run.sh
├── pyproject.toml
└── README.md
```

## Citation

```bibtex
@article{sim2026bapred,
  title     = {BA-Pred and RMSD-Pred: Integrated Graph Neural Network Models for Accurate Protein--Ligand Binding Affinity and Binding Pose Prediction},
  author    = {Sim, Jaemin and Lee, Juyong},
  journal   = {Journal of Chemical Information and Modeling},
  year      = {2026},
  doi       = {10.1021/acs.jcim.5c02591},
  publisher = {American Chemical Society (ACS)}
}
```

## License

Apache License 2.0
