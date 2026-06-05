"""Smoke tests: dataset graph construction + model forward on the example data.

Fast (a couple of ligands on CPU) -- guards against regressions in the PyG
data pipeline and weight loading, not numerical accuracy.
"""
from pathlib import Path

import torch
import pytest

from rmsdpred.data.data import PoseSelectionDataset, collate_pyg
from rmsdpred.model.model import PredictionRMSD

ROOT = Path(__file__).resolve().parent.parent
PROT = str(ROOT / "example" / "prot.pdb")
LIG = str(ROOT / "example" / "ligs.sdf")
REG = str(ROOT / "src" / "rmsdpred" / "weight" / "random" / "random_reg_seed0.pth")
CLS = str(ROOT / "src" / "rmsdpred" / "weight" / "random" / "random_cls_seed0.pth")


def _load(path, model):
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    sd = {k.removeprefix("base_model."): v for k, v in sd.items()}
    model.load_state_dict(sd)


@pytest.fixture(scope="module")
def dataset():
    return PoseSelectionDataset(protein_pdb=PROT, ligand_file=LIG)


def test_dataset_builds_pyg_graphs(dataset):
    assert len(dataset) > 0
    gp, gl, gc, error, name, adg = dataset[0]
    assert gp.token_res.shape[0] == gp.token_atom.shape[0]  # protein tokens per node
    assert gl.feat.shape[1] == 57          # ligand node features
    assert gl.pos_enc.shape[1] == 20       # random-walk PE
    assert gc.edge_attr.shape[1] == 15     # complex edge distance features
    assert gp.edge_index.shape[0] == 2


def test_model_forward_finite(dataset):
    bgp, bgl, bgc, error, name, adg = collate_pyg([dataset[0], dataset[1]])
    reg = PredictionRMSD(57, 256, 13, 25, 20, 4, 0)
    cls = PredictionRMSD(57, 256, 13, 25, 20, 4, 0)
    _load(REG, reg)  # 1:1 load (after base_model. prefix strip) proves param names match
    _load(CLS, cls)
    reg.train(False)
    cls.train(False)
    with torch.no_grad():
        rmsd = reg(bgp, bgl, bgc).view(-1)
        prob = torch.sigmoid(cls(bgp, bgl, bgc).view(-1))
    assert rmsd.shape[0] == 2 and prob.shape[0] == 2
    assert torch.isfinite(rmsd).all()
    assert ((prob >= 0) & (prob <= 1)).all()
