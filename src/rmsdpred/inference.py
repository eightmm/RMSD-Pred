import os
from pathlib import Path

import torch
import pandas as pd

from tqdm import tqdm

from dgl.dataloading import GraphDataLoader

from rmsdpred.data.data import PoseSelectionDataset
from rmsdpred.model.model import PredictionRMSD


WEIGHT_DIR = Path(__file__).resolve().parent / "weight"
DEFAULT_REG_WEIGHT = str(WEIGHT_DIR / "random" / "random_reg_seed0.pth")
DEFAULT_CLS_WEIGHT = str(WEIGHT_DIR / "random" / "random_cls_seed0.pth")


def inference(protein_pdb, ligand_file, output, batch_size, reg_weight=DEFAULT_REG_WEIGHT, cls_weight=DEFAULT_CLS_WEIGHT, device='cpu'):
    dataset = PoseSelectionDataset(
        protein_pdb=protein_pdb,
        ligand_file=ligand_file
    )

    loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    rmsd_model = PredictionRMSD(57, 256, 13, 25, 20, 4, 0).to(device)
    prob_model = PredictionRMSD(57, 256, 13, 25, 20, 4, 0).to(device)

    def _load_state_dict(ckpt_path, model):
        sd = torch.load(ckpt_path, weights_only=False, map_location=device)['model_state_dict']
        sd = {k.removeprefix('base_model.'): v for k, v in sd.items()}
        model.load_state_dict(sd)

    _load_state_dict(reg_weight, rmsd_model)
    _load_state_dict(cls_weight, prob_model)

    rmsd_model.eval()
    prob_model.eval()

    results = {
        "Name": [],
        "pRMSD": [],
        "Is_Above_2A": [],
        "ADG_Score": [],
    }

    with torch.no_grad():
        progress_bar = tqdm(total=len(loader.dataset), unit='ligand')

        for data in loader:
            bgp, bgl, bgc, error, names, adg_score = data
            bgp, bgl, bgc = bgp.to(device), bgl.to(device), bgc.to(device)

            rmsd = rmsd_model(bgp, bgl, bgc)
            prob = prob_model(bgp, bgl, bgc)

            rmsd = rmsd.view(-1)
            prob = prob.view(-1)

            prob = torch.sigmoid(prob)

            rmsd[error == 1] = torch.tensor(float('nan'))
            prob[error == 1] = torch.tensor(float('nan'))

            results["Name"].extend(names)
            results["pRMSD"].extend(rmsd.tolist())
            results["Is_Above_2A"].extend(prob.tolist())
            results["ADG_Score"].extend(adg_score.tolist())
            progress_bar.update(len(names))

        progress_bar.close()

    df = pd.DataFrame(results)
    df = df.round(4)
    df.to_csv(output, sep='\t', na_rep='NaN', index=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='RMSD-Pred: Predict protein-ligand binding pose RMSD using Graph Neural Networks'
    )
    parser.add_argument(
        '-r', '--protein_pdb',
        required=True,
        help='Receptor protein PDB file'
    )
    parser.add_argument(
        '-l', '--ligand_file',
        required=True,
        help='Ligand file (.sdf, .mol2, .dlg, .pdbqt, or .txt list)'
    )
    parser.add_argument(
        '-o', '--output',
        default='./result.tsv',
        help='Output results file (default: result.tsv)'
    )
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='Batch size for inference (default: 128)'
    )
    parser.add_argument(
        '--ncpu',
        default=4,
        type=int,
        help="Number of CPU workers (default: 4)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Compute device: cpu or cuda (default: cuda)'
    )
    parser.add_argument(
        '--reg_weight',
        type=str,
        default=DEFAULT_REG_WEIGHT,
        help='Regression model weight file (default: packaged random_reg_seed0)'
    )
    parser.add_argument(
        '--cls_weight',
        type=str,
        default=DEFAULT_CLS_WEIGHT,
        help='Classification model weight file (default: packaged random_cls_seed0)'
    )

    args = parser.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.ncpu)
    os.environ["MKL_NUM_THREADS"] = str(args.ncpu)
    torch.set_num_threads(args.ncpu)

    if args.device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    inference(
        protein_pdb=args.protein_pdb,
        ligand_file=args.ligand_file,
        output=args.output,
        batch_size=args.batch_size,
        reg_weight=args.reg_weight,
        cls_weight=args.cls_weight,
        device=device
    )


if __name__ == "__main__":
    main()
