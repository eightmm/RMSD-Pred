import os
import torch
import dgl

from rdkit import Chem  # type: ignore
from meeko import PDBQTMolecule, RDKitMolCreate

from dgl.data import DGLDataset  # type: ignore

from .ligand_atom_feature import mol_to_graph
from .protein_atom_feature import get_all_graph, prot_to_graph


def _process_dlg_pdbqt(file_path, is_dlg, only_cluster_leads=True):
    name = os.path.basename(file_path).split('.')[0]
    pdbqt_mol = PDBQTMolecule.from_file(
        file_path, name=name, is_dlg=is_dlg, skip_typing=True
    )
    rdkit_mols = RDKitMolCreate.from_pdbqt_mol(
        pdbqt_mol, only_cluster_leads=only_cluster_leads, keep_flexres=False
    )
    sdf_string, _ = RDKitMolCreate.write_sd_string(pdbqt_mol, only_cluster_leads=only_cluster_leads)

    adg_score = []
    for line in sdf_string.split('\n'):
        if '{' in line:
            words = line.split(',')
            free_energy = words[1].split(':')[1].strip()
            adg_score.append(float(free_energy))

    mols, err_tags, names = [], [], []
    for i, conf in enumerate(rdkit_mols[0].GetConformers()):
        mol = Chem.Mol(rdkit_mols[0])
        if mol is None:
            mols.append(None)
            err_tags.append(1)
        else:
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            mol = Chem.RemoveHs(mol)
            mols.append(mol)
            err_tags.append(0)
        names.append(f"{name}_{i}")
    return mols, err_tags, names, adg_score


def _process_sdf(file_path):
    supplier = Chem.SDMolSupplier(file_path, sanitize=False)
    return _process_supplier(supplier, file_path)


def _process_mol2(file_path):
    with open(file_path, 'r') as f:
        mol2_data = f.read()
    mol2_blocks = mol2_data.split('@<TRIPOS>MOLECULE')
    supplier = (
        Chem.MolFromMol2Block('@<TRIPOS>MOLECULE' + block, sanitize=False)
        for block in mol2_blocks[1:]
    )
    return _process_supplier(supplier, file_path)


def _process_supplier(supplier, file_path):
    ligands, err_tag, ligand_names = [], [], []
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for idx, mol in enumerate(supplier):
        if mol is not None:
            mol = Chem.RemoveHs(mol)
            ligands.append(mol)
            err_tag.append(0)
            mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') and mol.GetProp('_Name').strip() else base_name
            ligand_name = f"{mol_name}_{idx}"
            ligand_names.append(ligand_name)
        else:
            ligands.append(None)
            err_tag.append(1)
            ligand_names.append(f"{base_name}_err_{idx}")

    return ligands, err_tag, ligand_names, [float('nan')] * len(ligands)


def process_ligand_file(file_path, only_cluster_leads=True):
    extension = os.path.splitext(file_path)[-1].lower()

    if extension == '.dlg':
        return _process_dlg_pdbqt(file_path, is_dlg=True, only_cluster_leads=only_cluster_leads)
    elif extension == '.pdbqt':
        return _process_dlg_pdbqt(file_path, is_dlg=False, only_cluster_leads=only_cluster_leads)
    elif extension == '.sdf':
        return _process_sdf(file_path)
    elif extension == '.mol2':
        return _process_mol2(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def load_ligands(file_path, only_cluster_leads=True):
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == '.txt':
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        lig_mols, err_tags, lig_names, adg_scores = [], [], [], []
        for line in lines:
            assert os.path.isfile(line), f"File not found: {line}"
            file_ligands, file_err_tag, file_ligand_names, file_adg_score = process_ligand_file(line, only_cluster_leads=only_cluster_leads)
            lig_mols.extend(file_ligands)
            err_tags.extend(file_err_tag)
            lig_names.extend(file_ligand_names)
            adg_scores.extend(file_adg_score)
        return lig_mols, err_tags, lig_names, adg_scores

    elif file_extension in ['.sdf', '.mol2', '.dlg', '.pdbqt']:
        return process_ligand_file(file_path, only_cluster_leads=only_cluster_leads)
    else:
        raise ValueError("Unsupported file type. Use '.txt', '.sdf', '.mol2', '.dlg', or '.pdbqt'.")


class PoseSelectionDataset(DGLDataset):
    def __init__(self, protein_pdb, ligand_file, only_cluster_leads=True):
        super().__init__(name='Protein Ligand Binding Conformation RMSD Prediction')
        self.lig_mols, self.err_tags, self.lig_names, self.adg_scores = load_ligands(ligand_file, only_cluster_leads=only_cluster_leads)
        self.gp = prot_to_graph(protein_pdb)

    def __getitem__(self, idx):
        try:
            mol = self.lig_mols[idx]
            gl = mol_to_graph(mol)
            gp, gl, gc = get_all_graph(self.gp, gl)
            error = self.err_tags[idx]
            name = self.lig_names[idx]
            adg_score = self.adg_scores[idx] if idx < len(self.adg_scores) else float('nan')
        except:
            gl = self.lig_dummy_graph(num_nodes=3)
            gp, gl, gc = get_all_graph(self.gp, gl)
            error = self.err_tags[idx]
            name = self.lig_names[idx]
            adg_score = float('nan')

        return gp, gl, gc, error, name, adg_score

    def __len__(self):
        return len(self.lig_mols)

    def lig_dummy_graph(self, num_nodes):
        src = torch.randint(0, num_nodes, (10,))
        dst = torch.randint(0, num_nodes, (10,))
        gl = dgl.graph((src, dst), num_nodes=num_nodes)
        gl.ndata['feat'] = torch.zeros((num_nodes, 57)).float()
        gl.ndata['pos_enc'] = torch.zeros((num_nodes, 20)).float()
        gl.ndata['coord'] = torch.randn((num_nodes, 3)).float()
        gl.edata['feat'] = torch.zeros((10, 13)).float()
        return gl
