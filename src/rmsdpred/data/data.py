import os
import torch

from rdkit import Chem  # type: ignore
from meeko import PDBQTMolecule, RDKitMolCreate

from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import scatter, to_torch_csr_tensor, to_edge_index, get_self_loop_attr

from .ligand_atom_feature import get_mol_coordinate, get_atom_feature, get_bond_feature
from .protein_atom_feature import res_emb, emb, aa_letter

"""PyG port of data.py + protein_atom_feature.py + ligand_atom_feature graph builders.

Builds torch_geometric Data objects instead of DGLGraph. random_walk_pe matches
dgl.random_walk_pe (RW = D^{-1} A row-normalized, PE = diag(RW^{t+1})), sparse CSR.
Node order is preserved through pocket selection (ascending mask indices, same as
dgl.node_subgraph) so Embedding lookups / pooling / complex stitching stay aligned.
"""


def random_walk_pe(edge_index, num_nodes, k):
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, k).float()
    row = edge_index[0]
    deg = scatter(torch.ones(row.size(0)), row, dim_size=num_nodes, reduce='sum').clamp(min=1)
    value = (1.0 / deg)[row]
    adj = to_torch_csr_tensor(edge_index, value, size=(num_nodes, num_nodes))

    def diag(out):
        return get_self_loop_attr(*to_edge_index(out), num_nodes=num_nodes)

    out = adj
    pe = [diag(out)]
    for _ in range(k - 1):
        out = out @ adj
        pe.append(diag(out))
    return torch.stack(pe, dim=-1).float()


def scaler(distance):
    scale_list = [1.5 ** x for x in range(15)]
    center_list = [0 for _ in range(15)]
    scaled_tensor = torch.stack(
        [torch.exp(-((distance - center) ** 2) / float(scale))
         for scale, center in zip(scale_list, center_list)], axis=1,
    )
    return scaled_tensor


def prot_to_graph(pdb):
    lines = open(pdb).readlines()

    token_res = []
    token_atom = []
    coords = []
    for line in lines:
        res_type = line[17:20].strip()
        if line[:4] in ['ATOM', 'HETA'] and line[13] != 'H' and res_type != 'HOH' and line.split()[-1] != 'H':
            atom_type = line[12:17].strip()
            if atom_type == 'OXT' or res_type in ['LLP', 'PTR']:
                continue
            elif atom_type == res_type or atom_type == res_type[:2]:
                res_type = 'METAL'
                atom_type = 'METAL'
            elif res_type not in aa_letter:
                res_type = 'XXX'
                if not atom_type == 'SE':
                    atom_type = line[13]
            xyz = [float(line[idx:idx + 8]) for idx in range(30, 54, 8)]
            coords.append(xyz)
            token_res.append(res_emb.get(res_type, 20))
            token_atom.append(emb.get((res_type, atom_type), 168))

    n = len(token_atom)
    g = Data(
        token_res=torch.as_tensor(token_res).int(),
        token_atom=torch.as_tensor(token_atom).int(),
        coord=torch.as_tensor(coords).float(),
    )
    g.num_nodes = n
    return g


def mol_to_graph(mol):
    n = mol.GetNumAtoms()
    coord = get_mol_coordinate(mol)
    h = get_atom_feature(mol)
    edge_index, e = get_bond_feature(mol)

    g = Data(
        feat=h,
        edge_index=edge_index,
        edge_attr=e,
        coord=coord,
    )
    g.num_nodes = n
    return g


def pl_to_c_graph(gp, gl, cutoff=5):
    pcoord = gp.coord
    lcoord = gl.coord
    ccoord = torch.cat([pcoord, lcoord])
    npa = len(pcoord)
    nla = len(lcoord)

    distance_pl = torch.cdist(pcoord, lcoord)
    distance_pl = torch.where(distance_pl < cutoff, distance_pl, torch.zeros_like(distance_pl)).to_sparse()

    u, v = distance_pl.indices()
    dist = distance_pl.values()

    u, v = torch.cat([u, v + npa]), torch.cat([v + npa, u])
    dist = torch.cat([dist, dist])

    g = Data(
        edge_index=torch.stack([u, v], dim=0),
        edge_attr=scaler(dist),
        coord=ccoord,
    )
    g.num_nodes = npa + nla
    return g


def get_all_graph(gp, gl, cutoff=10):
    pcoord = gp.coord
    lcoord = gl.coord

    distance_pl = torch.cdist(pcoord, lcoord)

    mask = torch.where(distance_pl < cutoff, 1, 0).sum(1)
    mask = torch.where(mask > 1, 1, 0).bool()

    # node_subgraph: keep masked protein nodes (ascending order, no edges existed yet)
    gp = Data(
        token_res=gp.token_res[mask],
        token_atom=gp.token_atom[mask],
        coord=gp.coord[mask],
    )
    pcoord = gp.coord
    gp.num_nodes = pcoord.size(0)

    distance_pp = torch.cdist(pcoord, pcoord)
    distance_pp_select = torch.where(distance_pp < 4, distance_pp, torch.zeros_like(distance_pp)).to_sparse()
    u, v = distance_pp_select.indices()
    dist = distance_pp_select.values()

    gp.edge_index = torch.stack([u, v], dim=0)
    gp.edge_attr = scaler(dist)

    gc = pl_to_c_graph(gp, gl)

    gp.pos_enc = random_walk_pe(gp.edge_index, gp.num_nodes, 20)
    gl.pos_enc = random_walk_pe(gl.edge_index, gl.num_nodes, 20)

    return gp, gl, gc


def _process_dlg_pdbqt(file_path, is_dlg, only_cluster_leads=True):
    name = os.path.basename(file_path).split('.')[0]
    pdbqt_mol = PDBQTMolecule.from_file(file_path, name=name, is_dlg=is_dlg, skip_typing=True)
    rdkit_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol, only_cluster_leads=only_cluster_leads, keep_flexres=False)
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
    supplier = (Chem.MolFromMol2Block('@<TRIPOS>MOLECULE' + block, sanitize=False) for block in mol2_blocks[1:])
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
            ligand_names.append(f"{mol_name}_{idx}")
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
            fm, fe, fn, fa = process_ligand_file(line, only_cluster_leads=only_cluster_leads)
            lig_mols.extend(fm)
            err_tags.extend(fe)
            lig_names.extend(fn)
            adg_scores.extend(fa)
        return lig_mols, err_tags, lig_names, adg_scores
    elif file_extension in ['.sdf', '.mol2', '.dlg', '.pdbqt']:
        return process_ligand_file(file_path, only_cluster_leads=only_cluster_leads)
    else:
        raise ValueError("Unsupported file type. Use '.txt', '.sdf', '.mol2', '.dlg', or '.pdbqt'.")


class PoseSelectionDataset(Dataset):
    def __init__(self, protein_pdb, ligand_file, only_cluster_leads=True):
        super().__init__()
        self.lig_mols, self.err_tags, self.lig_names, self.adg_scores = load_ligands(
            ligand_file, only_cluster_leads=only_cluster_leads
        )
        self.gp = prot_to_graph(protein_pdb)

    def __getitem__(self, idx):
        try:
            mol = self.lig_mols[idx]
            gl = mol_to_graph(mol)
            gp, gl, gc = get_all_graph(self.gp, gl)
            error = self.err_tags[idx]
            name = self.lig_names[idx]
            adg_score = self.adg_scores[idx] if idx < len(self.adg_scores) else float('nan')
        except Exception:
            gl = self.lig_dummy_graph(num_nodes=3)
            gp, gl, gc = get_all_graph(self.gp, gl)
            error = self.err_tags[idx]
            name = self.lig_names[idx]
            adg_score = float('nan')

        return gp, gl, gc, error, name, adg_score

    def __len__(self):
        return len(self.lig_mols)

    def lig_dummy_graph(self, num_nodes):
        edge_index = torch.randint(0, num_nodes, (2, 10))
        g = Data(
            feat=torch.zeros((num_nodes, 57)).float(),
            edge_index=edge_index,
            edge_attr=torch.zeros((10, 13)).float(),
            coord=torch.randn((num_nodes, 3)).float(),
        )
        g.num_nodes = num_nodes
        return g


def collate_pyg(samples):
    gps, gls, gcs, errors, names, adg = zip(*samples)
    bgp = Batch.from_data_list(list(gps))
    bgl = Batch.from_data_list(list(gls))
    bgc = Batch.from_data_list(list(gcs))
    errors = torch.tensor(errors)
    adg = torch.tensor(adg)
    return bgp, bgl, bgc, errors, list(names), adg
