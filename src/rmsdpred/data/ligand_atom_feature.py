import torch  # type: ignore

from rdkit import Chem  # type: ignore

from .utils import one_hot


METAL = [
    "LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO",
    "PD", "AG", "CR", "FE", "V", "MN", "HG", 'GA', "CD", "YB", "CA", "SN", "PB", "EU",
    "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND", "GD", "TB",
    "DY", "ER", "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH", "AU",
]

PERIODIC_table = '''H  __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ __ He
Li Be __ __ __ __ __ __ __ __ __ __ B  C  N  O  F  Ne
Na Mg __ __ __ __ __ __ __ __ __ __ Al Si P  S  Cl Ar
K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe'''

PERIODIC = {}
for i, per in enumerate(PERIODIC_table.split('\n')):
    for j, atom in enumerate(per.split()):
        if atom != '__':
            PERIODIC[atom] = (i, j)

electronegativity_table = '''2.20 ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____
0.98 1.57 ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ 2.04 2.55 3.04 3.44 3.98 ____
0.93 1.31 ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ 1.61 1.90 2.19 2.58 3.16 ____
0.82 1.00 1.36 1.54 1.63 1.66 1.55 1.83 1.88 1.91 1.90 1.65 1.81 2.01 2.18 2.55 2.96 3.00
0.82 0.95 1.22 1.33 1.60 2.16 1.90 2.20 2.28 2.20 1.93 1.69 1.78 1.96 2.05 2.10 2.66 2.60'''

ELECTRONEGATIVITY = {}
for i, per in enumerate(electronegativity_table.split('\n')):
    for j, atom_electronegativity in enumerate(per.split()):
        if atom_electronegativity != '____':
            ELECTRONEGATIVITY[(i, j)] = float(atom_electronegativity)

allowable_atom = ['C', 'O', 'N', 'S', 'P', 'Se', 'F', 'Cl', 'Br', 'I', 'METAL']
allowable_period = [i for i in range(5)]
allowable_group = [i for i in range(18)]
allowable_degree = [i for i in range(7)]
allowable_totalHs = [i for i in range(5)]
allowable_hybrid = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]
allowable_bond = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
allowable_streo = [
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOTRANS,
    Chem.rdchem.BondStereo.STEREOZ,
]


def get_mol_coordinate(mol):
    return torch.tensor(mol.GetConformer().GetPositions()).float()


def atom_feature(atom):
    symbol = atom.GetSymbol()

    period, group = PERIODIC[symbol]
    negativity = [ELECTRONEGATIVITY[(period, group)] / 4]

    period = one_hot(period, allowable_period)
    group = one_hot(group, allowable_group)
    symbol = one_hot(symbol, allowable_atom)
    degree = one_hot(atom.GetDegree(), allowable_degree)
    total_H = one_hot(atom.GetTotalNumHs(), allowable_totalHs)
    hybrid = one_hot(atom.GetHybridization(), allowable_hybrid)
    aromatic = [atom.GetIsAromatic()]
    isinring = [atom.IsInRing()]
    radical = [atom.GetNumRadicalElectrons()]
    formal_charge = [atom.GetFormalCharge() * 0.2]
    return period + group + symbol + degree + total_H + hybrid + aromatic + isinring + radical + formal_charge + negativity


def bond_feature(bond):
    bond_type = one_hot(bond.GetBondType(), allowable_bond)
    bond_streo = one_hot(bond.GetStereo(), allowable_streo)
    isinring = [bond.IsInRing()]
    conjugated = [bond.GetIsConjugated()]
    return bond_type + bond_streo + isinring + conjugated


def get_atom_feature(mol):
    return torch.tensor([atom_feature(atom) for atom in mol.GetAtoms()]).float()


def get_indices(mol, smarts):
    return torch.tensor(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))


def get_bond_feature(mol):
    """Build bond graph directly in sparse form (edge_index, edge_attr).

    Returns both edge directions per bond (symmetric adjacency). The rotatable-bond
    flag replicates the original `torch.tensor([a,b]) in rotate` torch __contains__
    broadcast semantics exactly: rf=1 iff (a in rotate[:,0]) OR (b in rotate[:,1]).
    """
    rotate = get_indices(mol, "[!$(*#*)&!D1]-!@[!$(*#*)&!D1]")
    if len(rotate) != 0:
        rot_col0 = set(rotate[:, 0].tolist())
        rot_col1 = set(rotate[:, 1].tolist())
    else:
        rot_col0, rot_col1 = set(), set()

    src, dst, feats = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_feature(bond)
        for a, b in ((i, j), (j, i)):
            rf = [1 if (a in rot_col0 or b in rot_col1) else 0]
            src.append(a)
            dst.append(b)
            feats.append(bf + rf)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(feats, dtype=torch.float) if feats else torch.zeros((0, 13)).float()
    return edge_index, edge_attr


def get_distance_feature(distance):
    scale_list = [1.5 ** x for x in range(15)]
    center_list = [0 for _ in range(15)]
    scaled_distance = torch.stack(
        [torch.exp(-((distance - center) ** 2) / float(scale))
         for scale, center in zip(scale_list, center_list)], axis=1,
    )
    return scaled_distance


def get_indices_sparse(sparse, indices):
    if torch.sum(indices) == 0:
        return torch.zeros(len(sparse))
    else:
        indices = torch.where(sparse == indices, 1, 0)
        indices = torch.sum(indices, dim=-2)
        return indices


def get_smarts_feature(mol, smarts, index):
    indices = get_indices(mol, smarts)
    indices = get_indices_sparse(index, indices)
    return indices


def get_interact_feature(pmol, lmol, protein_node_idx, ligand_node_idx):
    hydrogen_accept_smarts = "[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]"
    hydrogen_donor_smarts = "[!$([#6,H0,-,-2,-3])]"

    electron_accept_smarts = "[!H0;F,Cl,Br,I,N+,$([OH]-*=[!#6]),+]"
    electron_donor_smarts = "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]"

    hydrophobic_smarts = "[C,c,S&H0&v2,F,Cl,Br,I&!$(C=[O,N,P,S])&!$(C#N);!$(C=O)]"

    l_hydrogen_accpt = get_smarts_feature(lmol, hydrogen_accept_smarts, ligand_node_idx)
    p_hydrogen_accpt = get_smarts_feature(pmol, hydrogen_accept_smarts, protein_node_idx)

    l_hydrogen_donor = get_smarts_feature(lmol, hydrogen_donor_smarts, ligand_node_idx)
    p_hydrogen_donor = get_smarts_feature(pmol, hydrogen_donor_smarts, protein_node_idx)

    l_electron_accpt = get_smarts_feature(lmol, electron_accept_smarts, ligand_node_idx)
    p_electron_accpt = get_smarts_feature(pmol, electron_accept_smarts, protein_node_idx)

    l_electron_donor = get_smarts_feature(lmol, electron_donor_smarts, ligand_node_idx)
    p_electron_donor = get_smarts_feature(pmol, electron_donor_smarts, protein_node_idx)

    l_hydrophobic = get_smarts_feature(lmol, hydrophobic_smarts, ligand_node_idx)
    p_hydrophobic = get_smarts_feature(pmol, hydrophobic_smarts, protein_node_idx)

    interact_adj = torch.stack(
        [l_hydrogen_accpt, l_hydrogen_donor, p_hydrogen_accpt, p_hydrogen_donor,
         l_electron_accpt, l_electron_donor, p_electron_accpt, p_electron_donor,
         l_hydrophobic, p_hydrophobic], dim=1,
    )
    return interact_adj


