import torch, dgl  # type: ignore

amino_acid_mapping = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}

amino_acid_mapping_reverse = {v: k for k, v in amino_acid_mapping.items()}
amino_acid_3_to_int = {amino_acid_mapping_reverse[k]: i for i, k in enumerate(sorted(amino_acid_mapping_reverse.keys()))}
amino_acid_1_to_int = {k: i for i, k in enumerate(sorted(amino_acid_mapping_reverse.keys()))}

aa_letter = list(amino_acid_mapping.keys())

secondary_structure_dict = {"H": 0, "B": 1, "E": 2, "G": 3, "I": 4, "T": 5, "S": 6, "-": 7}

res_emb = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    'XXX': 20, 'METAL': 21,
}

emb = {
    ('ALA', 'C'): 0, ('ALA', 'CA'): 1, ('ALA', 'CB'): 2, ('ALA', 'N'): 3, ('ALA', 'O'): 4,
    ('ARG', 'C'): 5, ('ARG', 'CA'): 6, ('ARG', 'CB'): 7, ('ARG', 'CD'): 8, ('ARG', 'CG'): 9, ('ARG', 'CZ'): 10, ('ARG', 'N'): 11, ('ARG', 'NE'): 12, ('ARG', 'NH1'): 13, ('ARG', 'NH2'): 14, ('ARG', 'O'): 15,
    ('ASN', 'C'): 16, ('ASN', 'CA'): 17, ('ASN', 'CB'): 18, ('ASN', 'CG'): 19, ('ASN', 'N'): 20, ('ASN', 'ND2'): 21, ('ASN', 'O'): 22, ('ASN', 'OD1'): 23,
    ('ASP', 'C'): 24, ('ASP', 'CA'): 25, ('ASP', 'CB'): 26, ('ASP', 'CG'): 27, ('ASP', 'N'): 28, ('ASP', 'O'): 29, ('ASP', 'OD1'): 30, ('ASP', 'OD2'): 31,
    ('CYS', 'C'): 32, ('CYS', 'CA'): 33, ('CYS', 'CB'): 34, ('CYS', 'N'): 35, ('CYS', 'O'): 36, ('CYS', 'SG'): 37,
    ('GLN', 'C'): 38, ('GLN', 'CA'): 39, ('GLN', 'CB'): 40, ('GLN', 'CD'): 41, ('GLN', 'CG'): 42, ('GLN', 'N'): 43, ('GLN', 'NE2'): 44, ('GLN', 'O'): 45, ('GLN', 'OE1'): 46,
    ('METAL', 'METAL'): 47,
    ('GLU', 'C'): 48, ('GLU', 'CA'): 49, ('GLU', 'CB'): 50, ('GLU', 'CD'): 51, ('GLU', 'CG'): 52, ('GLU', 'N'): 53, ('GLU', 'O'): 54, ('GLU', 'OE1'): 55, ('GLU', 'OE2'): 56,
    ('GLY', 'C'): 57, ('GLY', 'CA'): 58, ('GLY', 'N'): 59, ('GLY', 'O'): 60,
    ('HIS', 'C'): 61, ('HIS', 'CA'): 62, ('HIS', 'CB'): 63, ('HIS', 'CD2'): 64, ('HIS', 'CE1'): 65, ('HIS', 'CG'): 66, ('HIS', 'N'): 67, ('HIS', 'ND1'): 68, ('HIS', 'NE2'): 69, ('HIS', 'O'): 70,
    ('ILE', 'C'): 71, ('ILE', 'CA'): 72, ('ILE', 'CB'): 73, ('ILE', 'CD1'): 74, ('ILE', 'CG1'): 75, ('ILE', 'CG2'): 76, ('ILE', 'N'): 77, ('ILE', 'O'): 78,
    ('LEU', 'C'): 79, ('LEU', 'CA'): 80, ('LEU', 'CB'): 81, ('LEU', 'CD1'): 82, ('LEU', 'CD2'): 83, ('LEU', 'CG'): 84, ('LEU', 'N'): 85, ('LEU', 'O'): 86,
    ('LYS', 'C'): 87, ('LYS', 'CA'): 88, ('LYS', 'CB'): 89, ('LYS', 'CD'): 90, ('LYS', 'CE'): 91, ('LYS', 'CG'): 92, ('LYS', 'N'): 93, ('LYS', 'NZ'): 94, ('LYS', 'O'): 95,
    ('MET', 'C'): 96, ('MET', 'CA'): 97, ('MET', 'CB'): 98, ('MET', 'CE'): 99, ('MET', 'CG'): 100, ('MET', 'N'): 101, ('MET', 'O'): 102, ('MET', 'SD'): 103,
    ('PHE', 'C'): 104, ('PHE', 'CA'): 105, ('PHE', 'CB'): 106, ('PHE', 'CD1'): 107, ('PHE', 'CD2'): 108, ('PHE', 'CE1'): 109, ('PHE', 'CE2'): 110, ('PHE', 'CG'): 111, ('PHE', 'CZ'): 112, ('PHE', 'N'): 113, ('PHE', 'O'): 114,
    ('PRO', 'C'): 115, ('PRO', 'CA'): 116, ('PRO', 'CB'): 117, ('PRO', 'CD'): 118, ('PRO', 'CG'): 119, ('PRO', 'N'): 120, ('PRO', 'O'): 121,
    ('SER', 'C'): 122, ('SER', 'CA'): 123, ('SER', 'CB'): 124, ('SER', 'N'): 125, ('SER', 'O'): 126, ('SER', 'OG'): 127,
    ('THR', 'C'): 128, ('THR', 'CA'): 129, ('THR', 'CB'): 130, ('THR', 'CG2'): 131, ('THR', 'N'): 132, ('THR', 'O'): 133, ('THR', 'OG1'): 134,
    ('TRP', 'C'): 135, ('TRP', 'CA'): 136, ('TRP', 'CB'): 137, ('TRP', 'CD1'): 138, ('TRP', 'CD2'): 139, ('TRP', 'CE2'): 140, ('TRP', 'CE3'): 141, ('TRP', 'CG'): 142, ('TRP', 'CH2'): 143, ('TRP', 'CZ2'): 144, ('TRP', 'CZ3'): 145, ('TRP', 'N'): 146, ('TRP', 'NE1'): 147, ('TRP', 'O'): 148,
    ('TYR', 'C'): 149, ('TYR', 'CA'): 150, ('TYR', 'CB'): 151, ('TYR', 'CD1'): 152, ('TYR', 'CD2'): 153, ('TYR', 'CE1'): 154, ('TYR', 'CE2'): 155, ('TYR', 'CG'): 156, ('TYR', 'CZ'): 157, ('TYR', 'N'): 158, ('TYR', 'O'): 159, ('TYR', 'OH'): 160,
    ('VAL', 'C'): 161, ('VAL', 'CA'): 162, ('VAL', 'CB'): 163, ('VAL', 'CG1'): 164, ('VAL', 'CG2'): 165, ('VAL', 'N'): 166, ('VAL', 'O'): 167,
    ('UNK'): 168,
    ('XXX', 'C'): 169, ('XXX', 'N'): 170, ('XXX', 'O'): 171, ('XXX', 'S'): 172, ('XXX', 'P'): 173, ('XXX', 'SE'): 174,
}


def get_all_graph(gp, gl, cutoff=10):
    pcoord = gp.ndata['coord']
    lcoord = gl.ndata['coord']

    distance_pl = torch.cdist(pcoord, lcoord)

    distancs_mask = torch.where(distance_pl < cutoff, 1, 0).sum(1)
    distancs_mask = torch.where(distancs_mask > 1, 1, 0).bool()

    gp = dgl.node_subgraph(gp, distancs_mask)
    pcoord = gp.ndata['coord']

    distance_pp = torch.cdist(pcoord, pcoord)
    distance_pp_select = torch.where(distance_pp < 4, distance_pp, 0).to_sparse()
    u, v = distance_pp_select.indices()
    dist = distance_pp_select.values()

    gp.add_edges(u, v)
    gp.edata['dist'] = scaler(dist)

    gc = pl_to_c_graph(gp, gl)
    gp.ndata['pos_enc'] = dgl.random_walk_pe(gp, 20)
    gl.ndata['pos_enc'] = dgl.random_walk_pe(gl, 20)

    return gp, gl, gc


def pl_to_c_graph(gp, gl, cutoff=5):
    pcoord = gp.ndata['coord']
    lcoord = gl.ndata['coord']
    ccoord = torch.cat([pcoord, lcoord])
    npa = len(pcoord)
    nla = len(lcoord)

    distance_pl = torch.cdist(pcoord, lcoord)
    distance_pl = torch.where(distance_pl < cutoff, distance_pl, 0).to_sparse()

    u, v = distance_pl.indices()
    dist = distance_pl.values()

    u, v = torch.cat([u, v + npa]), torch.cat([v + npa, u])
    dist = torch.cat([dist, dist])

    g = dgl.DGLGraph()
    g.add_nodes(npa + nla)
    g.add_edges(u, v)
    g.ndata['coord'] = ccoord
    g.edata['dist'] = scaler(dist)

    return g


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
    g = dgl.DGLGraph()
    g.add_nodes(n)

    g.ndata['token_res'] = torch.as_tensor(token_res).int()
    g.ndata['token_atom'] = torch.as_tensor(token_atom).int()
    g.ndata['coord'] = torch.as_tensor(coords).float()

    return g
