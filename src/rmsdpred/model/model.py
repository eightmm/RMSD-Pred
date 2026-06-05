import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool

from .GatedGCNLSPE import GatedGCNLSPELayer

"""PredictionRMSD (PyG). Param names identical -> state_dict loads 1:1.

Inputs gp, gl, gc are torch_geometric Batch objects:
    gp: .token_res, .token_atom (int), .coord, .edge_attr (dist), .pos_enc, .edge_index, .batch
    gl: .feat, .coord, .edge_attr (feat), .pos_enc, .edge_index, .batch
    gc: .edge_attr (dist), .coord, .edge_index, .batch  (protein-then-ligand nodes)
"""


class PredictionRMSD(nn.Module):
    def __init__(self, in_size, emb_size, intra_edge_size, inter_edge_size, pose_size, num_layers, dropout_ratio=0.15):
        super().__init__()
        self.res_token_encoder = nn.Embedding(22, int(emb_size / 2))
        self.atom_token_encoder = nn.Embedding(175, int(emb_size / 2))
        self.protein_edge_encoder = nn.Linear(15, emb_size)

        self.ligand_node_encoder = nn.Linear(in_size, emb_size)
        self.ligand_edge_encoder = nn.Linear(intra_edge_size, emb_size)

        self.protein_pose_encoder = nn.Linear(pose_size, emb_size)
        self.ligand_pose_encoder = nn.Linear(pose_size, emb_size)

        self.complex_edge_encoder = nn.Linear(15, emb_size)

        self.protein_norm = nn.LayerNorm(emb_size)
        self.ligand_norm = nn.LayerNorm(emb_size)

        blocks = [
            nn.ModuleList([
                GatedGCNLSPELayer(
                    input_dim=emb_size,
                    output_dim=emb_size,
                    dropout=0.2,
                    batch_norm=True,
                )
                for _ in range(num_layers)
            ])
            for i in range(3)
        ]

        self.protein_block = blocks[0]
        self.ligand_block = blocks[1]
        self.complex_block = blocks[2]

        self.mlp_rmsd = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),
            nn.ELU(),
            nn.Dropout(p=dropout_ratio),
            nn.Linear(emb_size, 1),
        )

    def forward(self, gp, gl, gc):
        hpr = self.res_token_encoder(gp.token_res)
        hpa = self.atom_token_encoder(gp.token_atom)

        hp = torch.cat([hpr, hpa], 1)

        ep = self.protein_edge_encoder(gp.edge_attr)
        pp = self.protein_pose_encoder(gp.pos_enc)

        hl = self.ligand_node_encoder(gl.feat)
        el = self.ligand_edge_encoder(gl.edge_attr)
        pl = self.ligand_pose_encoder(gl.pos_enc)

        ec = self.complex_edge_encoder(gc.edge_attr)

        hp = self.protein_norm(hp)
        hl = self.ligand_norm(hl)

        hp_raw = hp
        hl_raw = hl

        gp_sizes = torch.bincount(gp.batch, minlength=gp.num_graphs).tolist()
        gl_sizes = torch.bincount(gl.batch, minlength=gl.num_graphs).tolist()

        # Precompute static gather indices for protein+ligand -> complex stitching.
        # Complex node order per sample is [protein nodes of s, ligand nodes of s],
        # matching pl_to_c_graph / Batch.from_data_list. Replaces per-layer python loops.
        Np = hp.size(0)
        device = hp.device
        to_complex = []   # index into cat([hp, hl]) -> complex node order
        hp_from_c = []    # index into hc -> recover hp order
        hl_from_c = []    # index into hc -> recover hl order
        gp_off = 0
        gl_off = 0
        cpos = 0
        for gp_size, gl_size in zip(gp_sizes, gl_sizes):
            to_complex.extend(range(gp_off, gp_off + gp_size))
            to_complex.extend(range(Np + gl_off, Np + gl_off + gl_size))
            hp_from_c.extend(range(cpos, cpos + gp_size))
            cpos += gp_size
            hl_from_c.extend(range(cpos, cpos + gl_size))
            cpos += gl_size
            gp_off += gp_size
            gl_off += gl_size
        to_complex = torch.tensor(to_complex, dtype=torch.long, device=device)
        hp_from_c = torch.tensor(hp_from_c, dtype=torch.long, device=device)
        hl_from_c = torch.tensor(hl_from_c, dtype=torch.long, device=device)

        hc = None
        for (protein_layer, ligand_layer, complex_layer) in zip(self.protein_block, self.ligand_block, self.complex_block):
            hp, pp, ep = protein_layer(gp.edge_index, hp, pp, ep)
            hl, pl, el = ligand_layer(gl.edge_index, hl, pl, el)

            hc = torch.cat([hp, hl], 0)[to_complex]
            pc = torch.cat([pp, pl], 0)[to_complex]

            hc, pc, ec = complex_layer(gc.edge_index, hc, pc, ec)

            hp = hc[hp_from_c]
            hl = hc[hl_from_c]

            hp = hp + hp_raw
            hl = hl + hl_raw

        h = global_add_pool(hc, gc.batch)

        rmsd = self.mlp_rmsd(h)

        return rmsd
