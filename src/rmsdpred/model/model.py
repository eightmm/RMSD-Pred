import torch
import torch.nn as nn

from dgl.nn.pytorch.glob import SumPooling

from .GatedGCNLSPE import GatedGCNLSPELayer


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

        self.sum_pooling = SumPooling()

    def forward(self, gp, gl, gc):
        hpr = self.res_token_encoder(gp.ndata['token_res'])
        hpa = self.atom_token_encoder(gp.ndata['token_atom'])

        hp = torch.cat([hpr, hpa], 1)

        ep = self.protein_edge_encoder(gp.edata['dist'])
        pp = self.protein_pose_encoder(gp.ndata['pos_enc'])

        hl = self.ligand_node_encoder(gl.ndata['feat'])
        el = self.ligand_edge_encoder(gl.edata['feat'])
        pl = self.ligand_pose_encoder(gl.ndata['pos_enc'])

        ec = self.complex_edge_encoder(gc.edata['dist'])

        hp = self.protein_norm(hp)
        hl = self.ligand_norm(hl)

        xp = gp.ndata['coord']
        xl = gl.ndata['coord']

        hp_raw = hp
        hl_raw = hl

        gp_batch_sizes = gp.batch_num_nodes()
        gl_batch_sizes = gl.batch_num_nodes()

        gp_start_indices = [0] + torch.cumsum(gp_batch_sizes[:-1], dim=0).tolist()
        gl_start_indices = [0] + torch.cumsum(gl_batch_sizes[:-1], dim=0).tolist()

        for (protein_layer, ligand_layer, complex_layer) in zip(self.protein_block, self.ligand_block, self.complex_block):
            hp, pp, ep = protein_layer(gp, hp, pp, ep)
            hl, pl, el = ligand_layer(gl, hl, pl, el)

            hc = []
            pc = []
            xc = []
            for gp_start, gp_size, gl_start, gl_size in zip(gp_start_indices, gp_batch_sizes, gl_start_indices, gl_batch_sizes):
                hp_slice = hp[gp_start:gp_start + gp_size]
                hl_slice = hl[gl_start:gl_start + gl_size]
                pp_slice = pp[gp_start:gp_start + gp_size]
                pl_slice = pl[gl_start:gl_start + gl_size]
                xp_slice = xp[gp_start:gp_start + gp_size]
                xl_slice = xl[gp_start:gp_start + gp_size]

                hc.append(torch.cat([hp_slice, hl_slice]))
                pc.append(torch.cat([pp_slice, pl_slice]))
                xc.append(torch.cat([xp_slice, xl_slice]))

            hc = torch.cat(hc)
            pc = torch.cat(pc)
            xc = torch.cat(xc)

            hc, pc, ec = complex_layer(gc, hc, pc, ec)

            hp_separated = []
            hl_separated = []
            start = 0
            for gp_size, gl_size in zip(gp_batch_sizes, gl_batch_sizes):
                hp_separated.append(hc[start: start + gp_size])
                start += gp_size
                hl_separated.append(hc[start: start + gl_size])
                start += gl_size

            hp = torch.cat(hp_separated)
            hl = torch.cat(hl_separated)

            hp += hp_raw
            hl += hl_raw

        h = self.sum_pooling(gc, hc)

        rmsd = self.mlp_rmsd(h)

        return rmsd
