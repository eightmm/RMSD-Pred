import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class GatedGCNLSPELayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, batch_norm, use_lapeig_loss=False, residual=True):
        super().__init__()
        self.in_channels = input_dim
        self.out_channels = output_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_lapeig_loss = use_lapeig_loss

        if input_dim != output_dim:
            self.residual = False

        self.A1 = nn.Linear(input_dim * 2, output_dim, bias=True)
        self.A2 = nn.Linear(input_dim * 2, output_dim, bias=True)
        self.B1 = nn.Linear(input_dim, output_dim, bias=True)
        self.B2 = nn.Linear(input_dim, output_dim, bias=True)
        self.B3 = nn.Linear(input_dim, output_dim, bias=True)
        self.C1 = nn.Linear(input_dim, output_dim, bias=True)
        self.C2 = nn.Linear(input_dim, output_dim, bias=True)

        self.bn_node_h = nn.BatchNorm1d(output_dim)
        self.bn_node_e = nn.BatchNorm1d(output_dim)

    def message_func_for_vij(self, edges):
        hj = edges.src['h']
        pj = edges.src['p']
        vij = self.A2(torch.cat((hj, pj), -1))
        return {'v_ij': vij}

    def message_func_for_pj(self, edges):
        pj = edges.src['p']
        return {'C2_pj': self.C2(pj)}

    def compute_normalized_eta(self, edges):
        return {'eta_ij': edges.data['sigma_hat_eta'] / (edges.dst['sum_sigma_hat_eta'] + 1e-6)}

    def forward(self, g, h, p, e):
        with g.local_scope():
            h_in = h
            p_in = p
            e_in = e

            g.ndata['h'] = h
            g.ndata['A1_h'] = self.A1(torch.cat((h, p), -1))
            g.ndata['B1_h'] = self.B1(h)
            g.ndata['B2_h'] = self.B2(h)

            g.ndata['p'] = p
            g.ndata['C1_p'] = self.C1(p)

            g.edata['e'] = e
            g.edata['B3_e'] = self.B3(e)

            g.apply_edges(fn.u_add_v('B1_h', 'B2_h', 'B1_B2_h'))
            g.edata['hat_eta'] = g.edata['B1_B2_h'] + g.edata['B3_e']
            g.edata['sigma_hat_eta'] = torch.sigmoid(g.edata['hat_eta'])
            g.update_all(fn.copy_e('sigma_hat_eta', 'm'), fn.sum('m', 'sum_sigma_hat_eta'))
            g.apply_edges(self.compute_normalized_eta)
            g.apply_edges(self.message_func_for_vij)
            g.edata['eta_mul_v'] = g.edata['eta_ij'] * g.edata['v_ij']
            g.update_all(fn.copy_e('eta_mul_v', 'm'), fn.sum('m', 'sum_eta_v'))
            g.ndata['h'] = g.ndata['A1_h'] + g.ndata['sum_eta_v']

            g.apply_edges(self.message_func_for_pj)
            g.edata['eta_mul_p'] = g.edata['eta_ij'] * g.edata['C2_pj']
            g.update_all(fn.copy_e('eta_mul_p', 'm'), fn.sum('m', 'sum_eta_p'))
            g.ndata['p'] = g.ndata['C1_p'] + g.ndata['sum_eta_p']

            h = g.ndata['h']
            p = g.ndata['p']
            e = g.edata['hat_eta']

            if self.batch_norm:
                h = self.bn_node_h(h)
                e = self.bn_node_e(e)

            h = F.relu(h)
            e = F.relu(e)
            p = torch.tanh(p)

            if self.residual:
                h = h_in + h
                p = p_in + p
                e = e_in + e

            h = F.dropout(h, self.dropout, training=self.training)
            p = F.dropout(p, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

            return h, p, e

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
