import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv,TransformerConv
from torch_geometric.nn import global_max_pool as gmp
import pandas as pd
import numpy as np

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(Attention, self).__init__()
        self.project_x = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xt = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.project_xq = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, x, xq):
        x = self.project_x(x)
        xq = self.project_xq(xq)
        a = torch.cat((x, xq), 1)
        return a

class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, num_features_xc=92, n_output=1, output_dim=128, dropout=0.2, layers=3, encoder = None):
        super(GATNet, self).__init__()

        # self.encoder = encoder
        self.n_output = n_output
        self.mol_conv = nn.ModuleList([])
        self.mol_conv.append(TransformerConv(num_features_xd, num_features_xd * 4, heads=2, dropout=dropout, concat=False))
        for i in range(layers-1):
             self.mol_conv.append(TransformerConv(num_features_xd * 4, num_features_xd * 4, heads=2, dropout=dropout, concat=False))
        self.mol_out_feats = num_features_xd * 4
        self.mol_seq_fc1 = nn.Linear(num_features_xd * 4, num_features_xd * 4)
        self.mol_seq_fc2 = nn.Linear(num_features_xd * 4, num_features_xd * 4)
        self.mol_bias = nn.Parameter(torch.rand(1, num_features_xd * 4))
        torch.nn.init.uniform_(self.mol_bias, a=-0.2, b=0.2)
        self.mol_fc_g1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.clique_conv = nn.ModuleList([])
        self.clique_conv.append(TransformerConv(num_features_xc, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        for i in range(layers-1):
             self.clique_conv.append(TransformerConv(num_features_xc * 4, num_features_xc * 4, heads=2, dropout=dropout, concat=False))
        self.clique_out_feats = num_features_xc * 4
        self.clique_seq_fc1 = nn.Linear(num_features_xc * 4, num_features_xc * 4)
        self.clique_seq_fc2 = nn.Linear(num_features_xc * 4, num_features_xc * 4)
        self.clique_bias = nn.Parameter(torch.rand(1, num_features_xc * 4))
        torch.nn.init.uniform_(self.clique_bias, a=-0.2, b=0.2)
        self.clique_fc_g1 = torch.nn.Linear(num_features_xc * 4, 1024)
        self.clique_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.attention = Attention(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(output_dim * 2, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_clique, data_pre):
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        clique_x, clique_edge_index, cli_batch = data_clique.x, data_clique.edge_index, data_clique.batch

        mol_n = mol_x.size(0)
        for i in range(len(self.mol_conv)):
            x = self.mol_conv[i](mol_x, mol_edge_index)
            if i < len(self.mol_conv) - 1:
                x = self.relu(x)
            if i == 0:
                mol_x = x
                continue
            mol_z = torch.sigmoid(self.mol_seq_fc1(x) + self.mol_seq_fc2(mol_x) + self.mol_bias.expand(mol_n, self.mol_out_feats))
            mol_x = mol_z * x + (1 - mol_z) * mol_x

        x = gmp(mol_x, mol_batch)  # global pooling
        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x)

        # clique graph embedding
        xq_n = clique_x.size(0)
        for i in range(len(self.clique_conv)):
            xq = self.clique_conv[i](clique_x, clique_edge_index)
            if i < len(self.clique_conv) - 1:
                xq = self.relu(xq)
            if i == 0:
                clique_x = xq
                continue
            clique_z = torch.sigmoid(self.clique_seq_fc1(xq) + self.clique_seq_fc2(clique_x) + self.clique_bias.expand(xq_n,
                                                                                                  self.clique_out_feats))
            clique_x = clique_z * xq + (1 - clique_z) * clique_x

        xq = gmp(clique_x, cli_batch)  # global max pooling
        # flatten
        xq = self.relu(self.clique_fc_g1(xq))
        xq = self.dropout(xq)
        xq = self.clique_fc_g2(xq)
        xq = self.dropout(xq)

        a = self.attention(x,xq)
        emb = torch.stack([x, xq], dim=1)
        a = a.unsqueeze(dim=2)
        emb = (a * emb).reshape(-1, 2 * 128)
        # add some dense layers
        xc = self.fc1(emb)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)

        return out
