"""
Graph encoder over the partial network (proposal: explicit topology encoding; also the
representation on which the learned backward policy scores the parents of a state).

Message passing on the padded dense adjacency A[b, parent, child]:
    h_v <- ReLU( W_s h_v + W_d mean_{children c} h_c + W_u mean_{parents p} h_p + b )
for `layers` rounds, followed by a linear projection to the token size.  Dense adjacency keeps
the module dependency-free (no torch_geometric); V <= 2N - 1 + 2 R_MAX nodes per state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):

    def __init__(self, in_dim, hidden, layers, out_dim):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.w_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.w_down = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.w_up = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x, adj, node_mask):
        """x [B, V, F], adj [B, V, V] (parent -> child), node_mask [B, V] -> [B, V, out_dim]"""
        h = F.relu(self.inp(x))
        deg_down = adj.sum(-1, keepdim=True).clamp(min=1.0)                 # children per node
        deg_up = adj.sum(-2).unsqueeze(-1).clamp(min=1.0)                   # parents per node
        adj_t = adj.transpose(1, 2)
        for ws, wd, wu in zip(self.w_self, self.w_down, self.w_up):
            m_down = torch.bmm(adj, h) / deg_down                            # mean over children
            m_up = torch.bmm(adj_t, h) / deg_up                              # mean over parents
            h = F.relu(ws(h) + wd(m_down) + wu(m_up))
            h = h * node_mask.unsqueeze(-1).to(h.dtype)
        return self.out(h)
