import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter


class SpectralRadiusNorm(BaseTransform):
    r"""Normalize adjacency by an upper‐bound on its spectral radius.

    Given an adjacency matrix \(A\) (implicitly via `edge_index` and
    edge features `E`), compute the induced 1-norm and ∞-norm:

        \[
          \|A\|_1 = \max_{j}\sum_{i} |A_{ij}|,\quad
          \|A\|_\infty = \max_{i}\sum_{j} |A_{ij}|.
        \]

    Let
        \[
          s = \min\bigl(\|A\|_1,\;\|A\|_\infty\bigr).
        \]
    Then normalize each edge feature by dividing by \(s\):

        \[
          A_{\mathrm{norm}} = \frac{A}{s},
          \quad\text{so that}\;\rho(A)\le s\;\Longrightarrow\;\rho(A_{\mathrm{norm}})\le1.
        \]

    Returns:
        dataflow (torch_geometric.Data):
            The same Data object with its edge features
            (`edge_attr` or `edge_weight`) scaled elementwise by \(1/s\).
    """

    def forward(self, data):
        """Args:
            data (torch_geometric.data.Data): The dataflow object to transform.

        Returns:
            torch_geometric.data.Data: The transformed dataflow object.
        """
        attr = getattr(data, "edge_attr", None)
        weight = getattr(data, "edge_weight", None)

        if attr is not None:
            edge_feat = attr  # shape [E, C]
        elif weight is not None:
            edge_feat = weight.view(-1, 1)  # shape [E, 1]
        else:
            edge_size = data.edge_index.size(1)
            edge_feat = data.edge_index.new_ones(edge_size, 1)  # [E, 1]

        # 2. Compute absolute values:
        abs_feat = edge_feat.abs()  # [E, C]

        # 3. Sum per-node (rows) and per-node (columns):
        row_idx = data.edge_index[0]  # source nodes, length E
        col_idx = data.edge_index[1]  # target nodes, length E

        row_sum = scatter(abs_feat, row_idx, dim=0, dim_size=data.num_nodes, reduce="sum")
        col_sum = scatter(abs_feat, col_idx, dim=0, dim_size=data.num_nodes, reduce="sum")
        # row_sum, col_sum: tensors of shape [num_nodes, C]

        # 4. Upper bound per channel:
        #    max over nodes → two [C] vectors → elementwise min → [C]
        bound = torch.min(row_sum.max(dim=0).values, col_sum.max(dim=0).values)

        # 5. Normalize (broadcast over edges) and write back:
        norm_feat = edge_feat / bound  # shape [E, C]

        if attr is not None:
            data.edge_attr = norm_feat  # preserve multi-D features
        else:
            # collapse back to 1D weight if needed
            data.edge_weight = norm_feat.view(-1)

        return data
