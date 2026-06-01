from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _graph_norm(adj: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        adj = adj * mask.unsqueeze(1)
    degree = torch.pow(adj.sum(-1) + 1e-8, -0.5)
    return degree.unsqueeze(-1) * adj * degree.unsqueeze(1)


def _cosine_edges(feat: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    feat = F.normalize(feat, p=2, dim=-1)
    if mask is not None:
        feat = feat * mask.unsqueeze(-1)
    return torch.bmm(feat, feat.transpose(1, 2))


class _QueryGraphConvolution(nn.Module):
    """Source-shaped GCN used by RRCP all/single models."""

    def __init__(self, hidden_dim: int, node_count: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(node_count, hidden_dim))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        node_size = adj.size(1)
        eye = torch.eye(node_size, device=adj.device).unsqueeze(0).expand_as(adj)
        adj = torch.clamp(adj, min=0.0) + eye
        adj = _graph_norm(adj)
        pre_sup = torch.matmul(feat, self.weight)
        out = torch.matmul(adj, pre_sup) + self.bias[:node_size].unsqueeze(0)
        # Keep source behavior: query token output only.
        return torch.tanh(out[:, 0, :])


class _RrcpGraphLearner(nn.Module):
    """Graph learner aligned to src/RRCP/graph.py."""

    def __init__(self, hidden_dim: int, class_num: int):
        super().__init__()
        self.alpha_it = 0.7
        self.beta_it = 0.5
        self.gcn_tt = _QueryGraphConvolution(hidden_dim=hidden_dim, node_count=class_num + 1)
        self.gcn_it = _QueryGraphConvolution(hidden_dim=hidden_dim, node_count=class_num + 1)

    def forward(
        self,
        input_text: torch.Tensor,
        input_img: torch.Tensor,
        base_text_features: torch.Tensor,
        base_img_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_tt = torch.cat([input_text, base_text_features], dim=1)
        feat_it = torch.cat([input_img, base_img_features], dim=1)
        edge_tt = _cosine_edges(feat_tt)
        edge_it = _cosine_edges(feat_it)
        graph_tt = self.gcn_tt(feat_tt, edge_tt)
        graph_it = self.gcn_it(feat_it, edge_it)
        graph_o = (graph_tt * self.alpha_it + (1.0 - self.alpha_it) * graph_it).unsqueeze(1)
        return (
            self.beta_it * base_text_features + (1.0 - self.beta_it) * graph_o,
            self.beta_it * base_img_features + (1.0 - self.beta_it) * graph_o,
        )


class _MaskedGraphConvolution(nn.Module):
    """Source-shaped variable-length GCN used by final RRCP model."""

    def __init__(self, hidden_dim: int, node_count: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(node_count, hidden_dim))
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        node_size = adj.size(1)
        eye = torch.eye(node_size, device=adj.device).unsqueeze(0).expand_as(adj)
        adj = torch.clamp(adj, min=0.0) + eye
        adj = _graph_norm(adj, mask=mask)
        pre_sup = torch.matmul(feat, self.weight)
        out = torch.matmul(adj, pre_sup) + self.bias[:node_size].unsqueeze(0)
        out = torch.tanh(out)
        return out * mask.unsqueeze(-1)


class _VariableLengthGraphLearner(nn.Module):
    """Graph learner aligned to src/graph_variable_length.py."""

    def __init__(self, hidden_dim: int, class_num: int):
        super().__init__()
        self.alpha_it = 0.7
        self.beta_it = 0.5
        self.class_num = class_num
        self.gcn_tt = _MaskedGraphConvolution(hidden_dim=hidden_dim, node_count=class_num + 1)
        self.gcn_it = _MaskedGraphConvolution(hidden_dim=hidden_dim, node_count=class_num + 1)

    def forward(
        self,
        input_text: torch.Tensor,
        input_img: torch.Tensor,
        base_text_features: torch.Tensor,
        base_img_features: torch.Tensor,
        text_mask: torch.Tensor,
        img_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_tt = torch.cat([input_text, base_text_features], dim=1)
        feat_it = torch.cat([input_img, base_img_features], dim=1)
        edge_tt = _cosine_edges(feat_tt, mask=text_mask)
        edge_it = _cosine_edges(feat_it, mask=img_mask)
        graph_tt = self.gcn_tt(feat_tt, edge_tt, text_mask)
        graph_it = self.gcn_it(feat_it, edge_it, img_mask)
        graph_o = graph_tt * self.alpha_it + (1.0 - self.alpha_it) * graph_it
        graph_o_query = graph_o[:, :1, :]
        expanded_query = graph_o_query.repeat(1, self.class_num, 1)
        text_out = self.beta_it * base_text_features + (1.0 - self.beta_it) * expanded_query
        img_out = self.beta_it * base_img_features + (1.0 - self.beta_it) * expanded_query
        return text_out * text_mask[:, 1:, None], img_out * img_mask[:, 1:, None]


class SourceFaithfulAllItemsModel(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        top_k: int,
        d_model: int,
        dropout: float = 0.0,
        strict_source: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        if strict_source and (text_dim != d_model or image_dim != d_model):
            raise ValueError(
                f"Strict source profile expects text_dim=image_dim=d_model (got {text_dim}, {image_dim}, {d_model})."
            )
        self.text_proj = nn.Identity() if strict_source else nn.Linear(text_dim, d_model)
        self.image_proj = nn.Identity() if strict_source else nn.Linear(image_dim, d_model)
        self.ret_text_proj = nn.Identity() if strict_source else nn.Linear(text_dim, d_model)
        self.ret_image_proj = nn.Identity() if strict_source else nn.Linear(image_dim, d_model)
        self.graph = _RrcpGraphLearner(hidden_dim=d_model, class_num=top_k)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.predict_linear_1 = nn.Linear(d_model * top_k * 2, d_model)
        self.predict_linear_2 = nn.Linear(d_model * 2, 1)
        self.label_embedding = nn.Linear(top_k, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_text: torch.Tensor,
        query_image: torch.Tensor,
        retrieved_text: torch.Tensor,
        retrieved_image: torch.Tensor,
        retrieved_label: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        q_text = self.text_proj(query_text).unsqueeze(1)
        q_img = self.image_proj(query_image).unsqueeze(1)
        r_text = self.ret_text_proj(retrieved_text[:, : self.top_k, :])
        r_img = self.ret_image_proj(retrieved_image[:, : self.top_k, :])
        valid_mask = mask[:, : self.top_k].unsqueeze(-1)
        r_text = r_text * valid_mask
        r_img = r_img * valid_mask
        text_emb, img_emb = self.graph(q_text, q_img, r_text, r_img)
        packed = torch.cat([img_emb, text_emb], dim=1)
        values, _ = self.multihead_attn(packed, packed, packed)
        output = self.relu(self.predict_linear_1(values.reshape(values.shape[0], -1)))
        label = self.label_embedding(retrieved_label[:, : self.top_k] * mask[:, : self.top_k])
        output = torch.cat([self.dropout(output), self.dropout(label)], dim=1)
        return self.predict_linear_2(output)


class SourceFaithfulSingleItemModel(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        d_model: int,
        dropout: float = 0.0,
        strict_source: bool = True,
    ):
        super().__init__()
        if strict_source and (text_dim != d_model or image_dim != d_model):
            raise ValueError(
                f"Strict source profile expects text_dim=image_dim=d_model (got {text_dim}, {image_dim}, {d_model})."
            )
        self.text_proj = nn.Identity() if strict_source else nn.Linear(text_dim, d_model)
        self.image_proj = nn.Identity() if strict_source else nn.Linear(image_dim, d_model)
        self.ret_text_proj = nn.Identity() if strict_source else nn.Linear(text_dim, d_model)
        self.ret_image_proj = nn.Identity() if strict_source else nn.Linear(image_dim, d_model)
        self.graph = _RrcpGraphLearner(hidden_dim=d_model, class_num=1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=8, batch_first=True)
        self.predict_linear_1 = nn.Linear(d_model * 2, d_model)
        self.predict_linear_2 = nn.Linear(d_model * 2, 1)
        self.label_embedding = nn.Linear(1, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_text: torch.Tensor,
        query_image: torch.Tensor,
        retrieved_text: torch.Tensor,
        retrieved_image: torch.Tensor,
        retrieved_label: torch.Tensor,
    ) -> torch.Tensor:
        q_text = self.text_proj(query_text).unsqueeze(1)
        q_img = self.image_proj(query_image).unsqueeze(1)
        r_text = self.ret_text_proj(retrieved_text).unsqueeze(1)
        r_img = self.ret_image_proj(retrieved_image).unsqueeze(1)
        text_emb, img_emb = self.graph(q_text, q_img, r_text, r_img)
        packed = torch.cat([img_emb, text_emb], dim=1)
        values, _ = self.multihead_attn(packed, packed, packed)
        output = self.relu(self.predict_linear_1(values.reshape(values.shape[0], -1)))
        label = self.label_embedding(retrieved_label.unsqueeze(1))
        output = torch.cat([self.dropout(output), self.dropout(label)], dim=1)
        return self.predict_linear_2(output)


class SourceFaithfulFinalModel(nn.Module):
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        top_k: int,
        d_model: int,
        threshold_of_rrcp: float,
        dropout: float = 0.0,
        strict_source: bool = True,
    ):
        super().__init__()
        self.top_k = top_k
        self.threshold = threshold_of_rrcp
        if strict_source and (text_dim != d_model or image_dim != d_model):
            raise ValueError(
                f"Strict source profile expects text_dim=image_dim=d_model (got {text_dim}, {image_dim}, {d_model})."
            )
        self.text_proj = nn.Identity() if strict_source else nn.Linear(text_dim, d_model)
        self.image_proj = nn.Identity() if strict_source else nn.Linear(image_dim, d_model)
        self.ret_text_proj = nn.Identity() if strict_source else nn.Linear(text_dim, d_model)
        self.ret_image_proj = nn.Identity() if strict_source else nn.Linear(image_dim, d_model)
        self.graph = _VariableLengthGraphLearner(hidden_dim=d_model, class_num=top_k)
        self.predict_linear_1 = nn.Linear(d_model, d_model)
        self.predict_linear_2 = nn.Linear(d_model * 2, 1)
        self.label_embedding = nn.Linear(top_k, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _prepare_selected(
        self,
        base_text: torch.Tensor,
        base_img: torch.Tensor,
        selected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, max_nodes, hidden_dim = base_text.shape
        text_processed = torch.zeros_like(base_text)
        img_processed = torch.zeros_like(base_img)
        text_mask = torch.ones(batch_size, max_nodes + 1, device=base_text.device)
        img_mask = torch.ones(batch_size, max_nodes + 1, device=base_text.device)
        for i in range(batch_size):
            valid_idx = torch.where(selected[i] > 0)[0]
            valid_nodes = int(valid_idx.numel())
            if valid_nodes > 0:
                text_processed[i, :valid_nodes] = base_text[i, valid_idx]
                img_processed[i, :valid_nodes] = base_img[i, valid_idx]
            text_mask[i, valid_nodes + 1 :] = 0.0
            img_mask[i, valid_nodes + 1 :] = 0.0
        return text_processed, img_processed, text_mask, img_mask

    def forward(
        self,
        query_text: torch.Tensor,
        query_image: torch.Tensor,
        retrieved_text: torch.Tensor,
        retrieved_image: torch.Tensor,
        retrieved_label: torch.Tensor,
        mask: torch.Tensor,
        rrcp: torch.Tensor,
    ) -> torch.Tensor:
        q_text = self.text_proj(query_text).unsqueeze(1)
        q_img = self.image_proj(query_image).unsqueeze(1)
        r_text = self.ret_text_proj(retrieved_text[:, : self.top_k, :])
        r_img = self.ret_image_proj(retrieved_image[:, : self.top_k, :])
        valid_mask = mask[:, : self.top_k]
        scores = rrcp[:, : self.top_k] * valid_mask
        selected = ((scores > self.threshold).float() * valid_mask).int()
        empty_rows = torch.all(selected == 0, dim=1)
        selected[empty_rows, 0] = 1
        text_sel, img_sel, text_mask, img_mask = self._prepare_selected(r_text, r_img, selected)
        text_emb, img_emb = self.graph(q_text, q_img, text_sel, img_sel, text_mask, img_mask)
        packed = torch.cat([img_emb, text_emb], dim=1)
        cxmi = torch.cat([scores, scores], dim=1).unsqueeze(-1)
        cxmi = cxmi / torch.clamp(cxmi.sum(dim=1, keepdim=True), min=1e-8)
        context = torch.matmul(packed.transpose(1, 2), cxmi).squeeze(-1)
        output = self.relu(self.predict_linear_1(context))
        label = self.label_embedding(retrieved_label[:, : self.top_k])
        output = torch.cat([self.dropout(output), self.dropout(label)], dim=1)
        return self.predict_linear_2(output)
