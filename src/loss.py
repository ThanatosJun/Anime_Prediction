import torch
import torch.nn.functional as F


def infonce_loss(aug_emb, orig_emb, tau=0.07):
    aug_emb = F.normalize(aug_emb, dim=-1)
    orig_emb = F.normalize(orig_emb, dim=-1)

    # cosine similarity matrix: (B, B)
    sim_matrix = torch.matmul(aug_emb, orig_emb.T) / tau

    # 對角線為 positive pair
    labels = torch.arange(sim_matrix.size(0), device=aug_emb.device)

    return F.cross_entropy(sim_matrix, labels)
