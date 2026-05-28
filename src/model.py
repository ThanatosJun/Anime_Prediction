import torch
from transformers import SwinModel
import torch.nn as nn


def load_model(config):
    model_name = config['model']['name']
    model = SwinModel.from_pretrained(model_name)
    return model


def get_embedding(model, pixel_values):
    outputs = model(pixel_values=pixel_values)
    return outputs.pooler_output  # (B, 1024)


def get_stage_embeddings(model, pixel_values):
    # reshaped_hidden_states: tuple of 4 tensors, each (B, C, H, W)
    #
    # stages[0]: (B, 128, 56, 56)  局部紋理、線條筆觸
    # stages[1]: (B, 256, 28, 28)  色塊分布、局部結構
    # stages[2]: (B, 512, 14, 14)  人物部位、光影風格
    # stages[3]: (B, 1024,  7,  7) 整體語義、畫風流派
    outputs = model(pixel_values=pixel_values, output_hidden_states=True)

    # 回傳: [(B,128), (B,256), (B,512), (B,1024)]
    return [stage.mean(dim=[2, 3]) for stage in outputs.reshaped_hidden_states]


class StageProjector(nn.Module):
    """將各 stage embedding 投影到相同維度。

    使用方式：
        projector = StageProjector(project_dim=256).to(device)
        optimizer = Adam(list(model.parameters()) + list(projector.parameters()))
        projected = projector(get_stage_embeddings(model, pixel_values))
        # projected: [(B,256), (B,256), (B,256), (B,256)]
    """
    _STAGE_DIMS = [128, 256, 512, 1024]

    def __init__(self, project_dim: int):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Linear(d, project_dim) for d in self._STAGE_DIMS
        ])

    def forward(self, embeds):
        # embeds: [(B,128), (B,256), (B,512), (B,1024)]
        # 回傳:   [(B,D),  (B,D),  (B,D),  (B,D)]
        return [proj(e) for proj, e in zip(self.projectors, embeds)]
