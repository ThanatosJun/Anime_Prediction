import torch
from transformers import SwinModel


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

    embed = []
    for stage in outputs.reshaped_hidden_states:
        embed.append(stage.mean(dim=[2, 3]))  # (B, C)

    # embed: [(B,128), (B,256), (B,512), (B,1024)]
    return embed 



