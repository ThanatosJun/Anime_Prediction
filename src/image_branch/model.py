from transformers import SwinModel


def load_model(config):
    model_name = config['model']['name']
    model = SwinModel.from_pretrained(model_name)
    return model


def get_embedding(model, pixel_values):
    outputs = model(pixel_values=pixel_values)
    return outputs.pooler_output  # (B, 768)
