import torch
from einops import reduce


def channel_max(data):
    return reduce(data, "b f h w -> () f () ()", "max")


def channel_min(data):
    return reduce(data, "b f h w -> () f () ()", "min")


def convert_float_to_uint8(data):
    max_val, min_val = channel_max(data), channel_min(data)
    return {'emb': ((data - min_val) / (max_val - min_val) * 255).type(torch.uint8),
            'max': max_val.type(torch.float16), 'min': min_val.type(torch.float16)}


def convert_uint8_to_float(data, index):
    emb = data['emb'][index].float()
    max_val, min_val = [data[k].squeeze(0).float() for k in ['max', 'min']]
    return emb * (max_val - min_val) / 255 + min_val
