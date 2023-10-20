import torch
import base64
from pathlib import Path

from handler_nst import handler_nst
from ts.context import Context, RequestProcessor
import yaml
import io
from torchvision.utils import save_image
from PIL import Image
import os
import struct
import json

content_path  = Path("./pytorch-AdaIN/input/content/cornell.jpg")
style_path = Path("./pytorch-AdaIN/input/style/woman_with_hat_matisse.jpg")
alpha = 1.0
style = 2 

with open("model_config.yaml", "r") as stream:
    try:
        yaml_conf = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


handler = handler_nst()

class Metrics():
    def __init__(self) -> None:
        pass
    def add_time(self, a, b, c, d):
        pass

manifest = {}
manifest["model"] = {"serializedFile": 'nst.pt'}
metrics = Metrics()

context = Context(model_name='densenet',
                  model_dir='./pytorch-AdaIN',
                  manifest=manifest,
                  batch_size=1,
                  gpu=0,
                  metrics=metrics,
                  mms_version=1, model_yaml_config=yaml_conf)
req = RequestProcessor(request_header={})
context._request_processor = [req]


handler.initialize(context)

with open(content_path, "rb") as image:
    img_content = image.read()
    img_content = bytearray(img_content)

with open(style_path, "rb") as image:
    img_style = image.read()
    img_style = bytearray(img_style)


alpha = bytearray(struct.pack("f", alpha))  
style = bytearray(struct.pack("d", style))

style_path_b = bytearray()
style_path_b.extend(str(style_path).encode())

files = [{'img_content': img_content, 'style_path': style_path_b, 'alpha': alpha}]

output = handler.handle(files, context)

output_name = Path('./') / '{:s}_stylized_{:s}{:s}'.format(
    content_path.stem, style_path.stem, '.jpg')
#save_image(img, str(output_name))

with open(str(output_name), "wb") as fh:
    fh.write(base64.decodebytes((output[0].encode('UTF-8'))))