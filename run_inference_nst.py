
from utils import convert_b64, predict_custom_trained_model_sample
import requests
from pathlib import Path
from torchvision.utils import save_image
import base64
import struct
import time

endpoint = 'gcp' #gcp or local

content_path  = Path("./pytorch-AdaIN/input/content/mehmet.jpg")
style_path = Path("./pytorch-AdaIN/input/style/sketch.png")
alpha = 1.0


if endpoint == 'local':

    with open(content_path, "rb") as image:
        img_content = image.read()
        img_content = bytearray(img_content)

    with open(style_path, "rb") as image:
        img_style = image.read()
        img_style = bytearray(img_style)


    output = requests.post("http://localhost:8080/predictions/nst", files=[('img_content',  img_content), ('img_style', img_style), ('alpha', bytearray(struct.pack("f", alpha)))])

    img = base64.b64encode(output.content)

    output_name = Path('./') / '{:s}_stylized_{:s}{:s}'.format(
        content_path.stem, style_path.stem, '.jpg')
    #save_image(img, str(output_name))

    with open(str(output_name), "wb") as fh:
        fh.write(base64.decodebytes(img))

else:

    alpha_p = base64.encodebytes(struct.pack("f", alpha)).decode("utf-8")

    instances = [{  "img_content": {"b64": convert_b64(content_path)} ,
                    "img_style": {"b64": convert_b64(style_path)},
                    "alpha": {"b64": alpha_p}}]

    
    output = predict_custom_trained_model_sample(
        project="116875092850",
        endpoint_id="2615538051362848768",
        location="us-central1",
        instances=instances
    )
    

    output_name = Path('./') / '{:s}_stylized_{:s}_alpha_{:s}{:s}'.format(
        content_path.stem, style_path.stem, str(alpha),'.jpg')
    #save_image(img, str(output_name))

    with open(str(output_name), "wb") as fh:
        fh.write(base64.decodebytes((output[0].encode('UTF-8'))))