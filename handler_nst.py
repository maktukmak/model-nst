
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import map_class_to_label
from captum.attr import IntegratedGradients
from PIL import Image
import base64
import io
import torch
from torchvision import transforms
import torch.nn.functional as F
import struct

class handler_nst(BaseHandler):

    """
    ImageClassifier handler class. This handler takes an image
    and returns the name of object in that image.
    """
    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None

        self.topk = 5

        size = 512
        crop = False

        transform_list = []
        if size != 0:
            transform_list.append(transforms.Resize(size))
        if crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.append(transforms.ToTensor())
        self.image_processing = transforms.Compose(transform_list)

    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor

        Args:
            data (List): Input data from the request is in the form of a Tensor

        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        print('Preprocessing Started')
        img_content = self.image_processing(Image.open(io.BytesIO(data[0].get("img_content")))).unsqueeze(0).to(self.device)

        print(img_content.shape)

        img_style = self.image_processing(Image.open(io.BytesIO(data[0].get("img_style")))).unsqueeze(0).to(self.device)

        print(img_style.shape)

        alpha = torch.tensor(struct.unpack('f', data[0].get("alpha"))).to(self.device)
        print(alpha)
        print('Preprocessing Done')

        return img_content, img_style, alpha
    
    def inference(self, *args, **kwargs):
        print('Inference Started')
        args = args[0]
        with torch.no_grad():
            results = self.model(*args, **kwargs)

        print('Inference Done')
        return results

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
    

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def postprocess(self, data):

        print('Postprocessing Started')
        transform = transforms.ToPILImage()
        image = transform(data.squeeze(dim=0))

        imb = io.BytesIO()
        image.save(imb, format='PNG')
        print('Postprocessing Done')

        output = [base64.b64encode(imb.getvalue()).decode("utf-8")]

        return output