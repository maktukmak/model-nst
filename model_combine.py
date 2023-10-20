import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import sys
sys.path.insert(0, './pytorch-AdaIN')

import net
from function import adaptive_instance_normalization, coral


decoder_pth = './pytorch-AdaIN/models/decoder.pth'
vgg_pth = './pytorch-AdaIN/models/vgg_normalised.pth'

class NST(torch.nn.Module):
    def __init__(self):
        super(NST, self).__init__()
        self.decoder = net.decoder
        self.vgg = net.vgg

        self.decoder.load_state_dict(torch.load(decoder_pth))
        self.vgg.load_state_dict(torch.load(vgg_pth))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

    def forward(self, content, style, alpha=1.0, interpolation_weights=None):

        assert (0.0 <= alpha <= 1.0)
        content_f = self.vgg(content)
        style_f = self.vgg(style)
        if interpolation_weights:
            _, C, H, W = content_f.size()
            feat = torch.FloatTensor(1, C, H, W).zero_()
            base_feat = adaptive_instance_normalization(content_f, style_f)
            for i, w in enumerate(interpolation_weights):
                feat = feat + w * base_feat[i:i + 1]
            content_f = content_f[0:1]
        else:
            feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return self.decoder(feat)


model = NST()
model.eval()
content = torch.randn((1,3,512,729))
style = torch.randn((1,3,693,512))
alpha = torch.tensor(1.0)

model(content, style, alpha)
traced_script_model = torch.jit.trace(model, example_inputs=(content, style, alpha))
traced_script_model.save('nst.pt')