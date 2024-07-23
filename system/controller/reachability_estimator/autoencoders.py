import torch
import torch.nn as nn
import math

from .types import transpose_image, untranspose_image
from system.types import Image

class ImageEncoder(nn.Module):
    def __init__(self, code_dim = 16, fc_2_dim=None, bias=True):
        super().__init__()
        self.code_dim = code_dim
        if fc_2_dim is None:
            fc_2_dim = int(math.sqrt(400 * code_dim)) # geometric average
        self.layers = nn.Sequential(
            # 4 x 64 x 64
            nn.Conv2d(4, 8, kernel_size=5, stride=2, bias=bias),
            nn.ReLU(),
            # 8 x 30 x 30
            nn.Conv2d(8, 12, kernel_size=5, stride=2, bias=bias),
            nn.ReLU(),
            # 12 x 13 x 13
            nn.Conv2d(12, 16, kernel_size=5, stride=2, bias=bias),
            nn.ReLU(),
            nn.Flatten(),
            # 16 x 5 x 5
            nn.Linear(16 * 5 * 5, fc_2_dim),
            nn.ReLU(),
            nn.Linear(fc_2_dim, code_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = transpose_image(x)
        return self.layers(x)


class ImageDecoder(nn.Module):
    def __init__(self, code_dim = 16, fc_2_dim=None, bias=True):
        super().__init__()
        if fc_2_dim is None:
            fc_2_dim = int(math.sqrt(400 * code_dim)) # geometric average
        self.layers = nn.Sequential(
            nn.Linear(code_dim, fc_2_dim),
            nn.ReLU(),
            nn.Linear(fc_2_dim, 400),
            nn.ReLU(),
            nn.Unflatten(1, (16, 5, 5)),
            nn.ConvTranspose2d(16, 12, kernel_size=5, stride=2, bias=bias),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 8, kernel_size=5, stride=2, bias=bias, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=5, stride=2, bias=bias, output_padding=1),
            nn.ReLU(),
        )
    def forward(self, x) -> Image:
        x = self.layers.forward(x)
        return untranspose_image(x)

class ImageAutoencoder(nn.Module):
    def __init__(self, code_dim=16, fc_2_dim=None, bias=True, optimizer_params={}):
        super().__init__()
        self.encoder = ImageEncoder(code_dim, fc_2_dim, bias)
        self.decoder = ImageDecoder(code_dim, fc_2_dim, bias)

        optimizer_params = dict(lr=5e-3, weight_decay=0) | optimizer_params

        self.optimizer = torch.optim.Adam(self.parameters(), **optimizer_params)

    def forward(self, x: Image) -> Image:
        code = self.encoder.forward(x)
        decode = self.decoder.forward(code)
        return decode

    def save(self, metadata, model_file):
        """ save current state of the model """
        state = {
            'metadata': metadata,
            'optimizer': self.optimizer.state_dict(),
            'nets': {
                'encoder': self.encoder.state_dict(),
                'decoder': self.decoder.state_dict(),
            }
        }
        torch.save(state, model_file)

    def load_state_dict(self, state):
        self.encoder.load_state_dict(state['nets']['encoder'])
        self.decoder.load_state_dict(state['nets']['decoder'])
        self.optimizer.load_state_dict(state['optimizer'])
