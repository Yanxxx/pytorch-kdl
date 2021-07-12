import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

class ExtendedSpatialSoftargMax(nn.Module):
  
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(ExtendedSpatialSoftargMax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature, depth):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
#         expected_xy = torch.cat([expected_x, expected_y], 1)
#         feature_keypoints = expected_xy.view(-1, self.channel*2)
        
        image_height = depth.shape[3]
        image_weight = depth.shape[2]
        
        ix = torch.round(expected_x * image_weight).long()
        iy = torch.round(expected_y * image_height).long()
        
        z = depth[:, 0, ix, iy] 
        z_prime_x = z / 900 * image_weight
        z_prime_y = z / 900 * image_height
        
        result = torch.cat([expected_x * z_prime_x, expected_y * z_prime_y, z], 1)
        feature_keypoints = result.view(-1, self.channel*3)

        return feature_keypoints
