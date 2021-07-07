import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

class ExtendedSpatialSoftargMax(nn.Module):
  
    def __init__(self, height, width, channel, temperature=None, camera_intrinsic=[450, 0 , 320, 0, 450, 240, 0, 0, 1],  data_format='NCHW'):
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
        self.camera_intrinsic = torch.Tensor(camera_intrinsic)
        self.x = self.camera_intrinsic[2]
        self.y = self.camera_intrinsic[5]
        self.f_x = self.camera_intrinsic[0]
        self.f_y = self.camera_intrinsic[4]

    def forward(self, feature, depth):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        # print(feature.shape)
        batch_size = feature.shape[0]
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

#        print('**************flattened feature', feature.shape)
        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        r_expected_x = torch.reshape(expected_x, (1, batch_size * self.channel))
        r_expected_y = torch.reshape(expected_y, (1, batch_size * self.channel))
#        print('coord', expected_x.shape, self.pos_x.shape)
#        expected_xy = torch.cat([expected_x, expected_y], 1)
#        print('spatial softmax', expected_xy.shape)
#        feature_keypoints = expected_xy.view(-1, self.channel*2)
        image_height = depth.shape[3]
        image_width = depth.shape[2]
        flattened_depth = depth.view(-1, image_height * image_width)
#        print('**************flattened depth', flattened_depth.shape)
        coord = expected_x * self.x - 1 + self.x + (expected_y * self.y + self.y - 1) * image_width
#        print(coord.shape, self.baseline.shape)
        coord = torch.reshape(torch.round(coord).long(), (batch_size, self.channel))
        Z = torch.take(flattened_depth, coord)
        Z = torch.reshape(Z, (1, batch_size * self.channel))
#        ix = torch.round(expected_x * image_width).long()
#        iy = torch.round(expected_y * image_height).long()
        
#        z = depth[:, 0, ix, iy] 
#        print('****************** depth dim',z.shape)
#        z = z.view(-1, self.height*self.width)
#        print('****************** depth dim',z.shape)
        X = Z / self.f_x * image_width / 2 * r_expected_x
        Y = Z / self.f_y * image_height / 2 * r_expected_y
        
#        result = torch.cat([X, Y, Z], 0)
#        r = torch.reshape(torch.transpose(result, 1, 0), (1, X.shape[1] * 3))
#        r = torch.reshape(r, (self.channel * 3, batch_size))
#        feature_keypoints = torch.transpose(r, 1, 0)
        
        feature_keypoints = torch.cat([X, Y, Z], 0)
#        print('extended', result.shape)
#        feature_keypoints = result.view(-1, self.channel*3)

        return feature_keypoints
