import torch
import torch.nn.functional as F
import torch.nn as nn
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
        self.camera_intrinsic = torch.Tensor([450, 0 , 320, 0, 450, 240, 0, 0, 1])
        self.x = self.camera_intrinsic[2]
        self.y = self.camera_intrinsic[5]
        
    def rcbaseline(self, batch_size):
        baseline = torch.arange(0,batch_size) * 640 *480
        baseline = torch.transpose(baseline.repeat(196,1),1,0)
        self.baseline = baseline.reshape(batch_size * 196, 1).to('cuda:0')
        

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
#        print('coord', expected_x.shape, self.pos_x.shape)
#        expected_xy = torch.cat([expected_x, expected_y], 1)
#        print('spatial softmax', expected_xy.shape)
#        feature_keypoints = expected_xy.view(-1, self.channel*2)
        image_height = depth.shape[3]
        image_width = depth.shape[2]
        flattened_depth = depth.view(-1, image_height * image_width*depth.shape[0])
#        print('**************flattened depth', flattened_depth.shape)
        self.rcbaseline(batch_size)
        coord = expected_x * self.x - 1 + self.x + (expected_y * self.y + self.y - 1) * image_width
#        print(coord.shape, self.baseline.shape)
        coord = torch.round(coord + self.baseline).long()
        z = torch.take(flattened_depth, coord)
#        ix = torch.round(expected_x * image_width).long()
#        iy = torch.round(expected_y * image_height).long()
        
#        z = depth[:, 0, ix, iy] 
#        print('****************** depth dim',z.shape)
#        z = z.view(-1, self.height*self.width)
#        print('****************** depth dim',z.shape)
        z_prime_x = z / 900 * image_width
        z_prime_y = z / 900 * image_height
        
        
        result = torch.cat([expected_x * z_prime_x, expected_y * z_prime_y, z], 1)
#        print('extended', result.shape)
        feature_keypoints = result.view(-1, self.channel*3)

        return feature_keypoints
