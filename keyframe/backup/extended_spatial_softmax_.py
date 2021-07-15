import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np

class ExtendedSpatialSoftmax(nn.Module):
  
    def __init__(self, height, width, channel, temperature=None, camera_intrinsic=[450, 0 , 320, 0, 450, 240, 0, 0, 1],  data_format='NCHW'):
        super(ExtendedSpatialSoftmax, self).__init__()
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

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        image_height = depth.shape[3]
        image_width = depth.shape[2]
        flattened_depth = depth.view(-1, image_height * image_width)
#        print('**************flattened depth', flattened_depth.shape)
        coord = expected_x * self.x - 1 + self.x + (expected_y * self.y + self.y - 1) * image_width
#        print(coord.shape, self.baseline.shape)
        coord = torch.round(coord).long()
        Z = torch.take(flattened_depth, coord)
        X = Z / self.f_x * image_width / 2 * expected_x
        Y = Z / self.f_y * image_height / 2 * expected_y
        xyz = torch.stack([X, Y, Z], 1).reshape((batch_size, 3, 1, self.channel))
        return xyz


class Transformation(nn.Module):
    
    def __init__(self):
        super(Transformation, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 1)
        self.conv2 = torch.nn.Conv2d(9, 3, 1)
        

    def forward(self, data):
        output = self.conv1(data)
        output = self.conv2(output)
        size = (output.shape[0], output.shape[1] * output.shape[3])
        return output.squeeze().transpose(2,1).reshape(size)
#        return self.conv2(output)
    

