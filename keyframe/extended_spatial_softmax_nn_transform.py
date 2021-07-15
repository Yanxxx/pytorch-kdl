import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import tmp_utils as utils


device = utils.selectDevice()

class ExtendedSpatialSoftmax(nn.Module):

    def __init__(self, height, width, channel, image_height, image_width, 
                 rotation, translation, 
                 camera_intrinsic=[450, 0 , 320, 0, 450, 240, 0, 0, 1],
                 temperature=None, data_format='NCHW'):
        super(ExtendedSpatialSoftmax, self).__init__()
        self.channel = channel
        self.camera_intrinsic = torch.Tensor(camera_intrinsic).to(device)
        self.rotation = rotation
        self.translation = translation
        self.br = torch.transpose(rotation, 1, 0)
        self.bt = - torch.matmul(rotation, translation)
        
        self.spatial_softmax = SpatialSoftmax(height, width, channel)
        self.spatial_projection = SpatialProjection(image_height, image_width, 
                                                    channel, camera_intrinsic)
        self.transform = Transformation()
        
    def forward(self, data, depth):
        output = self.spatial_softmax(data)
#        print(output.shape)
        output = self.spatial_projection.apply(output, depth, 
                                               self.channel, 
                                               self.camera_intrinsic)
        output = self.transform(output)
        return output

class SpatialSoftmax(nn.Module):

    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
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

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)

        return expected_xy
    
class SpatialProjection(torch.autograd.Function):
  
    @staticmethod
    def forward(ctx, xy, depth, channel, camera_intrinsic):
        x = camera_intrinsic[2]
        y = camera_intrinsic[5]
        f_x = camera_intrinsic[0]
        f_y = camera_intrinsic[4]
        batch_size = depth.shape[0]
        rx = torch.reshape(xy[:, 0], (batch_size, channel))
        ry = torch.reshape(xy[:, 1], (batch_size, channel))
        size = channel * batch_size
        image_height = depth.shape[3]
        image_width = depth.shape[2]
        flattened_depth = depth.view(-1, image_height * image_width)
        coord = rx * x - 1 + x + (ry * y + y - 1) * image_width
        coord = torch.round(coord).long()
        Z = torch.squeeze(torch.take(flattened_depth, coord))
        X = Z / f_x * image_width / 2 * rx
        Y = Z / f_y * image_height / 2 * ry
        xyz = torch.stack([X, Y, Z], 1).reshape((batch_size, 3, 1, channel))
        imsize = torch.Tensor([image_height, image_width, size]).to(device)
        ctx.save_for_backward(xyz, xy, camera_intrinsic, imsize)
        return xyz
#        Z = torch.squeeze(torch.reshape(Z, (channel * batch_size, 1)))
##        print(Z.shape, xy.shape, xy[:, 0].shape)
#        X = Z / f_x * image_width / 2 * xy[:, 0]
#        Y = Z / f_y * image_height / 2 * xy[:, 1]
#        imsize = torch.Tensor([image_height, image_width]).to(device)
##        print(X.shape, Y.shape, Z.shape)
#        feature_keypoints = torch.cat([X[None, :], Y[None, :], Z[None, :]], 0)
##        print(feature_keypoints.shape)
#        
#        return feature_keypoints

    @staticmethod
    def backward(ctx, grad_output):
        xyz, xy, camera_intrinsic, imsize = ctx.saved_tensors
        image_height, image_width, size = imsize
#        print(imsize)
        x = camera_intrinsic[2]
        y = camera_intrinsic[5]
        f_x = camera_intrinsic[0]
        f_y = camera_intrinsic[4]
        delta_xyz = grad_output.clone()
        target = xyz + delta_xyz
#        print(size)
        target = torch.squeeze(target).permute(1,2,0).transpose(2,1).reshape(3, int(size))
        x = target[0, :]
        y = target[1, :]
        z = target[2, :]
        cx = f_x * x * 2 / z / image_width
        cy = f_y * y * 2 / z / image_height
        new_xy = torch.cat([cx[None,:], cy[None,:]], 0)
        grad_input = torch.transpose(new_xy, 1, 0) - xy
        return grad_input, None, None, None
    

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
    
    
        
