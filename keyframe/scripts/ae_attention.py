# coding=utf-8
# Copyright 2021 The Yan Li, UTK, Knoxville, TN.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from submodules import NaiveCNN, PoseRegress, SpatialSoftmax
#from extended_spatial_softmax_nn_transform import ExtendedSpatialSoftmax


#class Attention(nn.Module):
#    def __init__(self, rotation, translation, 
#                 image_height,
#                 image_width,
#                 camera_intrinsic=[450, 0 , 320, 0, 450, 240, 0, 0, 1],
#                 spatial_height=31, 
#                 spatial_weight=21, spatial_channel=16):
#        super(Attention, self).__init__()
#        self.feature = NaiveCNN()
##        self.extended_spatial_max = ExtendedSpatialSoftargMax(31,21,196)
##        self.extended_spatial_max = ExtendedSpatialSoftargMax(spatial_height, 
##                                                              spatial_weight, 
##                                                              spatial_channel)
##        self.transform = Transformation(rotation, translation)
#        self.extended_spatial_softmax = ExtendedSpatialSoftmax(spatial_height,
#                                                               spatial_weight,
#                                                               spatial_channel,
#                                                               image_height,
#                                                               image_width,
#                                                               rotation, 
#                                                               translation, 
#                                                               camera_intrinsic)
#        self.pose_regress = PoseRegress()
#        self.rotation = rotation
#        self.translation = translation
#              
#    def forward(self, data, depth):
#        output = self.feature(data)
##        print(output.shape)
#        output = self.extended_spatial_softmax(output, depth)
##        torch.save(output, 'coords')
##        print('extended spatial soft(arg)max output: ', output.shape)
##        output = self.transform(output, batch_size, channel)
##        print(output.shape)
#        output = self.pose_regress(output)
##        print(output.shape)
#        return output
    
    


class Attention2D(nn.Module):
    def __init__(self, spatial_height=31, spatial_weight=21, spatial_channel=96):
        super(Attention2D, self).__init__()
        self.feature = NaiveCNN(input_channel=3, output_channel=96)
        self.extended_spatial_softmax = SpatialSoftmax(spatial_height,
                                                       spatial_weight,
                                                       spatial_channel)
        self.pose_regress = PoseRegress(192, 21)
              
    def forward(self, data):
        output = self.feature(data)
        output = self.extended_spatial_softmax(output)
        output = self.pose_regress(output)
        return output

from torchvision import models


class NaiveAttention(nn.Module):

    def __init__(self):
        super(NaiveAttention, self).__init__()
        self.feature = NaiveCNN(input_channel=4, output_channel=16)
        self.pose_regress = PoseRegress(10416, 21)
              
    def forward(self, data):
        data = self.feature(data)
        data = data.view(-1, data.shape[1] * data.shape[2] * data.shape[3])
        output = self.pose_regress(data)
        return output


class NaiveAttentionResnet(nn.Module):
    
    def __init__(self):
        super(NaiveAttentionResnet, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        modules = list(resnet18.children())
        conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2), 
                          padding=(3,3),bias=False)
        self.feature = nn.Sequential(conv1, *modules[1:7])
        self.l1 = PoseRegress(128*160, 21)
        
    def forward(self, data):
        data = self.feature(data)
        data = data.view(-1, data.shape[1] * data.shape[2] * data.shape[3])
        output = self.l1(data)
        return output
        
