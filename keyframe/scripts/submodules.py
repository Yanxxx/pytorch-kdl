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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

    
class NaiveCNN(nn.Sequential):
    
    def __init__(self, input_channel=4, output_channel=32):
        super().__init__()
        self.model = nn.Sequential(
          nn.Conv2d(input_channel, 32, 3),
          nn.Conv2d(32, 32, 3),
          nn.Conv2d(32, 32, 3),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(32, 64, 3, stride=2),
          nn.Conv2d(64, 128, 3),
          nn.Conv2d(128, 128, 3),
          nn.BatchNorm2d(128),
          nn.ReLU(),    
          nn.Conv2d(128, 256, 3, stride=2),
          nn.Conv2d(256, 256, 3),
          nn.Conv2d(256, 256, 3),
          nn.BatchNorm2d(256),
          nn.ReLU(),    
          nn.Conv2d(256, output_channel, 1),
          nn.BatchNorm2d(output_channel),
          nn.ReLU(),     
        )
    
    def forward(self, data):
        return self.model(data)
    
class NaiveCNND(nn.Sequential):
    
    def __init__(self, input_channel=4, output_channel=32):
        super().__init__()
        self.model = nn.Sequential(
          nn.Conv2d(4, 32, 3),
          nn.Conv2d(32, 32, 3),
          nn.Conv2d(32, 32, 3),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(32, 64, 3, stride=2),
          nn.Conv2d(64, 128, 3),
          nn.Conv2d(128, 128, 3),
          nn.BatchNorm2d(128),
          nn.ReLU(),    
          nn.Conv2d(128, 256, 3, stride=2),
          nn.Conv2d(256, 256, 3),
          nn.Conv2d(256, 256, 3),
          nn.BatchNorm2d(256),
          nn.ReLU(),    
          nn.Conv2d(256, output_channel, 1),
          nn.BatchNorm2d(output_channel),
          nn.ReLU(),     
        )
    
    def forward(self, data):
        return self.model(data)

class PoseRegress(nn.Sequential):
    
    def __init__(self, input_neurons=96, output_neurons=7):
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(input_neurons, 96),
          nn.BatchNorm1d(96),
          nn.Linear(96, 48),
          nn.BatchNorm1d(48),
          nn.ReLU(),
#          nn.Linear(42, 21),
#          nn.BatchNorm1d(21),
          nn.Linear(48, output_neurons),
          nn.Tanh()
                )
    def forward(self, data):
        return self.model(data)


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
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints
        

class PoseRegressionNaive(nn.Sequential):
  
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(10416, 128),
          nn.BatchNorm1d(128),
          nn.Linear(128, 48),
          nn.BatchNorm1d(48),
          nn.ReLU(),
#          nn.Linear(42, 21),
#          nn.BatchNorm1d(21),
          nn.Linear(48, 21),
          nn.Tanh()
        )
    
    def forward(self, data):
        data = data.view(-1, data.shape[1] * data.shape[2] * data.shape[3])
        return self.model(data)
 
class CoordinateTransformer(nn.Module):
    
    def __init__(self, height, width):
        super(CoordinateTransformer).__init__()
        
        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.Tensor(pos_x).float()
        pos_y = torch.Tensor(pos_y).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.xy = torch.cat((pos_x[None,:], pos_y[None, :]), 0)
        self.ones = torch.ones(height, width)
        self.l1 = nn.Conv2d(2, 2, 1)
        self.l2 = nn.Conv2d(3, 3, 1)
        
    def forward(self, depth):
        output = self.l1(self.xy)
        output = torch.cat((output, self.ones[None, :]), 0)
        output = output * depth
        output = self.l2(output)
        return output
        