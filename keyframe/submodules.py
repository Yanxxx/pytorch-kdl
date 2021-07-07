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

    
class FeatureNet(nn.Sequential):
    def __init__(self):
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
          nn.Conv2d(256, 16, 1),
          nn.BatchNorm2d(16),
          nn.ReLU(),     
        )
    
    def forward(self, data):
        return self.model(data)
    
    
class PoseRegression(nn.Sequential):
  
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(48, 96),
          nn.BatchNorm1d(96),
          nn.Linear(96, 48),
          nn.BatchNorm1d(48),
          nn.ReLU(),
#          nn.Linear(42, 21),
#          nn.BatchNorm1d(21),
          nn.Linear(48, 7),
          nn.BatchNorm1d(7),
          nn.Tanh()
        )
    
    def forward(self, data):
        return self.model(data)


class Transformation(nn.Module):
    
    def __init__(self, rotation, translation):
        super().__init__()
        self.rotation = rotation
        self.translation = translation
        self.br = torch.transpose(rotation, 1, 0)
        self.bt = torch.matmul(self.br, translation)

    def forward(self, data, batch_size, channel):
#        print('Transformation network input: ', data.shape)
        reprojected_pt = torch.matmul(self.rotation, data) + self.translation#[:,None]  
        
        r = torch.reshape(torch.transpose(reprojected_pt, 1, 0), (1, channel * batch_size * 3))
        r = torch.reshape(r, (channel* 3, batch_size))
        reprojected_pt = torch.transpose(r, 1, 0)
        
        return reprojected_pt
    
    def backward(self, data):
        size = list(data.shape)
        data = data.reshape(size[0], 3, size[1] // 3)
        data = data.permute(1, 2, 0)
        reprojected_pt = torch.matmul(self.br,torch.transpose(data,1,0)) + self.bt#[:,None]
        reprojected_pt = reprojected_pt.permute(2, 0, 1)
        reprojected_pt = reprojected_pt.reshape(size[0], size[1])
        
        return reprojected_pt
        

