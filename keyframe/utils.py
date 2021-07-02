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
import math
import numpy as np

    
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
          nn.Conv2d(256, 196, 1),
          nn.BatchNorm2d(196),
          nn.ReLU(),     
        )
    
    def forward(self, data):
        return self.model(data)
    
    
class PoseRegression(nn.Sequential):
  
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(588, 294),
          nn.BatchNorm1d(294),
          nn.Linear(294, 147),
          nn.BatchNorm1d(147),
          nn.ReLU(),
          nn.Linear(147, 42),
          nn.BatchNorm1d(42),
          nn.ReLU(),
          nn.Linear(42, 21),
          nn.BatchNorm1d(21),
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
        
    def forward(self, data):
        reprojected_pt = torch.matmul(self.rotation,data) + self.translation#[:,None]
        return reprojected_pt
    
    def backward(self, data):
        reprojected_pt = torch.matmul(self.br,data) + self.bt#[:,None]
        return reprojected_pt
        


def selectDevice():        
    torch.cuda.is_available()    
    if torch.cuda.is_available():  
      dev = "cuda:0" 
    else:  
      dev = "cpu"      
    return torch.device(dev)  


def euler2rotm(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def camTrans():
    rotation = torch.Tensor(euler2rotm([np.pi / 4, np.pi, -np.pi / 2]))
    translation = torch.Tensor(np.array([1.0, 0, 0.75]).reshape((3,1)))
    
    return rotation, translation