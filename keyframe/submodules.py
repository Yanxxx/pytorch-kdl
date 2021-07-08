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


