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
from submodules import FeatureNet2D, PoseRegression2D, SpatialSoftmax



class PointAttension2D(nn.Module):
    def __init__(self, spatial_height=31, spatial_weight=21, spatial_channel=16):
        super(PointAttension2D, self).__init__()
        self.feature = FeatureNet2D()
        self.extended_spatial_softmax = SpatialSoftmax(spatial_height,
                                                       spatial_weight,
                                                       spatial_channel)
        self.pose_regress = PoseRegression2D()
              
    def forward(self, data):
        output = self.feature(data)
        output = self.extended_spatial_softmax(output)
        output = self.pose_regress(output)
        return output
    
    