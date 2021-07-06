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
from submodules import FeatureNet, Transformation, PoseRegression
from extended_spatial_softmax import ExtendedSpatialSoftargMax



class Keyframe(nn.Module):
    def __init__(self, rotation, translation, spatial_height=31, 
                 spatial_weight=21, spatial_channel=196):
        super(Keyframe, self).__init__()
        self.feature = FeatureNet()
#        self.extended_spatial_max = ExtendedSpatialSoftargMax(31,21,196)
        self.extended_spatial_max = ExtendedSpatialSoftargMax(spatial_height, 
                                                              spatial_weight, 
                                                              spatial_channel)
        self.transform = Transformation(rotation, translation)
        self.pose_regress = PoseRegression()
        self.rotation = rotation
        self.translation = translation
              
    def forward(self, data, depth):
        output = self.feature(data)
#        print(output.shape)
        output = self.extended_spatial_max(output, depth)
#        print(output.shape)
        output = self.transform(output)
#        print(output.shape)
        output = self.pose_regress(output)
#        print(output.shape)
        return output
    
    