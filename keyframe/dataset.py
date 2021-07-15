#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:22:09 2021

@author: yan
"""

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

from datetime import datetime
import torch
import cv2 # for resize image
from os import listdir, mkdir
from os.path import join, splitext
import pickle
from torch.utils.data import Dataset
import random as r
import numpy as np
import math


class dataset(Dataset):
    def __init__(self, data_dir, set_range=[0, 0.75], preprocess=False):
        self.set_range = set_range
        self.data_dir = data_dir
        self.mean = [0.485, 0.456, 0.406] 
        self.std = [0.229, 0.224, 0.225]
        self.image_folder = join(data_dir, 'color')
        self.depth_folder = join(data_dir, 'depth')
        self.ee_folder = join(data_dir, 'ee')
        self.object_folder = join(data_dir, 'object_pos')
        self.targ_folder = join(data_dir, 'targ_pos')
        
        self.image_files = listdir(self.image_folder)
        self.depth_files = listdir(self.depth_folder)
        self.ee_files = listdir(self.ee_folder)
        self.object_files = listdir(self.object_folder)
        self.targ_files = listdir(self.targ_folder)
        
        self.length = int(len(self.image_files) * \
			(self.set_range[1] - self.set_range[0]))
        if not preprocess:
            self.loaddata()
        else:
            self.pre_process()
#        with open(join(data_dir, 'frames'), 'rb') as f:
#            self.frames = pickle.load(f)
    def loaddata(self):
        folder = join(self.data_dir, 'data')
        files = listdir(folder)
        self.colors = []
        self.depths = []
        self.targets = []
        self.objects = []
        self.robot_ees = []
        start = int(len(files) * self.set_range[0])
        end = int(len(files) * self.set_range[1])
        
        for count in range(start, end):
            filename = files[count]
            filename = join(self.data_dir, 'data', filename)
            print('loading file ', filename)
            with open(filename, 'rb') as f:
                data = torch.load(f)
            c = data['data']
            d = data['depth']
            t = data['targets']
            o = data['objects']
            e = data['robot_ee']
            for i in range(len(c)):
                self.colors.append(c[i])
                self.depths.append(d[i])
                self.targets.append(t[i])
                self.objects.append(o[i])
                self.robot_ees.append(e[i])
        
    def __len__(self):
        return self.length
    
    def pre_process(self):
        print('start preprocessing dataset')
        colors = []
        depths = []
        targets = []
        objects = []
        robot_ees = []
        
        for count, filename in enumerate(self.image_files):
            print('processing file', filename)
#            path = join(self.data_dir, 'cache', folder)
#            mkdir(path)
            data, depth, t, o , e = self.process_frame(filename)
            colors.append(data)
            depths.append(depth)
            targets.append(t)
            objects.append(o)
            robot_ees.append(e)
            if count % 100 == 99:
                fn = f'batch-{count//100}'
                fn = join(self.data_dir, 'data', fn)
                with open(fn, 'wb') as f:
                    torch.save({'data':colors, 'depth':depths, \
                                'targets':targets, 'objects':objects, \
                                'robot_ee':robot_ees}, f)
                colors = []
                depths = []
                targets = []
                objects = []
                robot_ees = []
    
    def process_frame(self, filename):
        images = self.loadfile(join(self.image_folder, filename))
        depths = self.loadfile(join(self.depth_folder, filename))
        ee = self.loadfile(join(self.ee_folder, filename))
        object_pos = self.loadfile(join(self.object_folder, filename))
        target_pos = self.loadfile(join(self.targ_folder, filename))
        color, depth = self.input_process(images[0,0], depths[0,0])
        t = target_pos[0][6]
        o = object_pos[0]
        e = ee[0]['pose']
        # print(o)
        o = np.concatenate((np.array(o[0]), np.array(o[1])), axis=0)
        t = np.concatenate((np.array(t[0]), np.array(t[1])), axis=0)
        e = np.concatenate((np.array(e[0]), np.array(e[1])), axis=0)
        t = torch.Tensor(t)
        o = torch.Tensor(o)
        e = torch.Tensor(e)
        
        return color, depth, t, o ,e
    
    def input_process(self, image, depth, dsize=(160, 120)):
        color = self.image_process(image, dsize)
        r_depth, depth = self.depth_process(depth, dsize)    
        data = torch.cat((color, r_depth), 2)        
        data = data.permute(2, 0 ,1)
        data = torch.reshape(data, (4, 120, 160))
        return data, depth
        
    def image_process(self, image, size=(160, 120)):        
        img = cv2.resize(image, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        img = img - self.mean
        img = img / self.std
        img = torch.Tensor(img)        
        return img
    
    def depth_process(self, depth, size=(160, 120)):
        resized_depth = cv2.resize(depth, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
        resized_depth = torch.reshape(torch.Tensor(resized_depth), (120,160,1))
        depth = torch.Tensor(depth).reshape((1,480,640))
        return resized_depth, depth
    
    def __getitem__(self, idx):
        c = self.colors[idx]
        d = self.depths[idx]
        t = self.targets[idx]
        o = self.objects[idx]
        e = self.robot_ees[idx]
        gt = torch.squeeze(torch.cat((t[None,:], o[None,:] , e[None,:]), 1))
#        print(gt.shape)
        return c, d, gt

    def loadfile(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
        
#mean = [0.485, 0.456, 0.406] 
#std = [0.229, 0.224, 0.225]
#
#data_path = '../datasets/key_frame_identifier/block-insertion-test/'
#img_folder = data_path + 'color/'
#depth_folder = data_path + 'depth/'
#pcl_file = '000000-1'

