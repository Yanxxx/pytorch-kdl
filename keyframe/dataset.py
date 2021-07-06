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


class dataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.mean = [0.485, 0.456, 0.406] 
        self.std = [0.229, 0.224, 0.225]
        self.image_folder = join(data_dir, 'color')
        self.depth_folder = join(data_dir, 'depth')
        self.info_folder = join(data_dir, 'info')
        self.action_folder = join(data_dir, 'action')
        self.image_files = listdir(self.image_folder)
        self.depth_files = listdir(self.depth_folder)
        self.info_files = listdir(self.info_folder)
        self.action_files = listdir(self.action_folder)
        self.length = len(self.image_files)
        with open(join(data_dir, 'frames'), 'rb') as f:
            self.frames = pickle.load(f)
        
    def __len__(self):
        return self.length
    
    def preProcess(self):
        print('start preprocessing dataset')
        for i in range(len(self.action_files)):
            filename = self.image_files[i]
            folder = splitext(filename)[0]
            print('process file', folder)
            path = join(self.data_dir, 'cache', folder)
            mkdir(path)
            self.processFrame(path, i)
    
    def processFrame(self, path, idx):
        images = self.loadfile(join(self.image_folder, self.image_files[idx]))
        depths = self.loadfile(join(self.depth_folder, self.depth_files[idx]))
        infos = self.loadfile(join(self.info_folder, self.info_files[idx]))
        actions = self.loadfile(join(self.action_folder, self.action_files[idx]))    
        
        for i in range(1, len(images)):
            if images[i] is None:
                continue
            if infos[i] is None:
                continue
            if actions[i] is None:
                continue
            if 'pose' not in actions[i].keys():
                continue
            data, depth = self.inputProcess(images[i, 0], depths[i, 0])
            gt = self.resolveInfo(infos, actions, i)
            file = join(path, str(i))
            torch.save({'data':data, 'depth':depth, 'gt':gt}, file)
#            pickle.dump(data, depth, gt, file)
#            print(type(data), type(depth), type(gt))
#            torch.save(data, depth, gt, file)
    
    def inputProcess(self, image, depth, dsize=(160, 120)):
        color = self.imageProcess(image, dsize)
        r_depth, depth = self.depthProcess(depth, dsize)    
        data = torch.cat((color, r_depth), 2)        
        data = data.permute(2, 0 ,1)
        data = torch.reshape(data, (4, 120, 160))
        return data, depth
        
    def imageProcess(self, image, size=(160, 120)):        
        img = cv2.resize(image, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        img = img - self.mean
        img = img / self.std
        img = torch.Tensor(img)        
        return img
    
    def depthProcess(self, depth, size=(160, 120)):
        resized_depth = cv2.resize(depth, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
        resized_depth = torch.reshape(torch.Tensor(resized_depth), (120,160,1))
        depth = torch.Tensor(depth).reshape((1,480,640))
        return resized_depth, depth
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        folder = splitext(filename)[0]
        files = listdir(join(self.data_dir, 'cache', folder))
        r.seed(datetime.now())
        selected_frame = r.randint(1, len(files) - 1)
#        if len(files) <= selected_frame:
#            print(len(files), selected_frame)
        d = torch.load(join(self.data_dir, 'cache', folder, files[selected_frame]))
        return d['data'], d['depth'], d['gt']

        
    def __getitem_backup__(self, idx):
        # image
        data = self.loadfile(join(self.image_folder, self.image_files[idx]))
        r.seed(datetime.now())
        selected_frame = r.randint(1, data.shape[0] - 2)        
#        print(self.image_files[idx], selected_frame)
        img = data[selected_frame,0]
        img = cv2.resize(img, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
        img = img / 255.0
        img = img - self.mean
        img = img / self.std
        
        # depth
        data = self.loadfile(join(self.depth_folder, self.depth_files[idx]))            
        depth = data[selected_frame,0]
        td = cv2.resize(depth, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)
        depth = torch.Tensor(depth).reshape((1,480,640))
        color = torch.Tensor(img)
        td = torch.Tensor(td)
        td = torch.reshape(td, (120,160,1))
        data = torch.cat((color, td), 2)        
        data = data.permute(2, 0 ,1)
        data = torch.reshape(data, (4, 120, 160))
        
        # info
        info = self.loadfile(join(self.info_folder, self.info_files[idx]))
        #action 
        action = self.loadfile(join(self.action_folder, self.action_files[idx]))        
        gt = self.resolveInfo(info, action, selected_frame)
        
        return data, depth, gt
    
    def loadfile(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    def resolveInfo(self, info, action, selected_frame):
        x = np.concatenate((np.array(info[selected_frame][5][0]), 
             np.array(info[selected_frame][5][1])),axis=0)
        y = np.concatenate((np.array(info[selected_frame][6][0]), 
             np.array(info[selected_frame][6][1])), axis=0)
        z = np.concatenate((np.array(action[selected_frame]['pose'][0]), 
             np.array(action[selected_frame]['pose'][1])),axis=0)
        
        gt = np.concatenate((x,y,z),axis=0)
        gt = torch.Tensor(gt)
#        gt = torch.Tensor(gt).reshape([1,gt.shape[0]])
        return gt
        
#mean = [0.485, 0.456, 0.406] 
#std = [0.229, 0.224, 0.225]
#
#data_path = '../datasets/key_frame_identifier/block-insertion-test/'
#img_folder = data_path + 'color/'
#depth_folder = data_path + 'depth/'
#pcl_file = '000000-1'
