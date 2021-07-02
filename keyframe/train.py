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
from dataset import dataset
from keyframe import Keyframe
import utils
import datetime
import torch
import torch.nn as nn


def main():
    device = utils.selectDevice()
    
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 1}
    
    rot, t = utils.camTrans()
    
    model = Keyframe(rot, t).to(device)
    
    data_path = '/workspace/datasets/key_frame_identifier/block-insertion-test/'
    
    training_set = dataset(data_path)
    train_generator = torch.utils.data.DataLoader(training_set, **params)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    max_epochs = 200000
    
    for epoch in range(max_epochs):
#        name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'
        
        curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = os.path.join(data_path, 'logs', curr_time, 'train')
        
        for data, depth, gt in train_generator:
            ps = model(data, depth)
            loss = criterion(ps, gt)
            if t % 100 == 99:
                print(t, loss.item())
            
            optimizer.zero_grad()

            loss.backward()
            
            optimizer.step()
            
            
            
            
            
            

    
    

    
