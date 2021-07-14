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
#from keyframe import Keyframe
from attention_autograd import Attention
import tmp_utils as utils
import datetime
import torch
import torch.nn as nn
from os import mkdir, getcwd
from os.path import join


def main():
    pre_process = False
    device = utils.selectDevice()
    
    rot, t = utils.camTrans()
    
    model = Attention(rot.to(device), t.to(device), 480, 640).to(device)    
    # preparing dataset
    print('preparing dataset')
    data_path = '/workspace/datasets/key_frame_identifier/block-insertion-test/'    
    training_set = dataset(data_path)
    if pre_process:
        training_set.preProcess()
        return
        
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 16}    
    train_generator = torch.utils.data.DataLoader(training_set, **params)
    # setup loss fucntion
    print('setting up loss function')
    criterion = nn.MSELoss()
    # setup optimizer 
    print('setup optimizer')
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000,3000, 6000], gamma=0.1)

    # maximum epochs
    max_epochs = 200000
#    max_epochs = 0
    epoch = 0
    checkpoint = { 
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_sched': scheduler}

    loss_val = []
    # start the training
    print('starting the training process')
    curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_path = join(getcwd(), 'checkpoints', curr_time)
    mkdir(model_path)
    loss_path = join(getcwd(), 'loss', curr_time)
    mkdir(loss_path)
    for epoch in range(max_epochs):
#        name = f'{FLAGS.task}-{FLAGS.agent}-{FLAGS.n_demos}-{train_run}'
        
#        curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
#        log_dir = os.path.join(data_path, 'logs', curr_time, 'train')
#
        
        for data, depth, gt in train_generator:
#            print(data.shape, depth.shape, gt.shape)
            data = data.to(device)
            depth = depth.to(device)
            gt = gt.to(device)
#            print(type(data))
            ps = model(data, depth)
            
#            print(gt.shape, ps.shape)
            
            loss = criterion(ps, gt)
#            print(t, loss.item())    
#            if t % 100 == 99:
#                print(t, loss.item())            
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()

        print('epoch: ', epoch, ', loss: ', loss.item())
        loss_val.append(loss.item())
        scheduler.step()
        
        if epoch % 1000 == 0:
            saved_model = 'checkpoint-{}.pth'.format(epoch)
#            with open(join(getcwd(), model_path, saved_model), 'rb') as f:
#                torch.save(checkpoint, f)
            torch.save(checkpoint, join(getcwd(), model_path, saved_model))
            with open(join(getcwd(), loss_path, 'loss.txt'), 'a') as f:
                f.write("\n".join(map(str, loss_val)))
            loss_val = []

            
if __name__ == '__main__':
    main()
