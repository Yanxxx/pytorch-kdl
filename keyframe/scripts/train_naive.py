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
from attention import NaiveAttention
import utils as utils
import datetime
import torch
import torch.nn as nn
from os import mkdir, getcwd
from os.path import join, dirname
from torch.utils.tensorboard import SummaryWriter


def main():
    device = utils.selectDevice(0)
    
    rot, t = utils.camTrans()
    
    model = NaiveAttention().to(device)
    # preparing dataset
    print('preparing dataset')
    data_path = '/workspace/datasets/block-insertion-test/'    
    print('loading train dataset.')
    training_set = dataset(data_path)#, [0, 0.1])
    print('loading validate dataset.')
    validating_set = dataset(data_path, [0.75, 1])
        
    params = {'batch_size': 64,
              'shuffle': True,
              'num_workers': 16}    
    train_generator = torch.utils.data.DataLoader(training_set, **params)
    validate_generator = torch.utils.data.DataLoader(validating_set, **params)
    # setup loss fucntion
    print('setting up loss function')
    criterion = nn.MSELoss()
    # setup optimizer 
    print('setup optimizer')
    learning_rate = 1e-3
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2000, 3000, 4000, 5000], gamma=0.1)

    # maximum epochs
    max_epochs = 10000
    epoch = 0
    checkpoint = { 
    'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_sched': scheduler}
    loss_val = []
    valid_loss = []
    
    # start the training
    print('starting the training process')
    curr_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_path = join(dirname(getcwd()), 'checkpoints', curr_time)
    mkdir(model_path)
    loss_path = join(dirname(getcwd()), 'loss', curr_time)
    mkdir(loss_path)
    
#    writer = SummaryWriter('../logs/fashion_mnist_experiment_1')
#    writer.add_graph(model)
    for epoch in range(max_epochs):
        
        model.train()
        for data, depth, gt in train_generator:
            data = data.to(device)
            depth = depth.to(device)            
            gt = gt.to(device)
            ps = model(data)         
            loss = criterion(ps, gt)       
            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()
        tl = loss.item()
        loss_val.append(tl)
                
        model.eval()        
        for data, depth, gt in validate_generator:
            data = data.to(device)
            depth = depth.to(device)
            gt = gt.to(device)  
#            print('validate ', gt.shape)
            ps = model(data)   
            loss = criterion(ps, gt)
#            print('validate ps ', ps.shape)
            vl = loss.item()
            valid_loss.append(vl)
            
        if epoch % 1000 == 0:
            saved_model = 'checkpoint-{}.pth'.format(epoch)
            torch.save(checkpoint, join(getcwd(), model_path, saved_model))
            with open(join(getcwd(), '../',loss_path, 'navie-train-loss.txt'), 'a') as f:
                f.write('\n')
                f.write("\n".join(map(str, loss_val)))
            loss_val = []
            with open(join(getcwd(), '../', loss_path, 'navie-validate-loss.txt'), 'a') as f:
                f.write('\n')
                f.write("\n".join(map(str, valid_loss)))
                valid_loss = []
            
        print('epoch: ', epoch, ', train-loss: ', tl, 'validate-loss', vl)
        scheduler.step()
#    writer.close()

            
if __name__ == '__main__':
    main()
