import torch 
from os.path import join
import datetime
from keyframe import Keyframe
import utils
from dataset import dataset


device = utils.selectDevice()
rot, t = utils.camTrans()    
model = Keyframe(rot.to(device), t.to(device)).to(device)    
path = 'checkpoints'
checkpoint_file = 'checkpoint-21000.pth'

checkpoint = torch.load(join(path, checkpoint_file))
model.load_state_dict(checkpoint['model'])

model.eval()

data_path = '/workspace/datasets/key_frame_identifier/block-insertion-test/' 
data = dataset(data_path)

color, depth, gt = data[1]
print(color.shape, depth.shape, gt.shape)

color = torch.reshape(color, (1, 4, 120, 160)).to(device)
depth = torch.reshape(depth, (1, 1, 480, 640)).to(device)
out = model(color, depth)
print(out.shape)


