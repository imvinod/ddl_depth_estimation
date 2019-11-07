from PIL import Image
import numpy as np
import cv2
import torch
from torch.autograd import Variable
import os
from .models import net
import sys
sys.path.append('../')
sys.path.append('.')
folder = os.path.dirname(os.path.abspath(__file__))


# Pre-processing and post-processing constants #
CMAP = np.load(os.path.join(folder, 'cmap_kitti.npy'))
DEPTH_COEFF = 800. # to convert into metres
HAS_CUDA = torch.cuda.is_available()
IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
MAX_DEPTH = 80.
MIN_DEPTH = 0.
NUM_CLASSES = 6
NUM_TASKS = 2 # segm + depth

def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD

def run(img):
    model = net(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS)
    if HAS_CUDA:
        _ = model.cuda()
        _ = model.eval()
        
    ckpt = torch.load(os.path.join(folder,'../weights/ExpKITTI_joint.ckpt'))
    model.load_state_dict(ckpt['state_dict'])
    
    img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2, 0, 1)[None]), requires_grad=False).float()
    if HAS_CUDA:
        img_var = img_var.cuda()

    segm, depth = model(img_var)
    segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                      img.shape[:2][::-1],
                      interpolation=cv2.INTER_CUBIC)
    depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                       img.shape[:2][::-1],
                       interpolation=cv2.INTER_CUBIC)

    segm = CMAP[segm.argmax(axis=2)].astype(np.uint8)
    depth = np.abs(depth)

    
    return depth