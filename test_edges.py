### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model

from tqdm import tqdm
from PIL import Image
import torch
import shutil
import video_utils
import image_transforms

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# additional enforced options for "edges"
opt.task = "edges2x"
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True
opt.resize_or_crop = "none"
opt.input_nc = 1

# loading initial frames from: ./datasets/NAME/test_frames
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# this directory will contain the generated data
output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


model = create_model(opt)

for i, data in enumerate(dataset):
    left = data['left_frame']
    right = data['right_frame']
    predicted_right = video_utils.next_frame_prediction(model, data['left_frame'])
    
    edges = left
    real = right
    pred = predicted_right

    #video_utils.save_tensor(
    #    edges, 
    #    output_dir + "/sample-%s_edges.jpg" % (str(i).zfill(5)),
    #)
    #video_utils.save_tensor(
    #    real, 
    #    output_dir + "/sample-%s_real.jpg" % (str(i).zfill(5)),
    #)
    video_utils.save_tensor(
        pred, 
        output_dir + "/sample-%s_pred.jpg" % (str(i).zfill(5)),
    )
    
