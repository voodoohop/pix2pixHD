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
import util.util as util

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# additional enforced options for "left2right"
opt.task = "left2right"
opt.label_nc = 0
opt.no_instance = True
opt.resize_or_crop = "none"

# loading initial frames from: ./datasets/NAME/test_frames
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# this directory will contain the generated videos
output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


model = create_model(opt)

for i, data in enumerate(dataset):
    left = data['left_frame']
    right = data['right_frame']
    predicted_right = video_utils.next_frame_prediction(model, data['left_frame'])
    
    predicted_left = image_transforms.flip_left_right(
        video_utils.next_frame_prediction(
            model, 
            image_transforms.flip_left_right(predicted_right)
        )
    )
    
    real = image_transforms.concatenate(left, right)
    pred = image_transforms.concatenate(left, predicted_right)
  
    video_utils.save_tensor(
        real, 
        output_dir + "/sample-%s_real.jpg" % (str(i).zfill(5)),
    )
    video_utils.save_tensor(
        pred, 
        output_dir + "/sample-%s_pred.jpg" % (str(i).zfill(5)),
    )

    img_list = [util.tensor2im(t.data[0]) for t in [real, pred]]
    video_utils.gif_from_images(img_list, output_dir + "/sample-%s.gif" % (str(i).zfill(5)))
