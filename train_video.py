### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import torch
from torch.autograd import Variable

import os
import numpy as np
import shutil
import video_utils

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    #opt.display_freq = 1
    #opt.print_freq = 1
    #opt.niter = 1
    #opt.niter_decay = 0
    #opt.max_dataset_size = 10
    pass

# additional enforced options for video
opt.task = "next_frame"
opt.label_nc = 0
opt.no_instance = True

# could be changed, but not tested
opt.resize_or_crop = "none"
opt.batchSize = 1

# this debug directory will contain input/output frame pairs
if opt.debug:
    debug_dir = os.path.join(opt.checkpoints_dir, opt.name, 'debug')
    if os.path.isdir(debug_dir):
        shutil.rmtree(debug_dir)
    os.mkdir(debug_dir)

if opt.scheduled_sampling:
    if opt.batchSize > 1:
        raise Exception('(for now) in "scheduled sampling" mode, --batchSize has to be 1')
    if not opt.serial_batches:
        raise Exception('(for now) in "scheduled sampling" mode, the --serial_batches option is necessary')
    if not opt.no_flip:
        raise Exception('(for now) in "scheduled sampling" mode, the --no_flip option is necessary')
    latest_generated_frame = None
    recursion = 0

data_loader = CreateDataLoader(opt) #this will look for a "frame dataset" at the location you specified
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
total_steps = (start_epoch-1) * dataset_size + epoch_iter
model = create_model(opt)
visualizer = Visualizer(opt)
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

import sys
from PIL import Image
from PIL import ImageFilter
from data.base_dataset import BaseDataset, get_params, get_transform, normalize

class Dataset(object):
    def __init__(self, mode="train"):
        from glob import glob
        self.dir_images = os.path.join(opt.dataroot, mode + "_images")

        if not os.path.isdir(self.dir_images):
            raise Exception('directory <%s> does not exist' % self.dir_images)

        self.img_paths = sorted(glob(self.dir_images+"/*.jpg"))
        self.img_count = len(self.img_paths)
        self.dataset_size = self.img_count - 1

        print("Dataset initialized from: %s" % self.dir_images)
        print("contains %d images" % (self.img_count))

    def get_sample(self, index):
        left_frame_path = self.img_paths[index]
        right_frame_path = self.img_paths[index+1]

        left_frame = Image.open(left_frame_path)
        right_frame = Image.open(right_frame_path)

        left_tensor = video_utils.im2tensor(left_frame)
        right_tensor = video_utils.im2tensor(right_frame)

        input_dict = {
            'left_frame': left_tensor,
            'left_path': left_frame_path,
            'right_frame': right_tensor,
            'right_path': right_frame_path,
        }

        return input_dict


EARLY_STOPPING = True
if EARLY_STOPPING:
    dev_loss_filepath = os.path.join(opt.checkpoints_dir, opt.name, 'dev_loss.txt')
    dev_set = Dataset(mode="val")
    round_without_improvement = 0
    dev_losses = []
    patience = 10
    best_g_loss = ("none", 1e6)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        ############## Forward Pass - frame t -> frame t+1 ######################

        if opt.scheduled_sampling and (latest_generated_frame is not None) and np.random.randn(1) < opt.ss_recursion_prob:
            left_frame = latest_generated_frame.detach()
            recursion += 1
        else:
            left_frame = Variable(data['left_frame'])
            recursion = 0

        right_frame = Variable(data['right_frame'])

        if opt.debug:
            video_utils.save_tensor(left_frame, debug_dir + "/step-%d-left-r%d.jpg" % (total_steps, recursion))
            video_utils.save_tensor(right_frame, debug_dir + "/step-%d-right.jpg" % total_steps)

        losses, latest_generated_frame = model(
            left_frame, None,
            right_frame, None,
            infer=opt.scheduled_sampling
        )

        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        ############### Backward Pass - frame1->frame2 ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward()
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()
        model.module.optimizer_D.step()

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
        model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

    if EARLY_STOPPING:
        # eval loss on dev set (after everything is saved)
        with torch.no_grad():
            epoch_sample_losses = []
            for i in range(dev_set.dataset_size):
                data = dev_set.get_sample(i)

                left_frame = Variable(data['left_frame'])      
                right_frame = Variable(data['right_frame'])

                losses, latest_generated_frame = model(
                    left_frame, None,
                    right_frame, None,
                    infer=False
                )

                # sum per device losses
                losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                loss_dict = dict(zip(model.module.loss_names, losses))

                # calculate final loss scalar
                loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

                epoch_sample_losses.append(loss_G.item())

        epoch_g_loss = np.mean(epoch_sample_losses)
        print("epoch", epoch, "G loss on dev set:", epoch_g_loss)
        with open(dev_loss_filepath, "a+") as f:
            f.write("epoch %s - dev loss %.4f\n" %(str(epoch), float(epoch_g_loss)))

        dev_losses.append(epoch_g_loss)
        if float(epoch_g_loss) < best_g_loss[1]:
            print("new best")
            with open(dev_loss_filepath, "a+") as f:
                f.write("new best\n")
            best_g_loss = (epoch, float(epoch_g_loss))
            round_without_improvement = 0
        else:
            round_without_improvement += 1

        if round_without_improvement >= patience:
            print("early stopping, best G loss:", best_g_loss)
            with open(dev_loss_filepath, "a+") as f:
                f.write("early stopping, best G loss: %.4f at epoch %s\n" % (float(best_g_loss[1]), str(best_g_loss[0])))
            sys.exit(0)
