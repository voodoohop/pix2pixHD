### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from PIL import ImageFilter

def to_edges(pil_img):
    img = pil_img.convert('L')
    img = img.filter(ImageFilter.CONTOUR)
    img = img.filter(ImageFilter.SMOOTH)
    return img

class EdgesDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### frames
        dir_frames = '_images'
        self.dir_frames = os.path.join(opt.dataroot, opt.phase + dir_frames)
        self.frame_paths = sorted(make_dataset(self.dir_frames))
        self.frame_count = len(self.frame_paths)
        self.dataset_size = self.frame_count

        print("FrameDataset initialized from: %s" % self.dir_frames)
        print("contains %d frames" % (self.frame_count))
      
    def __getitem__(self, index):

        left_frame_path = self.frame_paths[index]
        right_frame_path = self.frame_paths[index]

        frame = Image.open(left_frame_path)
        frame = frame.resize((512,512))
        
        left_frame = to_edges(frame)
        right_frame = frame

        params = get_params(self.opt, left_frame.size)
        transform = get_transform(self.opt, params)

        left_tensor = transform(left_frame)
        right_tensor = transform(right_frame)

        input_dict = {
            'left_frame': left_tensor, 
            'left_path': left_frame_path,
            'right_frame': right_tensor, 
            'right_path': right_frame_path,
        }

        return input_dict

    def __len__(self):
        # batchSize>1 not tested
        return self.dataset_size // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'FrameDataset'
