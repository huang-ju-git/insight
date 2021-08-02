# encoding: utf-8
"""
@author:  T. Zhang
@contact: tianyu1949@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY
from collections import defaultdict


@DATASET_REGISTRY.register()
class DMLPed12(ImageDataset):
    dataset_dir = ''
    dataset_name = "dml_ped12"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.ids = ['C','F','G','J','L','M','R','S','T','W','Y','Z']
        
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'dml_ped12')

        train, query, gallery = self.process_dir(data_dir)


        super(DMLPed12, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*/*.jpg'))
        pattern = re.compile(r'\/([A-Z])\/c([\d]+)_')
        camid_stats = defaultdict(int)
        train, query, gallery = [],[],[]
        for img_path in img_paths:
            if 'face' in img_path:
                continue 
            pid, camid = pattern.search(img_path).groups()
            if not  pid in self.ids:
                pid = 1000
            else:
                pid = self.ids.index(pid) 
            camid = int(camid)
            camid_stats[camid]+=1
            if pid<=5:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                train.append((img_path, pid, camid))
                
            else:
                if pid!=1000:
                    query.append((img_path, pid, camid))
                gallery.append((img_path, pid, camid))
        print(camid_stats)

        return train, query, gallery
