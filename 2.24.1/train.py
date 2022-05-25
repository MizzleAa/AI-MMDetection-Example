# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes, update_data_root)

def dataset(cfg):
    # Modify dataset type and path
    cfg.dataset_type = 'COCODataset'

    cfg.data.train.ann_file = './data/balloon/train/json/train.json'
    cfg.data.train.img_prefix = './data/balloon/train/image'
    cfg.data.train.classes = ('balloon',)

    cfg.data.val.ann_file = './data/balloon/val/json/val.json'
    cfg.data.val.img_prefix = './data/balloon/val/image'
    cfg.data.val.classes = ('balloon',)

    cfg.data.test.ann_file = './data/balloon/test/json/test.json'
    cfg.data.test.img_prefix = './data/balloon/test/image'
    cfg.data.test.classes = ('balloon',)


    # modify num classes of the model in box head and mask head
    cfg.model.roi_head.bbox_head.num_classes = 1
    cfg.model.roi_head.mask_head.num_classes = 1

    # We can still the pre-trained Mask RCNN model to obtain a higher performance
    #cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = "./work"
    cfg.device = "cudo:0"
    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # We can set the evaluation interval to reduce the evaluation times
    cfg.evaluation.interval = 12
    # We can set the checkpoint saving interval to reduce the storage cost
    cfg.checkpoint_config.interval = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    
    # We can also use tensorboard to log the training process
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')]

    # We can initialize the logger for training and have a look
    # at the final config used for training
    return cfg

def main():
    cfg = Config.fromfile('./configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py')
    cfg = dataset(cfg)
    
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)
    
if __name__ == '__main__':
    main()
