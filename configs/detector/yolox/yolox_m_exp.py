#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
from pathlib import Path

# 添加YOLOX路径
YOLOX_ROOT = Path(__file__).parent.parent.parent / "detector" / "YOLOX"
if str(YOLOX_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLOX_ROOT))

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        
        # YOLOX-M model config
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = "yolox_m"
        
        # dataset config
        self.data_dir = "/home/wh1234_/data/20251027_coco_811_640"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"
        
        # image directories
        self.train_name = "train2017"
        self.val_name = "val2017"
        self.test_name = "test2017"
        
        # class config
        self.num_classes = 4
        
        # training config
        self.max_epoch = 120
        self.data_num_workers = 8
        self.eval_interval = 1
        self.print_interval = 1
        self.seed = 42
        self.save_history_ckpt = False
        self.ckpt = "weights/yolox_m.pth"
        
        # output config
        self.output_dir = "runs/detector_cmp"
        
        # optimizer config
        self.basic_lr_per_img = 0.01 / 64.0
        self.warmup_epochs = 5
        
        # input config
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        
        # batch size (M规模通常使用较小的batch size)
        self.batch_size = 24

