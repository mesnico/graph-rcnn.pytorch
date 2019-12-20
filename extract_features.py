"""
Implementation of ECCV 2018 paper "Graph R-CNN for Scene Graph Generation".
Author: Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, Devi Parikh
Contact: jw2yang@gatech.edu
"""

import os
import pprint
import argparse
import numpy as np
import torch
import datetime

from lib.config import cfg
from lib.model import build_model
from lib.scene_parser.rcnn.utils.miscellaneous import mkdir, save_config, get_timestamp
from lib.scene_parser.rcnn.utils.comm import synchronize, get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger


def inference(cfg, args, model=None):
    """
    test scene graph generation model
    """
    if model is None:
        arguments = {}
        arguments["iteration"] = 0
        model = build_model(cfg, arguments, args.local_rank, args.distributed)
    model.inference(visualize=args.visualize)


def extract_visual_features(dataset='coco', algorithm='sg_imp'):
    ''' parse config file '''
    parser = argparse.ArgumentParser(description="Graph Reasoning Machine for Visual Question Answering")
    # parser.add_argument("--config-file", default="configs/baseline_res101.yaml")
    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--resume", type=int, default=0)
    # parser.add_argument("--evaluate", action='store_true')
    # parser.add_argument("--inference", action='store_true')
    # parser.add_argument("--instance", type=int, default=-1)
    # parser.add_argument("--use_freq_prior", action='store_true')
    # parser.add_argument("--visualize", action='store_true')
    # parser.add_argument("--algorithm", type=str, default='sg_baseline')
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    config_file = os.path.join('checkpoints', 'sgg_res101_joint_coco_inference.yaml')
    cfg.merge_from_file(config_file)

    cfg.inference = True

    # Override something in the configuration
    cfg.MODEL.DUMP_FEATURES = True
    cfg.DATASET.NAME = dataset
    cfg.MODEL.ALGORITHM = algorithm
    if algorithm == 'sg_imp':
        cfg.MODEL.WEIGHT_DET = '/media/nicola/Data/Workspace/TransformItInYourMind/graph-rcnn.pytorch/checkpoints/sg_imp_step_ckpt.pth'
    elif algorithm == 'sg_baseline':
        cfg.MODEL.WEIGHT_DET = '/media/nicola/Data/Workspace/TransformItInYourMind/graph-rcnn.pytorch/checkpoints/sg_baseline_ckpt.pth'
    else:
        raise ValueError("The network comes not pretrained with {} algorithm!".format(algorithm))
    # cfg.MODEL.USE_FREQ_PRIOR = args.use_freq_prior
    # cfg.MODEL.ALGORITHM = args.algorithm
    # cfg.freeze()

    if not os.path.exists("logs") and get_rank() == 0:
        os.mkdir("logs")
    logger = setup_logger("scene_graph_generation", "logs", get_rank(),
        filename="{}_{}.txt".format(args.algorithm, get_timestamp()))

    logger.info("Loaded configuration file {}".format(config_file))
    # output_config_path = os.path.join("logs", 'config.yml')
    # logger.info("Saving config into: {}".format(output_config_path))
    # save_config(cfg, output_config_path)

    # start inference for dumping features
    inference(cfg, args)
