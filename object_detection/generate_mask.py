# -*- coding: utf-8 -*-

"""
References
----------
https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
    Detectron2 Beginner's Tutorial
"""

import sys
import os
import os.path as osp
import logging
# import json
# import random
import time
import numpy as np
import cv2

import torch
# import torchvision
import subprocess

# import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
# from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask, ColorMode
# from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import MetadataCatalog

# from detectron2.structures import BoxMode

# User import
from object_detection.configs import cfg as gcfg
from object_detection.configs import ColorPrint
from object_detection.configs import get_specific_files
from object_detection.configs import get_specific_files_no_tag
# from object_detection.configs import get_specific_files_with_tag_in_name
# from object_detection.visualization import viz_image_grid


def value_statis(data, desc):
    labels, areas = np.unique(data, return_counts=True)

    sorted_idxs = np.argsort(-areas).tolist()
    labels = labels[sorted_idxs]
    areas = areas[sorted_idxs]
    print('== {}: {} {} =='.format(desc, data.shape, data.dtype))
    if len(labels) > 20:
        print('  - label:  {} {}'.format(labels.shape, labels.dtype))
        print('  - count:  {} {}'.format(areas.shape, areas.dtype))
    else:
        print('  - label:  {} {} {}'.format(labels.shape, labels.dtype, labels))
        print('  - count:  {} {} {}'.format(areas.shape, areas.dtype, areas))


def predict_image(im):
    cfg = get_cfg()

    # add project-specific config (e.g., TensorMask) here if you're not
    #   running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

    # Find a model from detectron2's model zoo. You can
    #   use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    '''
    look at the outputs. See
    https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    for specification
    '''
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)

    # We can use `Visualizer` to draw the predictions on the image.
    '''
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2)
    '''
    v = Visualizer(np.zeros_like(im),
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.0)
    im_mask = np.zeros_like(im, np.uint8)

    if outputs["instances"].has("pred_masks"):
        masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
        value_statis(masks, 'massk1')
        print('masks1: {}'.format(type(masks)))

        for idx_i in range(masks.shape[0]):
            im_mask[masks[idx_i, :, :]] = idx_i * 2 + 10

        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]

        print('masks2: {}'.format(type(masks)))
        num_instance = len(v._convert_masks(masks))
        print('  -> num_instances: {}'.format(num_instance))
        out = v.overlay_instances(masks=masks)
    else:
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    value_statis(out.get_image(), 'out.get_image()')
    value_statis(im_mask, 'im_mask')

    # return out.get_image()
    return im_mask


def predict_image_opt(im):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    im_mask = np.zeros_like(im, np.uint8)
    if outputs["instances"].has("pred_masks"):
        masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
        for idx_i in range(masks.shape[0]):
            im_mask[masks[idx_i, :, :]] = idx_i * 2 + 10
    return im_mask


def process_mynt_dataset():
    png_files = get_specific_files_no_tag(
        gcfg.get_dataset_img_dir, '.png', '-fseg', True)
    png_files = png_files[:5]
    ColorPrint.print_info('  - png_files: {}'.format(len(png_files)))

    base_dir = osp.basename(gcfg.get_dataset_img_dir)
    ColorPrint.print_info('  - base_dir: {}'.format(base_dir))

    img_num_per_bag = 500

    for idx in range(min(1200000, len(png_files))):
        idx_bag = int(idx / img_num_per_bag)

        save_dir = osp.join(
            gcfg.get_ou_dir,
            '{}_{:05d}-{:05d}'.format(
                base_dir, idx_bag * img_num_per_bag,
                (idx_bag + 1) * img_num_per_bag))
        if not osp.exists(save_dir):
            os.mkdir(save_dir)

        img_in = cv2.imread(png_files[idx])
        img_name = osp.basename(png_files[idx])
        ColorPrint.print_info('  - process {}'.format(img_name))
        img_ou = predict_image_opt(img_in)  # predict
        str_base, str_ext = osp.splitext(img_name)
        save_name = osp.join(save_dir, str_base + '-fseg' + str_ext)
        # img_ou = cv2.resize(img_ou, (img_in.shape[1], img_in.shape[0]))

        # grid_viz = viz_image_grid.VizImageGrid(im)

        cv2.imwrite(save_name, img_ou)


def process_kitti_dataset():
    # check kitti dir
    kitti_home = gcfg.get_dataset_kitti_seq
    if not osp.exists(kitti_home):
        raise ValueError

    drive_collections = list()
    for item_date in os.listdir(kitti_home):
        if not osp.isdir(osp.join(kitti_home, item_date)):
            continue
        subdir_date = osp.join(kitti_home, item_date)
        for item_drive in os.listdir(subdir_date):
            if not osp.isdir(osp.join(subdir_date, item_drive)):
                continue
            drive_collections.append(osp.join(subdir_date, item_drive))

    # check drive dir
    if len(drive_collections) == 0:
        raise ValueError

    drive_collections.sort()
    img_col = list()
    for item_dir in drive_collections:
        png_cam2 = get_specific_files_no_tag(
            osp.join(item_dir, 'image_02/data'), '.png', '-fseg', True)
        png_cam3 = get_specific_files_no_tag(
            osp.join(item_dir, 'image_03/data'), '.png', '-fseg', True)

        if len(png_cam2) != len(png_cam3):
            raise ValueError
        img_col.append(png_cam2)
        img_col.append(png_cam3)

    # generate mask with detectron2
    img_num_total = 0
    for seq in img_col:
        data_dir, _ = osp.split(seq[0])
        ColorPrint.print_info('- process_dir: {}'.format(data_dir))
        ColorPrint.print_info('- png_files:   {}'.format(len(seq)))
        img_num_total += len(seq)
        for item_png in seq:
            img_in = cv2.imread(item_png)
            str_dirname, str_basename = osp.split(item_png)
            str_base, str_ext = osp.splitext(str_basename)
            save_name = osp.join(str_dirname, str_base + '-fseg' + str_ext)
            ColorPrint.print_info('  - process {}'.format(str_basename))
            img_ou = predict_image_opt(img_in)  # predict
            cv2.imwrite(save_name, img_ou)
    ColorPrint.print_warn('processed {} images in total.'.format(img_num_total))


################################################################################
# main
################################################################################
if __name__ == '__main__':
    setup_logger()
    print('sys.version:               {}'.format(sys.version))
    print('np.__version__:            {}'.format(np.__version__))
    print('torch.__version__:         {}'.format(torch.__version__))
    print('torch.cuda.is_available(): {}'.format(torch.cuda.is_available()))

    print('\n')
    subprocess.call(['gcc', '--version'])

    time_beg = time.time()
    # process_mynt_dataset()
    process_kitti_dataset()
    time_end = time.time()
    ColorPrint.print_warn('elapsed {} seconds.'.format(time_end - time_beg))
