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
import json
import random
import time
import numpy as np
import cv2

import torch
import torchvision
import subprocess

import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode

# User import
from object_detection.configs import cfg as gcfg
from object_detection.configs import ColorPrint
from object_detection.configs import get_specific_files
from object_detection.configs import get_specific_files_with_tag_in_name
from object_detection.visualization import viz_image_grid


def get_image():
    img_url = 'http://images.cocodataset.org/val2017/000000439715.jpg'
    img_save = osp.join(gcfg.get_temp_dir, 'input.jpg')
    if not osp.exists(img_save):
        subprocess.call(['wget', img_url, '-q', '-O', img_save])
    im = cv2.imread(img_save)
    return im


def get_save_path():
    img_save = osp.join(gcfg.get_temp_dir, 'output.png')
    return img_save


def get_dataset():
    # download, decompress the data
    zip_url = ('https://github.com/matterport/Mask_RCNN/releases/download/'
               'v2.1/balloon_dataset.zip')
    zip_save = osp.join(gcfg.get_temp_dir, 'balloon_dataset.zip')
    if not osp.exists(zip_save):
        subprocess.call(['wget', zip_url, '-O', zip_save])

        cwd_path = os.getcwd()
        os.chdir(gcfg.get_temp_dir)
        subprocess.call(['unzip', zip_save])
        os.chdir(cwd_path)


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
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()


def get_balloon_dicts(img_dir):
    """
    Notes
    -----
    if your dataset is in COCO format, this cell can be replaced by the
        following three lines:

    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("my_dataset_train", {},
                            "json_annotation_train.json", "path/to/image/dir")
    register_coco_instances("my_dataset_val", {},
                            "json_annotation_val.json", "path/to/image/dir")
    """
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def detect_balloon():
    for d in ["train", "val"]:
        DatasetCatalog.register(
            "balloon_" + d,
            lambda d=d: get_balloon_dicts(
                gcfg.get_temp_dir + '/' + "balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")

    dataset_dicts = get_balloon_dicts(gcfg.get_temp_dir + '/' + "balloon/train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        save_path = osp.join(gcfg.get_temp_dir, osp.basename(d["file_name"]))
        cv2.imwrite(save_path, out.get_image())


def train_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR

    # 300 iterations seems good enough for this toy dataset;
    #   you may need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 300

    # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

    cfg.OUTPUT_DIR = gcfg.get_ou_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


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

    # Run a pre-trained detectron2 model
    enable_use_pre_trained = True
    if enable_use_pre_trained:
        img_in = get_image()
        img_ou = predict_image(img_in)
        cv2.imwrite(get_save_path(), img_ou)

    # Train on a custom dataset
    enable_detect_balloon = False
    if enable_detect_balloon:
        get_dataset()
        detect_balloon()

    # fine-tune a COCO-pretrained R50-FPN Mask R-CNN model
    #   on the balloon dataset
    enable_fine_tune = False
    if enable_fine_tune:
        '''
        nvidia-smi --loop=2
        '''
        train_model()

    enable_process_dataset = True
    if enable_process_dataset:
        png_files = get_specific_files(gcfg.get_dataset_img_dir, '.png', True)
        ColorPrint.print_info('  - png_files: {}'.format(len(png_files)))

        base_dir = osp.basename(gcfg.get_dataset_img_dir)
        ColorPrint.print_info('  - base_dir: {}'.format(base_dir))

        img_num_per_bag = 500

        for idx in range(min(1200000, len(png_files))):
            idx_bag = int(idx / img_num_per_bag)

            save_dir = osp.join(gcfg.get_ou_dir,
                                '{}_{:05d}-{:05d}'.format(
                                    base_dir,
                                    idx_bag * img_num_per_bag,
                                    (idx_bag + 1) * img_num_per_bag))
            if not osp.exists(save_dir):
                os.mkdir(save_dir)

            img_in = cv2.imread(png_files[idx])
            img_name = osp.basename(png_files[idx])
            ColorPrint.print_info('  - process {}'.format(img_name))
            img_ou = predict_image(img_in)
            save_name = osp.join(save_dir, img_name)

            # grid_viz = viz_image_grid.VizImageGrid(im)

            cv2.imwrite(save_name, img_ou)

    time_end = time.time()
    ColorPrint.print_warn('elapsed {} seconds.'.format(time_end - time_beg))
