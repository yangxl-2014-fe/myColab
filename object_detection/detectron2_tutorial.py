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


def get_model_list():
    """
    Where is MODEL.WEIGHTS placed after download from model_zoo? #773
      https://github.com/facebookresearch/detectron2/issues/773#issuecomment-580603027
    /home/ftx/Documents/yangxl-2014-fe/my_forked/detectron2/detectron2/model_zoo/model_zoo.py
    """
    yaml_dict = {  # 52 models
        # COCO Detection with Faster R-CNN (10)
        "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml": "137257644/model_final_721ade.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml": "137847829/model_final_51d356.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml": "137257794/model_final_b275ba.pkl",
        "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml": "137849393/model_final_f97cb7.pkl",
        "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml": "137849425/model_final_68d202.pkl",
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml": "137849458/model_final_280758.pkl",
        "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml": "138204752/model_final_298dad.pkl",
        "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml": "138204841/model_final_3e0943.pkl",
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml": "137851257/model_final_f6e8b1.pkl",
        "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml": "139173657/model_final_68b088.pkl",
        # COCO Detection with RetinaNet (3)
        "COCO-Detection/retinanet_R_50_FPN_1x.yaml": "190397773/model_final_bfca0b.pkl",
        "COCO-Detection/retinanet_R_50_FPN_3x.yaml": "190397829/model_final_5bd44e.pkl",
        "COCO-Detection/retinanet_R_101_FPN_3x.yaml": "190397697/model_final_971ab9.pkl",
        # COCO Detection with RPN and Fast R-CNN (3)
        "COCO-Detection/rpn_R_50_C4_1x.yaml": "137258005/model_final_450694.pkl",
        "COCO-Detection/rpn_R_50_FPN_1x.yaml": "137258492/model_final_02ce48.pkl",
        "COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml": "137635226/model_final_e5f7ce.pkl",
        # COCO Instance Segmentation Baselines with Mask R-CNN (10)
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml": "137259246/model_final_9243eb.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml": "137260150/model_final_4f86c3.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml": "137260431/model_final_a54504.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml": "137849525/model_final_4ce675.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml": "137849551/model_final_84107b.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml": "137849600/model_final_f10217.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml": "138363239/model_final_a2914c.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml": "138363294/model_final_0464b7.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml": "138205316/model_final_a3ec72.pkl",
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml": "139653917/model_final_2d9806.pkl",  # noqa
        # COCO Person Keypoint Detection Baselines with Keypoint R-CNN (4)
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml": "137261548/model_final_04e291.pkl",
        "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml": "137849621/model_final_a6e10b.pkl",
        "COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml": "138363331/model_final_997cc7.pkl",
        "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml": "139686956/model_final_5ad38f.pkl",
        # COCO Panoptic Segmentation Baselines with Panoptic FPN (3)
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_1x.yaml": "139514544/model_final_dbfeb4.pkl",
        "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml": "139514569/model_final_c10459.pkl",
        "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml": "139514519/model_final_cafdb1.pkl",
        # LVIS Instance Segmentation Baselines with Mask R-CNN (3)
        "LVISv0.5-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml": "144219072/model_final_571f7c.pkl",  # noqa
        "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml": "144219035/model_final_824ab5.pkl",  # noqa
        "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml": "144219108/model_final_5e3439.pkl",  # noqa
        # Cityscapes & Pascal VOC Baselines (2)
        "Cityscapes/mask_rcnn_R_50_FPN.yaml": "142423278/model_final_af9cf5.pkl",
        "PascalVOC-Detection/faster_rcnn_R_50_C4.yaml": "142202221/model_final_b1acc2.pkl",
        # Other Settings (11)
        "Misc/mask_rcnn_R_50_FPN_1x_dconv_c3-c5.yaml": "138602867/model_final_65c703.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_dconv_c3-c5.yaml": "144998336/model_final_821d0b.pkl",
        "Misc/cascade_mask_rcnn_R_50_FPN_1x.yaml": "138602847/model_final_e9d89b.pkl",
        "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml": "144998488/model_final_480dd8.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_syncbn.yaml": "169527823/model_final_3b3c51.pkl",
        "Misc/mask_rcnn_R_50_FPN_3x_gn.yaml": "138602888/model_final_dc5d9e.pkl",
        "Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml": "138602908/model_final_01ca85.pkl",
        "Misc/scratch_mask_rcnn_R_50_FPN_9x_gn.yaml": "183808979/model_final_da7b4c.pkl",
        "Misc/scratch_mask_rcnn_R_50_FPN_9x_syncbn.yaml": "184226666/model_final_5ce33e.pkl",
        "Misc/panoptic_fpn_R_101_dconv_cascade_gn_3x.yaml": "139797668/model_final_be35db.pkl",
        "Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml": "18131413/model_0039999_e76410.pkl",  # noqa
        # D1 Comparisons (3)
        "Detectron1-Comparisons/faster_rcnn_R_50_FPN_noaug_1x.yaml": "137781054/model_final_7ab50c.pkl",  # noqa
        "Detectron1-Comparisons/mask_rcnn_R_50_FPN_noaug_1x.yaml": "137781281/model_final_62ca52.pkl",  # noqa
        "Detectron1-Comparisons/keypoint_rcnn_R_50_FPN_1x.yaml": "137781195/model_final_cce136.pkl",
    }
    return sorted(yaml_dict.keys())


def download_model(model):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    print(f'loading {cfg.MODEL.WEIGHTS}')
    local_save_dir = '/home/ftx/.torch/fvcore_cache/detectron2'
    remote_url = 'https://dl.fbaipublicfiles.com/detectron2/'
    src_file = cfg.MODEL.WEIGHTS
    name = model.replace(".yaml", "")
    sub_dir = osp.join(remote_url, name)
    str_suffix = src_file[len(sub_dir) + 1:]
    dst_file = osp.join(local_save_dir, name + '/' + str_suffix)

    dst_dir = osp.split(dst_file)[0]
    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)
    if not osp.exists(dst_file):
        # subprocess.call(['wget', src_file, '-O', dst_file])
        print(f'wget {src_file} -O {dst_file}')
    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(im)


def download_all_pre_trained_models():
    models = get_model_list()
    print(f'models: {len(models)} {models}')

    im = get_image()
    for model in models:
        print(f'downloading {model}')
        download_model(model)


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
    enable_use_pre_trained = False
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

    enable_process_dataset = False
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
            img_ou = cv2.resize(img_ou, (img_in.shape[1], img_in.shape[0]))

            # grid_viz = viz_image_grid.VizImageGrid(im)

            cv2.imwrite(save_name, img_ou)

    # download all pre-trained models
    enable_download_all_pre_trained_models = True
    if enable_download_all_pre_trained_models:
        download_all_pre_trained_models()

    time_end = time.time()
    ColorPrint.print_warn('elapsed {} seconds.'.format(time_end - time_beg))
