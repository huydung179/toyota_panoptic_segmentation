# Copyright 2020 Toyota Research Institute.  All rights reserved.

# This script provides a demo inference a model trained on Cityscapes dataset.
import time
import warnings
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.models.detection.image_list import ImageList
from torchvision import transforms as T

from realtime_panoptic.models.rt_pano_net import RTPanoNet
from realtime_panoptic.config import cfg
import realtime_panoptic.data.panoptic_transform as P
from realtime_panoptic.utils.visualization import visualize_segmentation_image, visualize_detection_image
from utils import VideoGet

cityscapes_colormap = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]])

cityscapes_instance_label_name = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
warnings.filterwarnings("ignore", category=UserWarning)


def demo():
    # Parse the input arguments.
    parser = argparse.ArgumentParser(
        description="Simple demo for real-time-panoptic model")
    parser.add_argument("--config-file", metavar="FILE",
                        help="path to config", required=True)
    parser.add_argument("--pretrained-weight", metavar="FILE",
                        help="path to pretrained_weight", required=True)
    parser.add_argument("--input", default=0, metavar=int)
    parser.add_argument("--device", help="inference device", default='cuda')
    args = parser.parse_args()

    # General config object from given config files.
    cfg.merge_from_file(args.config_file)

    # Initialize model.
    model = RTPanoNet(
        backbone=cfg.model.backbone,
        num_classes=cfg.model.panoptic.num_classes,
        things_num_classes=cfg.model.panoptic.num_thing_classes,
        pre_nms_thresh=cfg.model.panoptic.pre_nms_thresh,
        pre_nms_top_n=cfg.model.panoptic.pre_nms_top_n,
        nms_thresh=cfg.model.panoptic.nms_thresh,
        fpn_post_nms_top_n=cfg.model.panoptic.fpn_post_nms_top_n,
        instance_id_range=cfg.model.panoptic.instance_id_range)
    device = args.device
    model.to(device)
    model.load_state_dict(torch.load(args.pretrained_weight))

    # Print out mode architecture for sanity checking.
    print(model)

    # Prepare for model inference.
    model.eval()

    transform = P.Compose([
        P.ToTensor(),
        P.Normalize(mean=cfg.input.pixel_mean,
                    std=cfg.input.pixel_std, to_bgr255=cfg.input.to_bgr255),
    ])

    vid = VideoGet()
    vid.stream.set(cv2.CAP_PROP_FOURCC,
                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    vid_getter = vid.start()

    count = 0
    while 1:
        if (cv2.waitKey(1) == ord('q')):
            vid_getter.stop()
            break
        frame = vid_getter.frame
        if frame is None:
            count += 1
            continue
        if count == 10:
            vid_getter.stop()
            break

        cv2.imshow('Frame', frame)
        input_image = Image.fromarray(frame)
        start = time.time()

        data = {'image': input_image}
        data = transform(data)
        with torch.no_grad():
            input_image_list = ImageList([data['image'].to(device)], image_sizes=[
                                         input_image.size[::-1]])
            panoptic_result, _ = model.forward(input_image_list)
            semseg_logics = [
                o.to('cpu') for o in panoptic_result["semantic_segmentation_result"]]
            semseg_prob = [torch.argmax(semantic_logit, dim=0)
                           for semantic_logit in semseg_logics]

            seg_vis = visualize_segmentation_image(
                semseg_prob[0], input_image, cityscapes_colormap)
            cv2.imshow('Panoptic', seg_vis)
        total_time = time.time() - start
        print("Inference time: {:.2f} ms".format(total_time * 1000))


if __name__ == "__main__":
    demo()
