# Copyright 2020 Toyota Research Institute.  All rights reserved.

# This script provides a demo inference a model trained on Cityscapes dataset.
import time
import warnings
import argparse
import cv2
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection.image_list import ImageList

from realtime_panoptic.models.rt_pano_net import RTPanoNet
from realtime_panoptic.config import cfg
import realtime_panoptic.data.panoptic_transform as P
from realtime_panoptic.utils.visualization import visualize_segmentation_image
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


def wait_for_stream(vid_getter):
    last_check = time.time()
    while vid_getter.frame is None:
        current_time = time.time()
        if current_time - last_check > 2:
            print("Waiting for stream...")
            last_check = current_time


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

    wait_for_stream(vid_getter)

    plt.figure(figsize=(10, 4))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    # Set no ticks.
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    frame = vid_getter.frame
    im1 = ax1.imshow(frame[:, :, ::-1])
    im2 = ax2.imshow(frame[:, :, ::-1])

    def update(i):
        start = time.time()

        frame = vid_getter.frame
        input_image = Image.fromarray(frame)
        data = {'image': input_image}
        data = transform(data)
        with torch.no_grad():
            input_image_list = ImageList([data['image'].to(device)], image_sizes=[
                input_image.size[::-1]])
            panoptic_result, _ = model.forward(input_image_list)
            semseg_logits = panoptic_result["semantic_segmentation_result"][0]
            semseg_prob = semseg_logits.argmax(dim=0)

            seg_vis = visualize_segmentation_image(
                semseg_prob, input_image, cityscapes_colormap)

        im1.set_data(frame[:, :, ::-1])
        im2.set_data(seg_vis.astype('uint8')[:, :, ::-1])

        total_time = time.time() - start
        print("Inference time: {:.2f} ms".format(total_time * 1000))

    ani = FuncAnimation(plt.gcf(), update, interval=1)
    plt.show()


if __name__ == "__main__":
    demo()
