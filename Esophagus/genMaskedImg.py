import skimage.io
import os
import numpy as np
import cv2

input_img_dir = './dataset/endoscope/val/img'
input_mask_dir = './result/multi_task_99/mask_dialated3_6_1/128.0'
output_dir = './128.0_output'
T = 0.5
L = '_2'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

overlay_color_mask = [255, 0, 0]

transparency = 0
transparency = 1 - transparency

names = os.listdir(input_img_dir)

for name in names:
    name = os.path.splitext(name)
    img = skimage.io.imread(os.path.join(input_img_dir, name[0] + name[1]))

    img = cv2.imdecode(np.fromfile(os.path.join(input_img_dir, name[0] + name[1]), dtype=np.uint8), -1)

    mask = cv2.imdecode(np.fromfile(os.path.join(input_mask_dir, name[0] + L + name[1]), dtype=np.uint8), -1)
    ret_mask, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    binary_mask, contours_mask, hierarchy_mask = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours_mask, -1, overlay_color_mask, 3)

    cv2.imencode('.png', img)[1].tofile(os.path.join(output_dir, name[0] + L + name[1]))