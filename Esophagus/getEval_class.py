import numpy as np
import skimage.io
import skimage.transform
import os
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import warnings
import cv2

model_output = 'result/multi_task_99'

parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir',
                    default='./' + model_output +'/ori_output')

parser.add_argument('--gt_dir',
                    default='./dataset/endoscope/val/label')
parser.add_argument('--masked_dir',
                    default='./'+ model_output +'/mask_dialated3_6_1')
parser.add_argument('--num_classes',
                    type=int,
                    default=2)
flags = parser.parse_args()

if not os.path.exists(flags.masked_dir):
    os.mkdir(flags.masked_dir)

def classification(output, gt):
    gt_label = np.unique(gt)
    group = 1
    if len(gt_label) == 1:
        if gt_label == 0:
            group = 0

    output_label = np.unique(output)

    if group == 1:
        if len(output_label) == 1 and output_label == 0:
            return 0, group
        return 1, group
    else:
        if len(output_label) == 1 and output_label == 0:
            return 1, group
        return 0, group

def classification2(output, gt, T):
    gt_label = np.unique(gt)
    group = 1
    if len(gt_label) == 1:
        if gt_label == 0:
            group = 0

    output = np.array(output)
    flat_output = output.flatten()
    flat_gt = gt.flatten()
    intersection = np.sum(flat_output * flat_gt)
    denominator = np.sum(flat_gt)

    if group == 1:
        if intersection / denominator > T:
            return 1, group
        return 0, group
    if group == 0:
        if np.sum(flat_output) / len(flat_gt) > (1 - T):
            return 0, group
        return 1, group

def get_T(class_dict, pred_dir):

    files = os.listdir(pred_dir)
    max_value = []
    for file in files:
        if file.find('.txt') != -1:
            continue
        max_value.append(class_dict[file])
    max_value.append(0.0)
    max_value = np.array(max_value).astype(np.float32)
    max_value = np.unique(max_value)
    max_value.sort()
    return max_value

def main(args):
    if not os.path.exists(flags.masked_dir):
        os.mkdir(flags.masked_dir)

    f = open(os.path.join(flags.pred_dir, 'class_dict.txt'), 'r')
    a = f.read()
    class_dict = eval(a)
    f.close()

    all_class_corr = []
    all_class_neg_corr = []
    all_class_pos_corr = []
    logs = []

    thresholds = get_T(class_dict, flags.pred_dir)

    for T in thresholds:
        with tf.device('/cpu:0'):
            class_corr = []
            class_pos_corr = []
            class_neg_corr = []

            list_dirs = os.walk(flags.pred_dir)
            for root, dirs, files in list_dirs:
                for f in files:
                    if (os.path.splitext(f)[1] == '.png'):
                        pred_path = os.path.join(root, f)
                        pred = cv2.imdecode(np.fromfile(pred_path, dtype=np.uint8), 1)
                        pred_class = class_dict[f]

                        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
                        pred = (pred > T).astype(np.int64)

                        gt_path = os.path.join(flags.gt_dir, f)
                        gt = skimage.io.imread(gt_path)
                        gt = (gt > (0.5 * 255)).astype(np.int64)


                        c, group = classification(pred, gt)
                        class_corr.append(float(float(pred_class > T) == group))
                        if group == 1:
                            class_pos_corr.append(float(float(pred_class > T) == group))
                        else:
                            class_neg_corr.append(float(float(pred_class > T) == group))
        print('T:', T, 'classAllAcc:', np.mean(class_corr), ' classPosAcc:', np.mean(class_pos_corr), ' classNegAcc:',
                      np.mean(class_neg_corr))
        logs.append('T: ' + str(T) + ' classAllAcc: ' + str(np.mean(class_corr)) + ' classPosAcc: ' + str(
                        np.mean(class_pos_corr)) + ' classNegAcc: ' + str(np.mean(class_neg_corr)))

        all_class_corr.append(np.mean(class_corr))
        all_class_neg_corr.append(np.mean(class_neg_corr))
        all_class_pos_corr.append(np.mean(class_pos_corr))

    plt.figure()
    plt.plot(thresholds, all_class_corr)
    plt.xlabel("T")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(flags.masked_dir, "class_accuracy.png"))

    plt.figure()
    tpr = np.array(all_class_pos_corr)
    fpr = (1.0 - np.array(all_class_neg_corr))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    class_auc_value = auc(fpr, tpr)
    plt.title("(AUC = %.4f)" % class_auc_value)
    plt.savefig(os.path.join(flags.masked_dir, "class_neg-pos_acc.png"))

    fp = open(os.path.join(flags.masked_dir, "result_class.txt"), 'w+')
    for log in logs:
        fp.write(log + '\n')
    fp.write('class_AUC = ' + str(class_auc_value) + '\n')
    fp.close()
    print("(AUC = %.4f)" % class_auc_value)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    tf.app.run()
