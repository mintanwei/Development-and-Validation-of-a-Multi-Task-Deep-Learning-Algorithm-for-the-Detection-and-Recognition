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

def get_T(file_path):
    files = os.listdir(file_path)
    max_value = []

    for file in files:
        if file.find('.txt') != -1:
            continue
        pred_path = os.path.join(file_path, file)
        pred = cv2.imdecode(np.fromfile(pred_path, dtype=np.uint8), 1)

        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
        temp_value = np.unique(pred)
        for value in temp_value:
            max_value.append(value)

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

    all_corr = []
    all_neg_corr = []
    all_pos_corr = []
    all_class_corr = []
    all_class_neg_corr = []
    all_class_pos_corr = []
    all_miou = []
    logs = []

    thresholds = get_T(flags.pred_dir)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    kerne2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for T in thresholds:
        with tf.device('/cpu:0'):
            tf.reset_default_graph()
            output = tf.placeholder(tf.int64, shape=[None, None], name='output')
            label = tf.placeholder(tf.int64, shape=[None, None], name='label')
            output2 = tf.reshape(output, [-1, ])
            label2 = tf.reshape(label, [-1, ])
            mean_iou, update_mean_iou = tf.metrics.mean_iou(output2, label2, num_classes=flags.num_classes)

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            save_dir = os.path.join(flags.masked_dir, str(T))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            corr = []
            pos_corr = []
            neg_corr = []

            class_corr = []
            class_pos_corr = []
            class_neg_corr = []

            list_dirs = os.walk(flags.pred_dir)
            for root, dirs, files in list_dirs:
                for f in files:
                    if (os.path.splitext(f)[1] == '.png'):
                        pred_path = os.path.join(root, f)
                        pred = cv2.imdecode(np.fromfile(pred_path, dtype=np.uint8), 1)
                        pred = cv2.erode(pred, kerne2)
                        pred = cv2.dilate(pred, kernel)

                        pred_class = class_dict[f]

                        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
                        pred = (pred > T).astype(np.int64)

                        gt_path = os.path.join(flags.gt_dir, f)
                        gt = skimage.io.imread(gt_path)
                        gt = (gt > (0.5 * 255)).astype(np.int64)

                        mIou, _ = sess.run([mean_iou, update_mean_iou], feed_dict={output: pred, label: gt})

                        c, group = classification(pred, gt)
                        corr.append(c)
                        if group == 1:
                            pos_corr.append(c)
                        else:
                            neg_corr.append(c)

                        class_corr.append(float(float(pred_class > T / 255.0 ) == group))
                        if group == 1:
                            class_pos_corr.append(float(float(pred_class > T / 255.0) == group))
                        else:
                            class_neg_corr.append(float(float(pred_class > T / 255.0) == group))

                        name = os.path.splitext(f)
                        save_name1 = os.path.join(save_dir, name[0] + '_1' + name[1])
                        gt = gt.astype(np.float32)
                        label1 = skimage.img_as_ubyte(gt)
                        skimage.io.imsave(save_name1, label1)

                        save_name2 = os.path.join(save_dir, name[0] + '_2' + name[1])
                        pred = pred.astype(np.float32)
                        label_pred = skimage.img_as_ubyte(pred)
                        skimage.io.imsave(save_name2, label_pred)

        print('T:', T, '  meanIou:', mIou, ' allAcc:', np.mean(corr), ' posAcc', np.mean(pos_corr), ' negAcc',
                      np.mean(neg_corr), len(pos_corr), len(neg_corr))
        print('classAllAcc:', np.mean(class_corr), ' classPosAcc:', np.mean(class_pos_corr), ' classNegAcc:',
                      np.mean(class_neg_corr))
        logs.append(
                    'T: ' + str(T) + ' meanIou: ' + str(mIou) + ' allAcc: ' + str(np.mean(corr)) + ' posAcc: ' + str(
                        np.mean(pos_corr)) + ' negAcc: ' + str(np.mean(neg_corr)) +
                    ' classAllAcc: ' + str(np.mean(class_corr)) + ' classPosAcc: ' + str(
                        np.mean(class_pos_corr)) + ' classNegAcc: ' + str(np.mean(class_neg_corr)))
        all_corr.append(np.mean(corr))
        all_neg_corr.append(np.mean(neg_corr))
        all_pos_corr.append(np.mean(pos_corr))
        all_class_corr.append(np.mean(class_corr))
        all_class_neg_corr.append(np.mean(class_neg_corr))
        all_class_pos_corr.append(np.mean(class_pos_corr))
        all_miou.append(mIou)

    plt.figure()
    plt.plot(thresholds, all_corr)
    plt.xlabel("T")
    plt.ylabel("accuracy")
    plt.savefig(os.path.join(flags.masked_dir, "accuracy.png"))

    plt.figure()
    tpr = np.array(all_pos_corr)
    fpr = (1.0 - np.array(all_neg_corr))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    auc_value = auc(fpr, tpr)
    plt.title("(AUC = %.4f)" % auc_value)
    plt.savefig(os.path.join(flags.masked_dir, "neg-pos_acc.png"))

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

    plt.figure()
    plt.plot(thresholds, all_miou)
    plt.xlabel("T")
    plt.ylabel("mIou")
    plt.savefig(os.path.join(flags.masked_dir, "mIou.png"))

    fp = open(os.path.join(flags.masked_dir, "result.txt"), 'w+')
    for log in logs:
        fp.write(log + '\n')
    fp.write('AUC = ' + str(auc_value) + '\n')
    fp.write('class_AUC = ' + str(class_auc_value) + '\n')
    fp.close()
    print("(AUC = %.4f)" % auc_value)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    tf.app.run()
