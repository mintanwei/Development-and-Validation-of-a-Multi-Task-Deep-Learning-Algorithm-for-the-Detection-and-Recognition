import os
import glob
import tensorflow as tf
import numpy as np
import argparse
import skimage.io
import MODEL2
import skimage.transform
import matplotlib.pyplot as plt

h = 224
w = 224
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir',
                    type=str,
                    default='dataset/endoscope/val/img')
parser.add_argument('--gt_dir',
                    type=str,
                    default='dataset/endoscope/val/label')
parser.add_argument('--model_dir',
                    type=str,
                    default='model/multi_task')
parser.add_argument('--model_name',
                    type=str,
                    default='model99.ckpt')
parser.add_argument('--save_dir',
                    type=str,
                    default='./result/multi_task_99')
parser.add_argument('--gpu',
                    type=int,
                    default=0)
flags = parser.parse_args()

def Iou(pred, label):
    h, w = pred.shape
    pred = np.reshape(pred, (h * w, 1))
    label = np.reshape(label, (h * w, 1))
    intersection = np.sum(np.multiply(label, pred))
    union = np.sum(label) + np.sum(pred)
    iou = (intersection + 1e-7) /(union - intersection + 1e-7)
    return iou

def precision(pred, label):
    h, w = pred.shape
    pred = np.reshape(pred, (h * w, 1))
    label = np.reshape(label, (h * w, 1))
    intersection = np.sum(np.equal(label, pred).astype(np.float32))
    return intersection / (h * w)

def precision_class(pred, label):
    if pred == label:
        return 1.0
    else:
        return 0.0

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
def sigmiod(score):
    return 1 / (1 + np.exp(-1 * score))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(flags):
    filename = './dataset/last_fc.tfrecords'
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    writer = tf.python_io.TFRecordWriter(filename)

    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.float32, shape=[None, h, w, 3], name='X')
        y = tf.placeholder(tf.float32, shape=[None, h, w, 1], name='y')
        image_label = tf.placeholder(tf.float32, shape=[None, 1], name='image_label')
        mode = tf.placeholder(tf.bool, name='mode')
        fc3, fc2, score_dsn6_up, score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse = MODEL2.dss_model(X, y, mode)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, os.path.join(flags.model_dir, flags.model_name))
        print(os.path.join(flags.model_dir, flags.model_name))

        names=os.listdir(flags.input_dir)
        for name in names:
            inputname=os.path.join(flags.input_dir, name)
            print(inputname)
            img = skimage.io.imread(inputname)
            img = skimage.transform.resize(img, (h, w))
            img = skimage.img_as_ubyte(img)

            img = img.astype(np.float32)


            inputgtname = os.path.join(flags.gt_dir, name)
            print(inputgtname)
            label1 = skimage.io.imread(inputgtname)
            label2 = skimage.transform.resize(label1, (h, w))
            label2 = skimage.img_as_ubyte(label2)
            label = (label1 >= (0.5 * 255)).astype(np.float32)

            label_unique = np.unique(label)
            if len(label_unique) == 1 and label_unique[0] == 0:
                curr_image_label = 0
            else:
                curr_image_label = 1

            fc2_value = sess.run(fc2, feed_dict={X: np.expand_dims(img, 0), y: np.expand_dims(np.expand_dims(label2, 0), 3), image_label: np.expand_dims(np.expand_dims(0, 0), 0), mode: False})
            fc2_value = fc2_value[0]
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(fc2_value.tostring()),
                'image_label': _int64_feature(curr_image_label)
            }))
            writer.write(example.SerializeToString())

        writer.close()

if __name__ == '__main__':
    main(flags)