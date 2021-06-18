import os
# import cv2
import glob
import tensorflow as tf
import numpy as np
import argparse
import skimage.io
import MODEL
import skimage.transform
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector

tag = 'last_fc'
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    type=str,
                    default='./dataset/last_fc.tfrecords')

parser.add_argument('--metadata_dir',
                    type=str,
                    default='./result/multi_task_99/visualization_high_dimension' + tag + '/metadata.tsv')

parser.add_argument('--visualization_output_dir',
                    type=str,
                    default='./result/multi_task_99/visualization_high_dimension' + tag)
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
    return intersection /(h * w)


def classification(gt):
    gt_label = np.unique(gt)
    group = 1
    if len(gt_label) == 1:
        if gt_label == 0:
            group = 0
    return group

def get_metadata(gt_dir, names):
    print('save metadata...')
    with open(flags.metadata_dir, 'w') as metadata_file:
        for name in names:
            inputgtname = os.path.join(gt_dir, name)
            # print(inputgtname)
            label1 = skimage.io.imread(inputgtname)
            label2 = skimage.transform.resize(label1, (h, w))

            image_label = classification(label2)
            metadata_file.write('%d\n' % image_label)
    # return flags.metadata_dir

def read_data(filename):
    print(filename)
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        }
    )

    image = tf.decode_raw(features['image'], tf.float32)
    image_label = features['image_label']

    image = tf.reshape(image, [1, 1024])
    # image = tf.image.resize_images(image, (newh, neww), method=2)
    image = tf.cast(image, tf.float32)
    return image, image_label

def batch_data():
    # print(os.path.join(FLAGS.data_dir, set_name + '.tfrecords'))
    print(flags.data_dir)
    image, image_label = read_data(flags.data_dir)
    images, image_labels = tf.train.shuffle_batch(
        [image, image_label], batch_size=1, num_threads=1,
        capacity=1,
        min_after_dequeue=0,
        allow_smaller_final_batch=True)
    return images, image_labels

def read_data2(names):
    images = []
    fc2, label = batch_data()

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            with open(flags.metadata_dir, 'w') as metadata_file:
                for i in range(len(names)):
                    fc2_value, label_value = sess.run(
                        [fc2, label])
                    metadata_file.write('%d\n' % label_value)
                    images.append(fc2_value.reshape(-1))
                    print(i)
        finally:
            coord.request_stop()
            coord.join(threads)

    return np.array(images)


def main(flags):
    if not os.path.exists(flags.visualization_output_dir):
        os.mkdir(flags.visualization_output_dir)

    g = tf.Graph()
    with g.as_default():
        names = os.listdir('./dataset/endoscope/val/img')
        data = read_data2(names)

        images = tf.Variable(data, name='images')


        with tf.Session() as sess:
            saver = tf.train.Saver([images])

            sess.run(tf.global_variables_initializer())
            saver.save(sess, os.path.join(flags.visualization_output_dir, 'images.ckpt'))

            config = projector.ProjectorConfig()
            # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = images.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = flags.metadata_dir
            # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(tf.summary.FileWriter(flags.visualization_output_dir), config)


        print('the end')


if __name__ == '__main__':
    main(flags)
