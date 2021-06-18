import os
import tensorflow as tf
import numpy as np
import argparse
import MODEL
import time
import datetime

from MODEL import train_op
from MODEL import loss_IOU
from MODEL import loss_CE
from MODEL import loss_Pre
from MODEL import class_balanced_cross_entropy_loss
from MODEL import focal_loss
from MODEL import focal_loss_class_balanced
from MODEL import loss_Class
from MODEL import loss_Pre_class
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python import pywrap_tensorflow

h = 300
w = 300
newh = 224
neww = 224
c_image = 3
c_label = 1

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',
                    default='data')

parser.add_argument('--test_dir',
                    default='data')

parser.add_argument('--model_dir',
                    default='model/multi_task_0728')

parser.add_argument('--restore_dir',
                    default='model/multi_task/model99.ckpt')

parser.add_argument('--restore',
                    default=True
                    )

parser.add_argument('--epochs',
                    type=int,
                    default=100)

parser.add_argument('--peochs_per_eval',
                    type=int,
                    default=1)

parser.add_argument('--logdir',
                    default='model/multi_task_0728/logs')

parser.add_argument('--batch_size',
                    type=int,
                    default=12)

parser.add_argument('--is_cross_entropy',
                    action='store_true',
                    default=True)

parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-4)

parser.add_argument('--decay_rate',
                    type=float,
                    default=0.9)


parser.add_argument('--decay_step',
                    type=int,
                    default=2)

parser.add_argument('--weight',
                    nargs='+',
                    type=float,
                    default=[1.0,1.0])

parser.add_argument('--random_seed',
                    type=int,
                    default=1234)

flags = parser.parse_args()

def data_augmentation(image,label,training=True):
    if training:
        image_label = tf.concat([image,label],axis = -1)
        print('image label shape concat',image_label.get_shape())

        maybe_flipped = tf.image.random_flip_left_right(image_label)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)
        maybe_flipped = tf.random_crop(maybe_flipped,size=[newh, neww, image_label.get_shape()[-1]])

        image = maybe_flipped[:, :, :-1]
        mask = maybe_flipped[:, :, -1:]

        return image, mask

def read_data(filename, augmentation=True):
    print(filename)
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_label': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }
    )

    label = features['label']
    image = features['image']
    image_label = features['image_label']

    image = tf.decode_raw(image, tf.uint8)
    image_shape = tf.stack([h, w, c_image])

    image = tf.reshape(image, image_shape)

    image = tf.cast(image, tf.float32)

    label = tf.decode_raw(label, tf.uint8)
    label_shape = tf.stack([h, w, c_label])
    label = tf.reshape(label, label_shape)

    label = tf.cast(tf.greater(label, 128), tf.float32)

    image_label = tf.cast(image_label, tf.float32)

    if augmentation:
        image, label = data_augmentation(image,label)
    else:
        pass

    return image, label, image_label

def batch_data(set_name):
    print(flags.data_dir, set_name + '.tfrecords')
    image, label, image_label = read_data(os.path.join(flags.data_dir, set_name + '.tfrecords'))
    images, labels, image_labels = tf.train.shuffle_batch(
        [image, label, image_label], batch_size=flags.batch_size, num_threads=4,
        capacity=32 * flags.batch_size,
        min_after_dequeue=16 * flags.batch_size,
        allow_smaller_final_batch=True)
    return images, labels, image_labels

def main(args):
    datasets = 'endoscope_ori300300train'
    test = 'endoscope_ori300300val'
    num_train = 2428
    num_test = 194
    max_pre = 0
    max_pre_class = 0
    with tf.Graph().as_default():
        train_logdir = os.path.join(flags.logdir, datasets)
        test_logdir = os.path.join(flags.logdir, test)

        if not os.path.exists(flags.model_dir):
            os.mkdir(flags.model_dir)

        if not os.path.exists(flags.logdir):
            os.mkdir(flags.logdir)

        if not os.path.exists(train_logdir):
            os.mkdir(train_logdir)
        if not os.path.exists(test_logdir):
            os.mkdir(test_logdir)

        X_train_batch_op, y_train_batch_op, label_train_batch_op = batch_data(datasets)
        X_test_batch_op, y_test_batch_op, label_test_batch_op = batch_data(test)
        X = tf.placeholder(tf.float32, shape=[None, newh, neww, 3], name='X')
        y = tf.placeholder(tf.float32, shape=[None, newh, neww, 1], name='y')
        image_label = tf.placeholder(tf.float32, shape=[None, 1], name='image_label')
        mode = tf.placeholder(tf.bool, name='mode')

        fc3, score_dsn6_up, score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse = MODEL.dss_model(X, y, mode)

        loss6 = focal_loss_class_balanced(score_dsn6_up, y)
        loss5 = focal_loss_class_balanced(score_dsn5_up, y)
        loss4 = focal_loss_class_balanced(score_dsn4_up, y)
        loss3 = focal_loss_class_balanced(score_dsn3_up, y)
        loss2 = focal_loss_class_balanced(score_dsn2_up, y)
        loss1 = focal_loss_class_balanced(score_dsn1_up, y)
        loss_fuse = focal_loss_class_balanced(upscore_fuse, y)

        loss_class = focal_loss(fc3, image_label)
        tf.summary.scalar("class_loss", loss_class)

        tf.summary.scalar("CE6", loss6)
        tf.summary.scalar("CE5", loss5)
        tf.summary.scalar("CE4", loss4)
        tf.summary.scalar("CE3", loss3)
        tf.summary.scalar("CE2", loss2)
        tf.summary.scalar("CE1", loss1)
        tf.summary.scalar("CE_fuse", loss_fuse)


        Loss = loss6 + loss5 + loss4 + loss3 + loss2 + 2 * loss1 + loss_fuse + 224 * 224 * loss_class

        tf.summary.scalar("CE_total", Loss)

        global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step,
                                                   tf.cast(num_train / flags.batch_size * flags.decay_step, tf.int32),
                                                   flags.decay_rate, staircase=True)

        var_list = tf.trainable_variables()
        with tf.control_dependencies(update_ops):
            training_op = train_op(Loss, learning_rate, var_list=var_list)


        print('Shuffle batch done')

        pred = tf.cast(tf.greater(upscore_fuse, 0.5), tf.float32)
        pred_class = tf.cast(tf.greater(tf.sigmoid(fc3), 0.5), tf.float32)

        precision = loss_Pre(pred, y)
        precision_class = loss_Pre_class(pred_class, image_label)

        tf.summary.scalar("precision", precision)
        tf.summary.scalar("precision_class", precision_class)

        tf.add_to_collection('inputs', X)
        tf.add_to_collection('inputs', mode)
        tf.add_to_collection('score_dsn6_up', score_dsn6_up)
        tf.add_to_collection('score_dsn5_up', score_dsn5_up)
        tf.add_to_collection('score_dsn4_up', score_dsn4_up)
        tf.add_to_collection('score_dsn3_up', score_dsn3_up)
        tf.add_to_collection('score_dsn2_up', score_dsn2_up)
        tf.add_to_collection('score_dsn1_up', score_dsn1_up)
        tf.add_to_collection('upscore_fuse', upscore_fuse)

        tf.summary.scalar("learning_rate", learning_rate)

        summary_op = tf.summary.merge_all()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            print(train_logdir)
            train_writer = tf.summary.FileWriter(train_logdir, sess.graph)
            test_writer = tf.summary.FileWriter(test_logdir)

            saver = tf.train.Saver(max_to_keep=50)

            var_list = tf.trainable_variables()
            # var_list1 = var_list[26:169]
            var_list1 = var_list
            saver2 = tf.train.Saver(var_list=var_list1)

            init = tf.global_variables_initializer()
            sess.run(init)

            if flags.restore:
                saver2.restore(sess, flags.restore_dir)
                print('restore ', flags.restore_dir)
            else:
                print('No model')

            try:
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                all_acc_train = []
                all_acc_train_class = []
                for epoch in range(flags.epochs):
                    for step_train in range(0,num_train,flags.batch_size):
                        train_img, train_label, train_img_label = sess.run([X_train_batch_op, y_train_batch_op, label_train_batch_op])
                        train_img_label = np.expand_dims(train_img_label, 1)
                        _, step_ce, step_summary, global_step_value, pre_value, pre_class_value = sess.run([training_op, Loss, summary_op, global_step, precision, precision_class], feed_dict={X: train_img, y: train_label, image_label: train_img_label, mode: True})

                        all_acc_train.append(pre_value)
                        all_acc_train_class.append(pre_class_value)
                        train_writer.add_summary(step_summary, global_step_value)
                        if global_step_value % 100 == 0:
                            print('global_step:{} loss:{} precision:{} class precision:{} time:{}'.format(global_step_value, step_ce, np.mean(all_acc_train), np.mean(all_acc_train_class), datetime.datetime.now()))
                            all_acc_train = []
                            all_acc_train_class = []
                    all_acc_test = []
                    all_acc_test_class = []
                    for step_test in range(0,num_test,flags.batch_size):
                        test_img, test_label, test_img_label = sess.run([X_test_batch_op, y_test_batch_op, label_test_batch_op])
                        test_img_label = np.expand_dims(test_img_label, 1)
                        loss_value,acc_test, acc_class_test, step_summary = sess.run([Loss, precision, precision_class, summary_op], feed_dict={X: test_img, y: test_label, image_label: test_img_label, mode: False})
                        all_acc_test.append(acc_test)
                        all_acc_test_class.append(acc_class_test)
                        test_writer.add_summary(step_summary, epoch * (
                            num_train // flags.batch_size) + step_test // flags.batch_size * num_train // num_test)
                    print('epoch:{} Test precision:{} Class precision{}'.format(epoch, np.mean(all_acc_test), np.mean(all_acc_test_class)))
                    saver.save(sess, '{}/model{}.ckpt'.format(flags.model_dir, epoch))

                    if np.mean(all_acc_test) >= max_pre:
                        max_pre = np.mean(all_acc_test)
                        print('detection optimum {} / {}'.format(max_pre, np.mean(all_acc_test_class)))
                    if np.mean(all_acc_test_class) >= max_pre_class:
                        max_pre_class = np.mean(all_acc_test_class)
                        print('classification optimum {} / {}'.format(np.mean(all_acc_test), max_pre_class))
            finally:
                coord.request_stop()
                coord.join(threads)
                saver.save(sess, "{}/model.ckpt".format(flags.model_dir))


if __name__ == '__main__':
    tf.app.run()
