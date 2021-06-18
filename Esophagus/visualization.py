
import os
# import cv2
import glob
import tensorflow as tf
import numpy as np
import argparse
import skimage.io
import MODEL3
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.cm as CM


h = 224
w = 224

model_output = 'result/multi_task_99'
parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir',
                    default='./' + model_output +'/ori_output')

parser.add_argument('--gt_dir',
                    default='./dataset/endoscope/val/label')

parser.add_argument('--model_dir',
                    type=str,
                    default='./model/multi_task')
parser.add_argument('--model_name',
                    type=str,
                    default='model99.ckpt')

parser.add_argument('--input_dir',
                    type=str,
                    default='./dataset/special_test')

parser.add_argument('--save_dir',
                    type=str,
                    default='./result/multi_task_99')
parser.add_argument('--gpu',
                    type=int,
                    default=0)
parser.add_argument('--visualization_output_dir',
                    type=str,
                    default='./result/multi_task_99/visualization')
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

layer_dict = ['conv1_2', 'pool1', 'conv2_2', 'pool2', 'conv3_3', 'pool3', 'conv4_3', 'pool4', 'conv5_3', 'pool5', 'pool6', 'conv3_dsn6', 'conv3_dsn5', 'conv4_dsn4', 'conv4_dsn3', 'conv4_dsn2']

def main(flags):
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    if not os.path.exists(os.path.join(flags.save_dir, 'ori_output')):
        os.mkdir(os.path.join(flags.save_dir, 'ori_output'))
    T = 0.5
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(tf.float32, shape=[None, h, w, 3], name='X')
        y = tf.placeholder(tf.float32, shape=[None, h, w, 1], name='y')
        mode = tf.placeholder(tf.bool, name='mode')

        layers, fc3, score_dsn6_up, score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse = MODEL3.dss_model(X, y, mode)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, os.path.join(flags.model_dir, flags.model_name))

        names = os.listdir(flags.input_dir)

        for name in names:
            inputname = os.path.join(flags.input_dir, name)
            print(inputname)
            if not os.path.exists(os.path.join(flags.visualization_output_dir)):
                os.mkdir(os.path.join(flags.visualization_output_dir))

            if not os.path.exists(os.path.join(flags.visualization_output_dir, name)):
                os.mkdir(os.path.join(flags.visualization_output_dir, name))

            img = skimage.io.imread(inputname)
            img = skimage.transform.resize(img, (h, w))
            img = skimage.img_as_ubyte(img)
            img = img.astype(np.float32)

            inputgtname = os.path.join(flags.gt_dir, name)
            print(inputgtname)
            label1 = skimage.io.imread(inputgtname)
            label2 = skimage.transform.resize(label1, (h, w))
            label2 = skimage.img_as_ubyte(label2)

            if not os.path.exists(flags.visualization_output_dir):
                os.mkdir(flags.visualization_output_dir)

            index = 0
            for layer in layers:
                print(os.path.join(flags.visualization_output_dir, name, layer_dict[index]))
                if index < 11:
                    index += 1
                    continue
                print(index)
                print(layer.name)
                layer_value = sess.run(layer, feed_dict={X: np.expand_dims(img, 0), y: np.expand_dims(np.expand_dims(label2, 0), 3), mode: False})
                layer_value = np.squeeze(layer_value, axis=0)
                for idx in range(0, layer_value.shape[2]):
                    plt.figure(1, figsize=(3, 3), dpi=10)
                    plt.axis('off')
                    plt.imshow(layer_value[:, :, idx], vmin=-1, vmax=1)
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.savefig(os.path.join(flags.visualization_output_dir, name, layer_dict[index] + '.png'))
                index += 1

                plt.close('all')


            layer_value = sess.run(score_dsn6_up, feed_dict={X: np.expand_dims(img, 0), \
                                                     y: np.expand_dims(np.expand_dims(label2, 0), 3), mode: False})
            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(layer_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.colorbar()
            plt.margins(0, 0)
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'score_dsn6_up.png'))
            plt.close('all')

            layer_value = sess.run(score_dsn5_up, feed_dict={X: np.expand_dims(img, 0), \
                                                     y: np.expand_dims(np.expand_dims(label2, 0), 3), mode: False})

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(layer_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.colorbar()
            plt.margins(0, 0)
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'score_dsn5_up.png'))
            plt.close('all')

            layer_value = sess.run(score_dsn4_up, feed_dict={X: np.expand_dims(img, 0), \
                                                     y: np.expand_dims(np.expand_dims(label2, 0), 3), mode: False})

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(layer_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.colorbar()
            plt.margins(0, 0)
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'score_dsn4_up.png'))
            plt.close('all')

            layer_value = sess.run(score_dsn3_up, feed_dict={X: np.expand_dims(img, 0), \
                                                     y: np.expand_dims(np.expand_dims(label2, 0), 3), mode: False})

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(layer_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'score_dsn3_up.png'))
            plt.close('all')

            layer_value = sess.run(score_dsn2_up, feed_dict={X: np.expand_dims(img, 0), \
                                                     y: np.expand_dims(np.expand_dims(label2, 0), 3), mode: False})

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(layer_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'score_dsn2_up.png'))
            plt.close('all')

            layer_value = sess.run(score_dsn1_up, feed_dict={X: np.expand_dims(img, 0), \
                                                     y: np.expand_dims(np.expand_dims(label2, 0), 3), mode: False})

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(layer_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'score_dsn1_up.png'))
            plt.close('all')

            layer_value = sess.run(upscore_fuse, feed_dict={X: np.expand_dims(img, 0), \
                                                     y: np.expand_dims(np.expand_dims(label2, 0), 3), mode: False})

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(layer_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'upscore_fuse.png'))
            plt.close('all')

            score_dsn6_up_value, score_dsn5_up_value, score_dsn4_up_value, score_dsn3_up_value, score_dsn2_up_value, score_dsn1_up_value, upscore_fuse_value = sess.run([score_dsn6_up, score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse],
                                  feed_dict={X: np.expand_dims(img, 0), y: np.expand_dims(np.expand_dims(label2, 0), 3),
                                             mode: False})

            labelh, labelw = label1.shape

            merged = np.squeeze(score_dsn6_up_value)
            merged = skimage.transform.resize(merged, (labelh, labelw))
            label_pred = (merged > T).astype(np.float32)
            label_pred = skimage.img_as_ubyte(label_pred)
            skimage.io.imsave(os.path.join(flags.visualization_output_dir, name, 'score_dsn6_up.png'), label_pred)

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(score_dsn6_up_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'heat_map_score_dsn6_up.png'))
            plt.close('all')

            merged = np.squeeze(score_dsn5_up_value)
            merged = skimage.transform.resize(merged, (labelh, labelw))
            label_pred = (merged > T).astype(np.float32)
            label_pred = skimage.img_as_ubyte(label_pred)
            skimage.io.imsave(os.path.join(flags.visualization_output_dir, name, 'score_dsn5_up.png'), label_pred)

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(score_dsn5_up_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'heat_map_score_dsn5_up.png'))
            plt.close('all')

            merged = np.squeeze(score_dsn4_up_value)
            merged = skimage.transform.resize(merged, (labelh, labelw))
            label_pred = (merged > T).astype(np.float32)
            label_pred = skimage.img_as_ubyte(label_pred)
            skimage.io.imsave(os.path.join(flags.visualization_output_dir, name, 'score_dsn4_up.png'), label_pred)

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(score_dsn4_up_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'heat_map_score_dsn4_up.png'))
            plt.close('all')

            merged = np.squeeze(score_dsn3_up_value)
            merged = skimage.transform.resize(merged, (labelh, labelw))
            label_pred = (merged > T).astype(np.float32)
            label_pred = skimage.img_as_ubyte(label_pred)
            skimage.io.imsave(os.path.join(flags.visualization_output_dir, name, 'score_dsn3_up.png'), label_pred)

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(score_dsn3_up_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'heat_map_score_dsn3_up.png'))
            plt.close('all')

            merged = np.squeeze(score_dsn2_up_value)
            merged = skimage.transform.resize(merged, (labelh, labelw))
            label_pred = (merged > T).astype(np.float32)
            label_pred = skimage.img_as_ubyte(label_pred)
            skimage.io.imsave(os.path.join(flags.visualization_output_dir, name, 'score_dsn2_up.png'), label_pred)

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(score_dsn2_up_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'heat_map_score_dsn2_up.png'))
            plt.close('all')

            merged = np.squeeze(score_dsn1_up_value)
            merged = skimage.transform.resize(merged, (labelh, labelw))
            label_pred = (merged > T).astype(np.float32)
            label_pred = skimage.img_as_ubyte(label_pred)
            skimage.io.imsave(os.path.join(flags.visualization_output_dir, name, 'score_dsn1_up.png'), label_pred)

            plt.figure(1, figsize=(3, 3), dpi=10)
            plt.axis('off')
            plt.imshow(np.squeeze(score_dsn2_up_value), vmin=-1, vmax=1)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            # plt.colorbar()
            plt.savefig(os.path.join(flags.visualization_output_dir, name, 'heat_map_score_dsn1_up.png'))
            plt.close('all')

            merged = np.squeeze(upscore_fuse_value)
            merged = skimage.transform.resize(merged, (labelh, labelw))
            merged = skimage.transform.resize(merged, (labelh, labelw))
            label_pred = (merged > T).astype(np.float32)
            label_pred = skimage.img_as_ubyte(label_pred)
            skimage.io.imsave(os.path.join(flags.visualization_output_dir, name, 'upscore_fuse_value.png'), label_pred)

if __name__ == '__main__':
    main(flags)