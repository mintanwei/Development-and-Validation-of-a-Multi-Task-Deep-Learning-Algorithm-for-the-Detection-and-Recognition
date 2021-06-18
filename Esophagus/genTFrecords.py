import os
import sys
import tensorflow as tf
import skimage.io
import skimage.transform
import random
import numpy as np

h = 300
w = 300

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(unused_argv):
    for id in range(2):
        set = ['train', 'val']
        img_dir_train = './dataset/endoscope/' + set[id]
        filename = os.path.join('data', 'endoscope' + str(h) + str(w) + set[id] +'.tfrecords')
        if(not os.path.exists('data')):
            os.makedirs('data')
        print('writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        fileInfos = []
        list_dirs = os.walk(os.path.join(img_dir_train, 'img'))
        for root, dirs, files in list_dirs:
            for f in files:
                if(os.path.splitext(f)[1] == '.png'):
                    img_path = os.path.join(root, f)
                    fileInfos.append((img_path, os.path.join(img_dir_train, 'label', f)))

        random.shuffle(fileInfos)
        for filePath, fileLabel in fileInfos:
            img = skimage.io.imread(filePath)
            label = skimage.io.imread(fileLabel)
            img = skimage.transform.resize(img, (h, w))
            img = skimage.img_as_ubyte(img)
            label = skimage.transform.resize(label, (h, w))
            label = skimage.img_as_ubyte(label)
            unique_label = np.unique(label)
            image_label = 1
            if len(unique_label) == 1 and unique_label[0] == 0:
                image_label = 0
                # else:
                #     image_label = 1
                # print(label.shape[:])
            print('resize', filePath, img.shape)
            print('resize', fileLabel, img.shape)
            assert img.shape[:] == (h, w, 3)
            assert img.dtype == 'uint8'
            assert label.shape[:] == (h, w)
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _bytes_feature(label.tostring()),
                'image': _bytes_feature(img.tostring()),
                'image_label': _int64_feature(image_label)
            }))
            writer.write(example.SerializeToString())
            print('reading', filePath)
        writer.close()

if __name__ == '__main__':
    tf.app.run()
