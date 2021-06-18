import os
import tensorflow as tf
import numpy as np

learning_rate=0.001
num_class=2
loss_weight = np.array([1,1])

h = 224
w = 224

g_mean = [162.23,91.74,78.50]
VGG_MEAN = [103.939, 116.779, 123.68]

vgg16_npy_path = "../vgg16.npy"

data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

def conv2d(x,weight,n_filters,training,name,activation=tf.nn.relu):
    with tf.variable_scope('layer{}'.format(name)):
        for index,filter in enumerate(n_filters):
            conv = tf.layers.conv2d(x,filter,weight,strides=1,padding='same',activation=activation,name='conv_{}'.format(index+1))
            conv = tf.layers.batch_normalization(conv,training=training,name='bn_{}'.format(index+1))

        if activation == None:
            return conv

        conv = activation(conv,name='relu{}_{}'.format(name,index+1))

        return conv

def pool2d(x,pool_size,pool_stride,name):
    pool = tf.layers.max_pooling2d(x, pool_size, pool_stride, name='pool_{}'.format(name),padding='same')
    return pool

def deconv2d(x,kernel,strides,training,name,output_shape,activation=None):
    kernel_shape=[kernel,kernel,1,1]
    strides=[1,strides,strides,1]
    kernel=tf.get_variable('weight_{}'.format(name),shape=kernel_shape,initializer=tf.random_normal_initializer(mean=0,stddev=1))
    deconv = tf.nn.conv2d_transpose(x, kernel, strides=strides,output_shape=output_shape, padding='SAME',
                                        name= 'upsample_{}'.format(name))
    deconv = tf.layers.batch_normalization(deconv, training=training, name='bn{}'.format(name))
    if activation == None:
        return deconv

    deconv = activation(deconv, name='sigmoid_{}'.format(name))

    return deconv


def upsampling_2d(tensor,name,size=(2,2)):
    h_,w_,c_ = tensor.get_shape().as_list()[1:]
    h_multi,w_multi = size
    h = h_multi * h_
    w = w_multi * w_
    target = tf.image.resize_nearest_neighbor(tensor,size=(h,w),name='deconv_{}'.format(name))

    return target

def upsampling_concat(input_A,input_B,name):
    upsampling = upsampling_2d(input_A,name=name,size=(2,2))
    up_concat = tf.concat([upsampling,input_B],axis=-1,name='up_concat_{}'.format(name))
    return up_concat

def fc_layer(bottom, output, name, act=tf.nn.relu):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])
        shape_W = [dim, output]
        initial_W = tf.truncated_normal(shape_W, stddev=0.1, dtype=tf.float32)
        W = tf.Variable(initial_W, dtype=tf.float32)
        shape_b = [output]
        initial_b = tf.constant(0, shape=shape_b, dtype=tf.float32)
        b = tf.Variable(initial_b, dtype=tf.float32)
    return act(tf.matmul(x, W) + b)


def conv_layer(bottom, name, trainable=True):
    with tf.variable_scope(name):
        filt = get_conv_filter(name, trainable)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name, trainable)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def get_conv_filter(name, trainable):
    return tf.Variable(data_dict[name][0], trainable=trainable, name="filter")

def get_bias(name, trainable):
    return tf.Variable(data_dict[name][1], trainable=trainable, name="biases")

def dss_model(input, y, training):
    # Convert RGB to BGR
    red, green, blue = tf.split(input, 3, 3)
    assert red.get_shape().as_list()[1:] == [h, w, 1]
    assert green.get_shape().as_list()[1:] == [h, w, 1]
    assert blue.get_shape().as_list()[1:] == [h, w, 1]
    bgr = tf.concat([
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2]
    ], 3)
    assert bgr.get_shape().as_list()[1:] == [h, w, 3]

    output_shape = tf.shape(y)

    layers = []

    with tf.variable_scope('saliency', reuse=tf.AUTO_REUSE):
        # conv1_1 = conv2d(bgr, (3, 3), [64], training, name='conv1_1')
        # conv1_2 = conv2d(conv1_1, (3, 3), [64], training, name='conv1_2')
        # pool1 = pool2d(conv1_2,pool_size=(2,2),pool_stride=2,name='pool1')
        conv1_1 = conv_layer(bgr, "conv1_1")
        conv1_2 = conv_layer(conv1_1, "conv1_2")
        pool1 = max_pool(conv1_2, 'pool1')
        print('conv1_2', conv1_2.get_shape().as_list())
        print('pool1', pool1.get_shape().as_list())
        layers.append(conv1_2)
        layers.append(pool1)

        # conv2_1=conv2d(pool1, (3, 3), [128], training, name='conv2_1')
        # conv2_2=conv2d(conv2_1, (3, 3), [128], training, name='conv2_2')
        # pool2 = pool2d(conv2_2,pool_size=(2,2),pool_stride=2,name='pool2')
        conv2_1 = conv_layer(pool1, "conv2_1")
        conv2_2 = conv_layer(conv2_1, "conv2_2")
        pool2 = max_pool(conv2_2, 'pool2')
        print('conv2_2', conv2_2.get_shape().as_list())
        print('pool2', pool2.get_shape().as_list())
        layers.append(conv2_2)
        layers.append(pool2)

        # conv3_1 = conv2d(pool2, (3, 3), [256], training, name='conv3_1')
        # conv3_2 = conv2d(conv3_1, (3, 3), [256], training, name='conv3_2')
        # conv3_3 = conv2d(conv3_2, (3, 3), [256], training, name='conv3_3')
        # pool3 = pool2d(conv3_3, pool_size=(2, 2), pool_stride=2, name='pool3')

        conv3_1 = conv_layer(pool2, "conv3_1")
        conv3_2 = conv_layer(conv3_1, "conv3_2")
        conv3_3 = conv_layer(conv3_2, "conv3_3")
        pool3 = max_pool(conv3_3, 'pool3')
        print('conv3_3', conv3_3.get_shape().as_list())
        print('pool3', pool3.get_shape().as_list())

        layers.append(conv3_3)
        layers.append(pool3)

        # conv4_1 = conv2d(pool3, (3, 3), [512], training, name='conv4_1')
        # conv4_2 = conv2d(conv4_1, (3, 3), [512], training, name='conv4_2')
        # conv4_3 = conv2d(conv4_2, (3, 3), [512], training, name='conv4_3')
        # pool4 = pool2d(conv4_3, pool_size=(2, 2), pool_stride=2, name='pool4')
        conv4_1 = conv_layer(pool3, "conv4_1")
        conv4_2 = conv_layer(conv4_1, "conv4_2")
        conv4_3 = conv_layer(conv4_2, "conv4_3")
        pool4 = max_pool(conv4_3, 'pool4')
        print('conv4_3', conv4_3.get_shape().as_list())
        print('pool4', pool4.get_shape().as_list())
        layers.append(conv4_3)
        layers.append(pool4)

        # conv5_1 = conv2d(pool4, (3, 3), [512], training, name='conv5_1')
        # conv5_2 = conv2d(conv5_1, (3, 3), [512], training, name='conv5_2')
        # conv5_3 = conv2d(conv5_2, (3, 3), [512], training, name='conv5_3')
        # pool5 = pool2d(conv5_3, pool_size=(3, 3), pool_stride=2, name='pool5')
        conv5_1 = conv_layer(pool4, "conv5_1")
        conv5_2 = conv_layer(conv5_1, "conv5_2")
        conv5_3 = conv_layer(conv5_2, "conv5_3")
        pool5 = max_pool(conv5_3, 'pool5')

        print('conv5_3', conv5_3.get_shape().as_list())
        print('pool5', pool5.get_shape().as_list())
        layers.append(conv5_3)
        layers.append(pool5)

        pool6 = pool2d(pool5, pool_size=(3, 3), pool_stride=1, name='pool5a')
        layers.append(pool6)

        conv1_dsn6 = conv2d(pool6, (7, 7), [512], training, name='conv1-dsn6')
        conv2_dsn6 = conv2d(conv1_dsn6, (7, 7), [512], training, name='conv2-dsn6')
        conv3_dsn6 = conv2d(conv2_dsn6, (1, 1), [1], training, name='conv3-dsn6',activation=None)
        score_dsn6_up = deconv2d(conv3_dsn6,64,32,training,name='upsample32_in_dsn6_sigmoid-dsn6',output_shape=output_shape,activation=tf.nn.sigmoid)
        layers.append(conv3_dsn6)

        conv1_dsn5 = conv2d(conv5_3, (5, 5), [512], training, name='conv1_dsn5')
        conv2_dsn5 = conv2d(conv1_dsn5, (5, 5), [512], training, name='conv2-dsn5')
        conv3_dsn5 = conv2d(conv2_dsn5, (1, 1), [1], training, name='conv3-dsn5',activation=None)
        score_dsn5_up = deconv2d(conv3_dsn5, 32, 16, training, name='upsample16_in_dsn5_sigmoid-dsn5',output_shape=output_shape,activation=tf.nn.sigmoid)
        layers.append(conv3_dsn5)

        conv1_dsn4 = conv2d(conv4_3, (5, 5), [256], training, name='conv1-dsn4')
        conv2_dsn4 = conv2d(conv1_dsn4, (5, 5), [256], training, name='conv2-dsn4')
        conv3_dsn4 = conv2d(conv2_dsn4, (1, 1), [1], training, name='conv3-dsn4',activation=None)
        score_dsn6_up_4=deconv2d(conv3_dsn6,8,4,training,name='upsample4_dsn6',output_shape=tf.shape(conv3_dsn4))
        score_dsn5_up_4=deconv2d(conv3_dsn5,4,2,training,name='upsample2_dsn5',output_shape=tf.shape(conv3_dsn4))
        concat_dsn4=tf.concat([score_dsn6_up_4,score_dsn5_up_4,conv3_dsn4],axis=-1,name='concat_dsn4')
        conv4_dsn4 = conv2d(concat_dsn4, (1, 1), [1], training, name='conv4-dsn4',activation=None)
        score_dsn4_up = deconv2d(conv4_dsn4, 16, 8, training, name='upsample8_in_dsn4_sigmoid-dsn4',output_shape=output_shape,activation=tf.nn.sigmoid)
        layers.append(conv4_dsn4)

        conv1_dsn3 = conv2d(conv3_3, (5, 5), [256], training, name='conv1-dsn3')
        conv2_dsn3 = conv2d(conv1_dsn3, (5, 5), [256], training, name='conv2-dsn3')
        conv3_dsn3 = conv2d(conv2_dsn3, (1, 1), [1], training, name='conv3-dsn3')
        score_dsn6_up_3 = deconv2d(conv3_dsn6, 16, 8, training, name='upsample8_dsn6',output_shape=tf.shape(conv3_dsn3))
        score_dsn5_up_3 = deconv2d(conv3_dsn5, 8, 4, training, name='upsample4_dsn5',output_shape=tf.shape(conv3_dsn3))
        concat_dsn3 = tf.concat([score_dsn6_up_3, score_dsn5_up_3, conv3_dsn3], axis=-1, name='concat_dsn3')
        conv4_dsn3 = conv2d(concat_dsn3, (1, 1), [1], training, name='conv4-dsn3',activation=None)
        score_dsn3_up = deconv2d(conv4_dsn3, 8, 4, training, name='upsample4_in_dsn3_sigmoid-dsn3',output_shape=output_shape,activation=tf.nn.sigmoid)
        layers.append(conv4_dsn3)

        conv1_dsn2 = conv2d(conv2_2, (3, 3), [128], training, name='conv1-dsn2')
        conv2_dsn2 = conv2d(conv1_dsn2, (3, 3), [128], training, name='conv2-dsn2')
        conv3_dsn2 = conv2d(conv2_dsn2, (1, 1), [1], training, name='conv3-dsn2')
        score_dsn6_up_2 = deconv2d(conv3_dsn6, 32, 16, training, name='upsample16_dsn6',output_shape=tf.shape(conv3_dsn2))
        score_dsn5_up_2 = deconv2d(conv3_dsn5, 16,8, training, name='upsample8_dsn5',output_shape=tf.shape(conv3_dsn2))
        score_dsn4_up_2 = deconv2d(conv3_dsn4, 8, 4, training, name='upsample4_dsn4',output_shape=tf.shape(conv3_dsn2))
        score_dsn3_up_2 = deconv2d(conv3_dsn3, 4, 2, training, name='upsample2_dsn3',output_shape=tf.shape(conv3_dsn2))
        concat_dsn2 = tf.concat([score_dsn6_up_2, score_dsn5_up_2,score_dsn4_up_2,score_dsn3_up_2, conv3_dsn2], axis=-1, name='concat_dsn2')
        conv4_dsn2 = conv2d(concat_dsn2, (1, 1), [1], training, name='conv4-dsn2',activation=None)
        score_dsn2_up = deconv2d(conv4_dsn2, 4, 2, training, name='upsample2_in_dsn2_sigmoid-dsn2',output_shape=output_shape,activation=tf.nn.sigmoid)
        layers.append(conv4_dsn2)

        conv1_dsn1 = conv2d(conv1_2, (3, 3), [128], training, name='conv1-dsn1')
        conv2_dsn1 = conv2d(conv1_dsn1, (3, 3), [128], training, name='conv2-dsn1')
        conv3_dsn1 = conv2d(conv2_dsn1, (1, 1), [1], training, name='conv3-dsn1',activation=None)
        score_dsn6_up_1 = deconv2d(conv3_dsn6, 64, 32, training, name='upsample32_dsn6',output_shape=tf.shape(conv3_dsn1))
        score_dsn5_up_1 = deconv2d(conv3_dsn5, 32, 16, training, name='upsample16_dsn5',output_shape=tf.shape(conv3_dsn1))
        score_dsn4_up_1 = deconv2d(conv3_dsn4, 16, 8, training, name='upsample8_dsn4',output_shape=tf.shape(conv3_dsn1))
        score_dsn3_up_1 = deconv2d(conv3_dsn3, 8, 4, training, name='upsample4_dsn3',output_shape=tf.shape(conv3_dsn1))
        concat_dsn1 = tf.concat([score_dsn6_up_1, score_dsn5_up_1, score_dsn4_up_1, score_dsn3_up_1, conv3_dsn1], axis=-1,
                                name='concat_dsn1')
        score_dsn1_up = conv2d(concat_dsn1, (1, 1), [1], training, name='conv4-dsn1',activation=tf.nn.sigmoid)

        concat_upscore = tf.concat([score_dsn6_up,score_dsn5_up,score_dsn4_up,score_dsn3_up,score_dsn2_up,score_dsn1_up],
                                   axis=-1,name='concat')
        upscore_fuse = conv2d(concat_upscore,(1,1),[1],training,name='new-score-weighting',activation=tf.nn.sigmoid)

        pool5_global = tf.reduce_mean(pool5, [1, 2])
        print(pool5_global.get_shape().as_list())
        fc1 = fc_layer(pool5_global, 4096, name='fc1', act=tf.nn.relu)
        if training == True:
            fc1 = tf.nn.dropout(fc1, 0.5)
        fc2 = fc_layer(fc1, 1024, name='fc2', act=tf.nn.relu)
        # if training == True:
        #     fc2 = tf.nn.dropout(fc2, 0.5)
        fc3 = fc_layer(fc2, 1, name='fc3', act=tf.identity)




        # conv1_dsn4_classification = conv2d(conv4_3, (5, 5), [256], training, name='conv1-dsn4-classification')
        # conv2_dsn4_classification = conv2d(conv1_dsn4_classification, (5, 5), [256], training, name='conv2-dsn4-classification')
        # conv3_dsn4_classification = conv2d(conv2_dsn4_classification, (1, 1), [1], training, name='conv3-dsn4-classification',activation=None)
        #
        # conv1_dsn3_classification = conv2d(conv3_3, (5, 5), [256], training, name='conv1-dsn3-classification')
        # conv2_dsn3_classification = conv2d(conv1_dsn3_classification, (5, 5), [256], training, name='conv2-dsn3-classification')
        # conv3_dsn3_classification = conv2d(conv2_dsn3_classification, (1, 1), [1], training, name='conv3-dsn3-classification')
        #
        # conv1_dsn2_classification = conv2d(conv2_2, (3, 3), [128], training, name='conv1-dsn2-classification')
        # conv2_dsn2_classification = conv2d(conv1_dsn2_classification, (3, 3), [128], training, name='conv2-dsn2-classification')
        # conv3_dsn2_classification = conv2d(conv2_dsn2_classification, (1, 1), [1], training, name='conv3-dsn2-classification')
        #
        # conv1_dsn1_classification = conv2d(conv1_2, (3, 3), [128], training, name='conv1-dsn1-classification')
        # conv2_dsn1_classification = conv2d(conv1_dsn1_classification, (3, 3), [128], training, name='conv2-dsn1-classification')
        # conv3_dsn1_classification = conv2d(conv2_dsn1_classification, (1, 1), [1], training, name='conv3-dsn1-classification',activation=None)

        #score_dsn6_up_1 = deconv2d(conv3_dsn6_classification, 64, 32, training, name='upsample32_dsn6',output_shape=tf.shape(conv3_dsn1))
        # print(fc3.get_shape().as_list())
        # print(score_dsn6_up.get_shape().as_list())
        # print(score_dsn1_up.get_shape().as_list())





        # if training == True:
        #     image_label = tf.cast(image_label, tf.float32)
        #     temp = tf.expand_dims(image_label, 2)
        # else:
        #     temp = tf.expand_dims(fc3, 2)
        #
        # temp = tf.expand_dims(temp, 3)
        # score_dsn6_up = tf.multiply(score_dsn6_up, temp)
        # score_dsn5_up = tf.multiply(score_dsn5_up, temp)
        # score_dsn4_up = tf.multiply(score_dsn4_up, temp)
        # score_dsn3_up = tf.multiply(score_dsn3_up, temp)
        # score_dsn2_up = tf.multiply(score_dsn2_up, temp)
        # score_dsn1_up = tf.multiply(score_dsn1_up, temp)
        # upscore_fuse = tf.multiply(upscore_fuse, temp)



        return layers, fc3, score_dsn6_up, score_dsn5_up, score_dsn4_up, score_dsn3_up, score_dsn2_up, score_dsn1_up, upscore_fuse


def loss_Class(y_pred, y_true):
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    return cross_entropy_mean

def loss_CE(y_pred,y_true):
    '''flat_logits = tf.reshape(y_pred,[-1,num_class])
    flat_labels = tf.reshape(y_true,[-1,num_class])
    class_weights = tf.constant(loss_weight,dtype=np.float32)
    weight_map = tf.multiply(flat_labels,class_weights)
    weight_map = tf.reduce_sum(weight_map,axis=1)

    loss_map = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels,logits=flat_logits)

    weighted_loss = tf.multiply(loss_map,weight_map)

    cross_entropy_mean = tf.reduce_mean(weighted_loss)'''
    #cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_true*tf.log(y_pred)))

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #将x的数据格式转化成dtype
    '''labels = tf.cast(y_true,tf.int32)

    flat_logits = tf.reshape(y_pred,(-1,num_class))

    epsilon = tf.constant(value=1e-10)

    flat_logits=tf.add(flat_logits,epsilon)
    tf.shape(flat_logits, name='flat_logits')

    flat_labels = tf.reshape(labels,(-1,1))
    tf.shape(flat_labels,name='flat_shape1')

    labels = tf.reshape(tf.one_hot(flat_labels,depth=num_class),[-1,num_class])
    tf.shape(flat_labels, name='flat_shape2')

    softmax = tf.nn.softmax(flat_logits)
    tf.shape(softmax, name='softmaxx')
    cross_entropy = -tf.reduce_sum(tf.multiply(labels*tf.log(softmax+epsilon),loss_weight),axis=[1])
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')'''

    return cross_entropy_mean

#IOU损失
def loss_IOU(y_pred,y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]
    flat_logits = tf.reshape(y_pred, [-1, H * W])
    flat_labels = tf.reshape(y_true, [-1, H * W])
    intersection = tf.reduce_sum(flat_logits * flat_labels, axis=1) #沿着第一维相乘求和
    denominator = tf.reduce_sum(flat_logits, axis=1) + tf.reduce_sum(flat_labels, axis=1) - intersection
    iou = tf.reduce_mean((intersection + 1e-7) / (denominator + 1e-7))

    return iou

def loss_Pre(pred, label):
    H, W, _ = pred.get_shape().as_list()[1:]
    flat_logits = tf.reshape(pred, [-1, H * W])
    flat_labels = tf.reshape(label, [-1, H * W])
    return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(flat_logits, flat_labels), tf.float32), axis= 1) / (H * W))
    # intersection = tf.reduce_sum(flat_logits * flat_labels, axis=1) #沿着第一维相乘求和
    # denominator = tf.reduce_sum(flat_logits, axis=1) + tf.reduce_sum(flat_labels, axis=1) - intersection
    # iou = tf.reduce_mean((intersection + 1e-7) / (denominator + 1e-7))

    # return iou
def loss_Pre_class(pred, label):
    # print(pred.get_shape())
    # num = pred.get_shape().as_list()[0]
    flat_logits = tf.reshape(pred, [-1, tf.shape(pred)[0]])
    flat_labels = tf.reshape(label, [-1, tf.shape(pred)[0]])
    return tf.reduce_mean(tf.cast(tf.equal(flat_logits, flat_labels), tf.float32))


def focal_loss(output, label):
    label = tf.cast(tf.greater(label, 0.5), tf.float32)
    p = tf.sigmoid(output)
    pos_p = tf.multiply(p, label)
    neg_p = tf.multiply((1.0 - p), (1.0 - label))
    sum_p = pos_p + neg_p
    final_p = tf.clip_by_value(sum_p, 1e-12, (1.0 - 1e-12))
    final_log = tf.multiply(-1.0, tf.log(final_p))
    final_loss = tf.multiply(0.25, tf.multiply((1.0 - final_p) ** 2, final_log))
    return tf.reduce_sum(final_loss)

def focal_loss_class_balanced(output, label):
    label = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(label)
    num_labels_neg = tf.reduce_sum(1.0 - label)
    num_total = num_labels_pos + num_labels_neg

    p = tf.sigmoid(output)
    # p = output
    pos = tf.multiply(p, label)
    neg = tf.multiply((1.0 - p), (1.0 - label))
    final_p = pos + neg
    pos_p = tf.clip_by_value(pos, 1e-12, (1.0 - 1e-12))
    neg_p = tf.clip_by_value(neg, 1e-12, (1.0 - 1e-12))
    pos_loss = tf.multiply(tf.multiply(-1.0, tf.log(pos_p)), label)
    neg_loss = tf.multiply(tf.multiply(-1.0, tf.log(neg_p)), (1.0 - label))

    # neg_ratio = tf.cond(num_labels_pos > 0.0, lambda: num_labels_pos / num_total, lambda: 0.5)
    # pos_ratio = tf.cond(num_labels_neg > 0.0, lambda: num_labels_neg / num_total, lambda: 0.5)
    pos_ratio = num_labels_neg / num_total
    neg_ratio = num_labels_pos / num_total
    sum_loss = pos_ratio * pos_loss + neg_ratio * neg_loss

    final_loss = tf.multiply((1.0 - final_p) ** 2, sum_loss)
    return tf.reduce_sum(final_loss)

# def class_loss(output, label):
#     label = tf.cast(tf.greater(label, 0.5), tf.float32)
#     p = output
#     pos = tf.multiply(p, label)
#     neg = tf.multiply((1.0 - p), (1.0 - label))
#     final_p = pos + neg


def class_balanced_cross_entropy_loss(output, label):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    Returns:
    Tensor that evaluates the loss
    """

    labels = tf.cast(tf.greater(label, 0.5), tf.float32)

    num_labels_pos = tf.reduce_sum(labels)
    num_labels_neg = tf.reduce_sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = tf.cast(tf.greater_equal(output, 0), tf.float32)
    loss_val = tf.multiply(output, (labels - output_gt_zero)) - tf.log(
        1 + tf.exp(output - 2 * tf.multiply(output, output_gt_zero)))

    loss_pos = tf.reduce_mean(-tf.multiply(labels, loss_val))
    loss_neg = tf.reduce_mean(-tf.multiply(1.0 - labels, loss_val))

    final_loss = 2.0 * num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    return final_loss

def train_op(loss, learning_rate, var_list):

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    return optimizer.minimize(loss,global_step=global_step,var_list=var_list)

if __name__ == '__main__':
    output = np.array([[20, 1, 1], [100, 100, 1], [100, 1, 1]], dtype=np.float32)
    print(output.shape)
    one_hot_label = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    # loss = focal_loss(output, one_hot_label)
    loss1 = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=one_hot_label, logits=output))
    loss2 = focal_loss_class_balanced(output, one_hot_label)
    with tf.Session() as sess:
        print(loss1.eval())
        print(loss2.eval())
        # print(p.eval())
        # print(pos_p.eval())
        # print(neg_p.eval())
        # print(pos_loss.eval())
        # print(neg_loss.eval())
        # print(sum_loss.eval())
        # print(final_p.eval())
        # print(final_loss.eval())

