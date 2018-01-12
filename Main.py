# coding=utf-8
import tensorflow as tf
from PIL import Image
import numpy as np
import os

batch_size = 16
batch_num = 224
test_batch_start = 200
Image_width = 28
Image_height = 28
N_classes = 14
learning_rate = 0.001
is_testing = False

sess = tf.InteractiveSession()


# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# create model use placeholder
x = tf.placeholder(tf.float32, shape=[None, Image_height * Image_width])
y_ = tf.placeholder(tf.float32, shape=[None, N_classes])
x_image = tf.reshape(x,[-1,28,28,1])


# first layer
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # 28->14

# second layer 32-64
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # 14->7

# densely connected layer
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
w_fc2 = weight_variable([1024, 14])
b_fc2 = bias_variable([14])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
scores = tf.nn.xw_plus_b(h_fc1_drop, w_fc2,b_fc2,name="scores")

# train and evaluate the model
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99).minimize(cross_entropy)

# 定义评测准确率的操作
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_dir = "C:\TRAIN"


# 使用队列读取
def read_decode():
    imagepaths, labels = list(), list()
    label = 0
    list2 = os.listdir(train_dir)
    labels = []

    for i in range(0, len(list2)):
        path = os.path.join(train_dir, list2[i])
        list_child = os.listdir(path)
        for j in range(0, len(list_child)):
            label_s = []
            path_whole = os.path.join(path, list_child[j])
            imagepaths.append(path_whole)
            for num in range(N_classes):
                if num == label:
                    label_s.append(1)
                else:
                    label_s.append(0)

            labels.append(label_s)
        label += 1

    img = []
    for i in range(len(imagepaths)):
        pixel = []
        image = Image.open(imagepaths[i])
        for w in range(Image_width):
            for h in range(Image_height):
                if image.getpixel((h, w)) == 255:
                    pixel.append(0)
                else:
                    pixel.append(1)
        img.append(pixel)

    return img, labels

tf.global_variables_initializer().run()
X, label = read_decode()
index = [n for n in range(0, N_classes*256)]
np.random.shuffle(index)
image_batch = [[0 for x in range(batch_size)] for y in range(batch_num)]
label_batch = [[0 for x in range(batch_size)] for y in range(batch_num)]
for j in range(batch_num):
    for z in range(batch_size):
        image_batch[j][z] = X[index[j * batch_size + z]]
        label_batch[j][z] = label[index[j * batch_size + z]]

test_image_batch = []
test_label_batch = []
#validation_image_batch = []
#validation_label_batch = []
for i in range(test_batch_start, batch_num):
    for j in range(batch_size):
        test_image_batch.append(image_batch[i][j])
        test_label_batch.append(label_batch[i][j])

saver = tf.train.Saver()
# for i in range(9,11):
#    for j in range(batch_size):
#        validation_image_batch.append(image_batch[i][j])
#        validation_label_batch.append(label_batch[i][j])

if is_testing :
    saver.restore(sess, "./model.ckpt")
    train_accuracy = accuracy.eval(feed_dict={x: test_image_batch, y_: test_label_batch, keep_prob: 1})
    print("-->training accuracy %.4f" % (train_accuracy))
else :
    for i in range(1,200):
        for j in range(test_batch_start):
            train_step.run(feed_dict={x: image_batch[j], y_: label_batch[j], keep_prob: 0.5})

        # train_accuracy = accuracy.eval(feed_dict={x: validation_image_batch, y_: validation_label_batch, keep_prob: 1})
        # print("-->step %d, training accuracy %.4f" % (i, train_accuracy))

        # if train_accuracy > 0.985:
        train_accuracy = accuracy.eval(feed_dict={x: test_image_batch, y_: test_label_batch, keep_prob: 1})
        print("-->step %d, training accuracy %.4f" % (i, train_accuracy))

        if(train_accuracy > 0.999):
            saver.save(sess, "./model.ckpt")