import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import pandas as pd
import cv2
import urllib

# Functions and classes for loading and using the Inception model.
from rnk.inception import inception


model = inception.Inception()


classes = ['jackets_blazers', 'underwear_brief_or_boxers', 'trousers_chinos','trousers_cropped', 'jackets_denim',
          'jeans_denim', 'trousers_drop_crotch', 'jackets_leather', 'shorts', 'T-shirt']
n_class = 10
n_dim = 2048

x = tf.placeholder(tf.float32, shape=[None, n_dim], name='x')

y_true_labels = tf.placeholder(tf.float32, shape=[None, n_class], name='y_true_labels')

y_true_cls = tf.argmax(y_true_labels, axis=1)

n_hidden_1 = 2048
n_hidden_2 = 2048

weigths = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_class])),
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
}

layer_1 = tf.add(tf.matmul(x, weigths['h1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, weigths['h2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)

y_pred_layer = tf.matmul(layer_2, weigths['out']) + biases['out']

y_pred_cls = tf.argmax(y_pred_layer, axis=1)

y_pred_prbl = tf.nn.softmax(y_pred_layer, name='y_pred_prbl')

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_layer,labels=y_true_labels))

learning_rate=1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



sess = tf.Session()
sess.run(tf.global_variables_initializer())

model_path = "/home/shresth/Django/shresth/rnk/Inception Clothes Recognition Gray new/ICRGmodel.ckpt"
saver = tf.train.Saver()
saver.restore(sess, model_path)



classes = ['jackets_blazers', 'underwear_brief_or_boxers', 'trousers_chinos','trousers_cropped', 'jackets_denim',
          'jeans_denim', 'trousers_drop_crotch', 'jackets_leather', 'shorts', 'T-shirt']
n_class = 10
n_dim = 2048

x = tf.placeholder(tf.float32, shape=[None, n_dim], name='x')

y_true_labels = tf.placeholder(tf.float32, shape=[None, n_class], name='y_true_labels')

y_true_cls = tf.argmax(y_true_labels, axis=1)

n_hidden_1 = 2048
n_hidden_2 = 2048

weigths = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_class])),
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
}

layer_1 = tf.add(tf.matmul(x, weigths['h1']), biases['b1'])
layer_1 = tf.nn.relu(layer_1)

layer_2 = tf.add(tf.matmul(layer_1, weigths['h2']), biases['b2'])
layer_2 = tf.nn.relu(layer_2)

y_pred_layer = tf.matmul(layer_2, weigths['out']) + biases['out']

y_pred_cls = tf.argmax(y_pred_layer, axis=1)

y_pred_prbl = tf.nn.softmax(y_pred_layer, name='y_pred_prbl')

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_layer,labels=y_true_labels))

learning_rate=1e-4
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())

model_path = "/home/shresth/Django/shresth/rnk/Inception Clothes Recognition Gray new/ICRGmodel.ckpt"
saver = tf.train.Saver()
saver.restore(sess, model_path)


user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
headers={'User-Agent':user_agent,}

def callit(link):

    filename = 'mm'
    path = os.getcwd() + '/' + filename
    imagefile = open(path + ".jpeg", 'wb')
    imagefile.write(urllib.request.urlopen(link).read())
    imagefile.close()

    img = cv2.imread( os.getcwd() + '/mm.jpeg')
    img = np.array(img)

    #request= urllib.request.Request(link, None, headers) #The assembled request
    #req = urllib.request.urlopen(request)
    #arry = np.asarray(bytearray(req.read()), dtype=np.uint8)
    #img = cv2.imdecode(arry, -1) # 'Load it as it is'
    #img = np.array(img)

    imgr = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    new_imgr = np.zeros((len(imgr),len(imgr[0]),3))
    for i in range(len(imgr)):
        for j in range(len(imgr[0])):
            new_imgr[i][j][0] = imgr[i][j]
            new_imgr[i][j][1] = imgr[i][j]
            new_imgr[i][j][2] = imgr[i][j]

    t_val = model.transfer_values(image=new_imgr)
    t_val = [t_val]

    layerr = sess.run(y_pred_layer, {x: t_val, y_true_labels: [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]})

    layerr = np.array(layerr)
    inn = layerr.argsort()
    cls = classes[int(inn[0][9])] + " and may be " + classes[int(inn[0][8])] + " or " + classes[int(inn[0][7])]

    return cls, layerr
