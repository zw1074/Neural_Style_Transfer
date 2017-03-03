#########################
# Author: Zihao Wang    #
# Email: zw1074@nyu.edu #
# Date: Mar. 3, 2017    #
#########################
# Mainly come from the neural style paper: 
# http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

import tensorflow as tf
from scipy.misc import imresize
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import argparse

parser = argparse.ArgumentParser(description="Neural Style")
parser.add_argument('-c', '--content', type=str, default='./content.img', help='input content image')
parser.add_argument('-s', '--style', type=str, default='./style.img', help='input style image')
parser.add_argument('-v', '--vgg', type=str, default='./vgg16.npy', help='vgg weight path')
parser.add_argument('-o', '--output', type=str, default='./output.png', help='output path')
parser.add_argument('-l', '--loop', type=int, default=1000, help='loop time')
parser.add_argument('-a', '--alpha', type=float, default=1e-3, help='alpha parameter, big value will focus on content')
parser.add_argument('-b', '--beta', type=float, default=1.0, help='beta parmeter, big value will focus on style')
parser.add_argument('-w', '--weight', type=float, default=1.0, help='weight parameter for variance in loss')
args = parser.parse_args()


# Define VGG mean for unprocess
VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print path

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, bgr):
        """
        load variable from npy to build the VGG

        :param bgr: rgb image [batch, height, width, 3] values scaled [0, 1]
        """


        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")

    
    def get_style(self):
        return [self.conv1_1, self.conv2_1, self.conv3_1, self.conv4_1, self.conv5_1]
    
    def get_content(self):
        return self.conv4_2

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    # def fc_layer(self, bottom, name):
    #     with tf.variable_scope(name):
    #         shape = bottom.get_shape().as_list()
    #         dim = 1
    #         for d in shape[1:]:
    #             dim *= d
    #         x = tf.reshape(bottom, [-1, dim])

    #         weights = self.get_fc_weight(name)
    #         biases = self.get_bias(name)

    #         # Fully connected layer. Note that the '+' operation automatically
    #         # broadcasts the biases.
    #         fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

    #         return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

def preprocess(img):
    im = img.astype('float32')
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    return im

def unprocess(img):
    im = img.astype('float32')
    im[:,:,0] += 103.939
    im[:,:,1] += 116.779
    im[:,:,2] += 123.68
    return im

class Neural_Style(object):
    """Neural Style Generator"""
    
    def __init__(self, content, style):
        """
        Params:
            content -- content image
            style -- style image
        """
        self.content = content.astype('float32')
        self.style = style.astype('float32')
        self.sizes = [64, 128, 256, 512, 512]
        self.sizes2 = [224, 112, 56, 28, 14]
    
    def _loss_content(self, original, generation):
        return tf.nn.l2_loss(original-generation)
    
    def _loss_style(self, styles_conv1_1, 
                          styles_conv2_1, 
                          styles_conv3_1, 
                          styles_conv4_1, 
                          styles_conv5_1, 
                          generation_styles):
        loss = 0.0
        original_styles = [styles_conv1_1, styles_conv2_1, styles_conv3_1, styles_conv4_1, styles_conv5_1]
        for i in xrange(5):
            G1 = tf.reshape(tf.squeeze(original_styles[i]), shape=[-1, self.sizes[i]])
            G1 = tf.matmul(tf.transpose(G1), G1)
            G2 = tf.reshape(tf.squeeze(generation_styles[i]), shape=[-1, self.sizes[i]])
            G2 = tf.matmul(tf.transpose(G2), G2)
            loss = loss + 1/5.0*tf.nn.l2_loss(G1-G2)/(2*self.sizes[i]**2*self.sizes2[i]**4)
        return loss
        
        
    def build(self, alpha, beta, weight3):
        """
        Params:
            alpha -- big for content
            beta -- big for style
        """
        self.alpha = alpha
        self.beta = beta
        self.weight3 = weight3
        
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys[:-6]):
            print(i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))
    
    def generation(self, content_size, iters=1000, SAVE_PATH=None):
        
        img = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
        
        styles_conv1_1 = tf.placeholder(tf.float32, shape=[1, self.sizes2[0], self.sizes2[0], self.sizes[0]])
        styles_conv2_1 = tf.placeholder(tf.float32, shape=[1, self.sizes2[1], self.sizes2[1], self.sizes[1]])
        styles_conv3_1 = tf.placeholder(tf.float32, shape=[1, self.sizes2[2], self.sizes2[2], self.sizes[2]])
        styles_conv4_1 = tf.placeholder(tf.float32, shape=[1, self.sizes2[3], self.sizes2[3], self.sizes[3]])
        styles_conv5_1 = tf.placeholder(tf.float32, shape=[1, self.sizes2[4], self.sizes2[4], self.sizes[4]])
        content_ = tf.placeholder(tf.float32, shape=[1, self.sizes2[3], self.sizes2[3], self.sizes[3]])
        
        sess = tf.InteractiveSession()
        first_net = Vgg16(args.vgg)
        first_net.build(img)
        styles = first_net.get_style()
        contents = first_net.get_content()

        # get styles and content
        contents_val = sess.run([contents], feed_dict={img: [preprocess(self.content)]})[0]
        styles_val = sess.run([styles], feed_dict = {img: [preprocess(self.style)]})[0]

#         noise = np.random.normal(size=, scale=np.std(self.content) * 0.1)
#             img_generation = tf.Variable(tf.random_normal([1,224,224,3]) * 0.256, name='generation', trainable=True)
        img_generation = tf.Variable(preprocess(self.content).reshape(1,224,224,3))
        second_net = Vgg16('vgg16/vgg16.npy')
        second_net.build(img_generation)
        g_styles = second_net.get_style()
        g_contents = second_net.get_content()
        variational = self.weight3 * 2 * (tf.nn.l2_loss(img_generation[:,1:,:,:] - img_generation[:,:223,:,:]) / 
                                      (1*223*224*3) + 
                                      tf.nn.l2_loss(img_generation[:,:,1:,:] - img_generation[:,:,:223,:]) /
                                      (1*223*224*3))
#         g_styles, g_contents = VGG_net(img_generation, weights="./vgg_trial/vgg16_weights.npz", sess=sess)
        loss = self.alpha * self._loss_content(content_, g_contents) + \
               self.beta * self._loss_style(styles_conv1_1, 
                                          styles_conv2_1, 
                                          styles_conv3_1, 
                                          styles_conv4_1, 
                                          styles_conv5_1, g_styles) + variational
        feed = {styles_conv1_1: styles_val[0],
                styles_conv2_1: styles_val[1],
                styles_conv3_1: styles_val[2],
                styles_conv4_1: styles_val[3],
                styles_conv5_1: styles_val[4],
                content_: contents_val}
        global_step = tf.Variable(0, name="global_step", trainable=False)
        lr = tf.train.exponential_decay(1.0, 
                                        global_step,
                                        10,
                                        0.9,
                                        staircase=True)
        grad_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        for i in xrange(iters):
            _, self.generat, loss_val = sess.run([grad_op, img_generation, loss], feed_dict=feed)

#             plt.imshow(self.generat.reshape(224, 224, 3))
            if i % 100 == 0:
                print loss_val
                print i
        self.generat = np.maximum(0, unprocess(self.generat.reshape(224,224,3)))
        self.generat /= np.max(self.generat)
#         self.generat /= np.max(self.generat)
        self.genrat = (self.generat*255).astype('uint8')
        self.genrat = imresize(self.generat, content_size)
        scipy.misc.imsave(SAVE_PATH, self.genrat)
        print "Output image saved."

def main():
    content_img = imread(args.content)
    content_size = content_img.shape
    content_img = imresize(imread(args.content), (224, 224, 3))
    feature_img = imresize(imread(args.style), (224, 224, 3))
    neural = Neural_Style(content_img, feature_img)
    neural.build(args.alpha, args.beta, args.weight)
    neural.generation(content_size, iters=args.loop, SAVE_PATH=args.output)

if __name__ == '__main__':
    main()