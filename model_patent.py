# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:03:33 2018

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
a = parser.parse_args()

EPS = 1e-12

def conv(batch_input, out_channels, filter_size, pad, stride=1):
    conv = tf.layers.conv2d(batch_input, out_channels,filter_size,strides=(stride,stride), padding=pad)
    return conv

def batchnorm(inputs):
    normalized = tf.layers.batch_normalization(inputs, axis=-1)
    return normalized

def deconv(batch_input, out_channels, filter_size, pad):
    conv = tf.layers.conv2d_transpose(batch_input, out_channels, filter_size, strides=(2,2), padding=pad ,kernel_initializer=tf.random_normal_initializer(0, 0.02))
    return conv

def evaluation(logits, labels):
      correct_prediction = tf.equal(logits, labels)
      correct_batch = tf.reduce_mean(tf.cast(correct_prediction, tf.int32), 1)
      return tf.reduce_sum(tf.cast(correct_batch, tf.float32)), correct_batch
       
def create_generator_1(generator_inputs, generator_outputs_channels,name="generator_1", reuse=False):
#    U_Net结构
    with tf.variable_scope(name,reuse=reuse):
        layers = []
        with tf.variable_scope("encoder_1"):
            # encoder_1: [batch, 32, 32, 1 ] => [batch, 16, 16, 32]
            convolved = conv(generator_inputs, 32, 3, pad='SAME')
            convolved = batchnorm(convolved)
            convolved = tf.nn.relu(convolved)
            output = tf.nn.max_pool(convolved,[1,2,2,1],[1,2,2,1],padding='SAME') 
            
            layers.append(output)
    
        layer_specs = [
            64, # encoder_2: [batch, 16, 16, 32 ] => [batch, 8, 8, 64]
            128, # encoder_3: [batch, 8, 8, 64] => [batch, 4, 4, 128]
            256, # encoder_4: [batch, 4, 4, 128] => [batch, 2, 2, 256]
        ]
    
        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(output, out_channels, 3, pad='SAME')
                convolved = batchnorm(convolved)
                convolved = tf.nn.relu(convolved)
                output = tf.nn.max_pool(convolved,[1,2,2,1],[1,2,2,1],padding='SAME') 
                layers.append(output)
    
        layer_specs = [
            (128, 0.0),   # decoder_1: [batch, 2, 2, 256] => [batch, 4, 4, 128]
            (64, 0.0),   # decoder_2: [batch, 4, 4, 128] => [batch, 8, 8, 64]
            (32, 0.0),   # decoder_3: [batch, 8, 8, 64] => [batch, 16, 16, 32]
        ]
        num_encoder_layers = len(layers)  
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    inputs = layers[-1]
                else:
                    inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)  #跨层连接
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(inputs, out_channels, 3, pad='SAME')
                output = batchnorm(output)
                output = tf.nn.relu(output)
    
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
    
                layers.append(output)
    
        # decoder_4: [batch, 16, 16, ngf ] => [batch, 32, 32, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            inputs = tf.concat([layers[-1], layers[0]], axis=3)
            output = deconv(inputs, generator_outputs_channels,3,pad='SAME')
            output = tf.tanh(output)
            layers.append(output)
        return layers[-1]

def create_generator_2(generator_inputs, generator_outputs_channels,name="generator_2", reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        layers = []
        with tf.variable_scope("encoder_1"):
            # encoder_1: [batch, 32, 32, 1 ] => [batch, 16, 16, 32]
            convolved = conv(generator_inputs, 32, 3, pad='SAME')
            convolved = batchnorm(convolved)
            convolved = tf.nn.relu(convolved)
            output = tf.nn.max_pool(convolved,[1,2,2,1],[1,2,2,1],padding='SAME') 
            
            layers.append(output)
    
        layer_specs = [
            64, # encoder_2: [batch, 16, 16, 32 ] => [batch, 8, 8, 64]
            128, # encoder_3: [batch, 8, 8, 64] => [batch, 4, 4, 128]
            256, # encoder_4: [batch, 4, 4, 128] => [batch, 2, 2, 256]
        ]
    
        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(output, out_channels, 3, pad='SAME')
                convolved = batchnorm(convolved)
                convolved = tf.nn.relu(convolved)
                output = tf.nn.max_pool(convolved,[1,2,2,1],[1,2,2,1],padding='SAME') 
                layers.append(output)
    
        layer_specs = [
            (128, 0.0),   # decoder_1: [batch, 2, 2, 256] => [batch, 4, 4, 128]
            (64, 0.0),   # decoder_2: [batch, 4, 4, 128] => [batch, 8, 8, 64]
            (32, 0.0),   # decoder_3: [batch, 8, 8, 64] => [batch, 16, 16, 32]
        ]
        num_encoder_layers = len(layers)  
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    inputs = layers[-1]
                else:
                    inputs = tf.concat([layers[-1], layers[skip_layer]], axis=3)  #跨层连接
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(inputs, out_channels, 3, pad='SAME')
                output = batchnorm(output)
                output = tf.nn.relu(output)
    
                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)
    
                layers.append(output)
    
        # decoder_4: [batch, 16, 16, ngf ] => [batch, 32, 32, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            inputs = tf.concat([layers[-1], layers[0]], axis=3)
            output = deconv(inputs, generator_outputs_channels,3,pad='SAME')
            output = tf.tanh(output)
            layers.append(output)
        return layers[-1]

# 判别器网络
def create_discriminator(inputs_s, inputs_o, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        inputs = tf.concat([inputs_s, inputs_o], axis=3)  #  32*32*2
        layer1 = conv(inputs, 32, 3, pad='SAME')    # 32*32*2 --> 32*32*32
        layer1 = batchnorm(layer1)
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 

        layer2 = conv(pool1, 64, 3, pad='SAME')   # 16*16*32 --> 16*16*64
        layer2 = batchnorm(layer2)
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 

        layer3 = conv(pool2, 128, 3, pad='SAME')   # 8*8*64 --> 8*8*128
        layer3 = batchnorm(layer3)
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer4 = conv(pool3, 256, 3, pad='SAME')   # 4*4*128 --> 4*4*256
        layer4 = batchnorm(layer4)
        layer4 = tf.nn.relu(layer4)
        pool4 = tf.nn.max_pool(layer4,[1,2,2,1],[1,2,2,1],padding='SAME')   # 2*2*256
        
        batch_size = int(pool4.get_shape()[0])
        dense = tf.reshape(pool4, [batch_size,-1])
        dense1 = tf.layers.dense(inputs=dense, units=512)
        dense1 = tf.nn.relu(dense1)
        dense2 = tf.layers.dense(inputs=dense1, units=128)
        dense2 = tf.nn.relu(dense2)
        dense3 = tf.layers.dense(inputs=dense2, units=1)
        output = tf.sigmoid(dense3)
        return output


def features(img, name, reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        layer1 = conv(img, 32, 3, pad='SAME')  # 32*32*1 --> 16*16*32
        layer1 = batchnorm(layer1)
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer2 = conv(pool1, 64, 3, pad='SAME')  # 16*16*32 --> 8*8*64
        layer2 = batchnorm(layer2)
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer3 = conv(pool2, 128, 3, pad='SAME')    # 8*8*64 --> 4*4*128
        layer3 = batchnorm(layer3)
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        layer4 = conv(pool3, 128, 3, pad='SAME')  # 4*4*128 --> 2*2*128
        layer4 = batchnorm(layer4)
        layer4 = tf.nn.relu(layer4)
        pool4 = tf.nn.max_pool(layer4,[1,2,2,1],[1,2,2,1],padding='SAME') 
        # 2*2*128=512
        batch_size1 = int(pool4.get_shape()[0])
        dense = tf.reshape(pool4, [batch_size1,-1])
        return dense

def matching(f1,f2):
    with tf.variable_scope('match_layers'):
        all_feature = tf.concat([f1, f2], axis=1)  # 两个特征向量连接 
        dense1 = tf.layers.dense(all_feature, 512) # 512*2 --> 512
        dense1 = tf.nn.relu(dense1)
        dense2 = tf.layers.dense(dense1, 128)
        dense2 = tf.nn.relu(dense2)
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output

# 双分支, tensor结构
def features_matching(inputs_r, inputs_g):
    with tf.variable_scope("matching"):
        f1 = features(inputs_r, 'inputs_r')
        f2 = features(inputs_g, 'inputs_g')
        output = matching(f1,f2)   
        return output

# 双通道
def features_matching_1(inputs_r, inputs_g, name="matching", reuse=False):
    with tf.variable_scope(name,reuse=reuse):
    # layers: [batch, 32, 32, in_channels] => [batch, 4, 4, ndf*2]
        img = tf.concat([inputs_r, inputs_g], axis=3)  # 两个输入图片连接 # 32*32*2
        layer1 = conv(img, 64, 3, pad='SAME')  # 32*32*2 --> 16*16*64
        layer1 = batchnorm(layer1)
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer2 = conv(pool1, 128, 3, pad='SAME')  # 16*16*64 --> 8*8*128
        layer2 = batchnorm(layer2)
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer3 = conv(pool2, 256, 3, pad='SAME')    # 8*8*128 --> 4*4*256
        layer3 = batchnorm(layer3)
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME')
        
        layer4 = conv(pool3, 256, 3, pad='SAME')  # 4*4*256 --> 2*2*256
        layer4 = batchnorm(layer4)
        layer4 = tf.nn.relu(layer4)
        pool4 = tf.nn.max_pool(layer4,[1,2,2,1],[1,2,2,1],padding='SAME') 
        # 2*2*256
        batch_size = int(pool4.get_shape()[0])
        dense = tf.reshape(pool4, [batch_size,-1])
        dense1 = tf.layers.dense(dense, 512)
        dense1 = tf.nn.relu(dense1)
        dense2 = tf.layers.dense(dense1, 128)
        dense2 = tf.nn.relu(dense2)
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output

# 双分支特征图结构，特征提取
def features_2(img, name, reuse=False):
    with tf.variable_scope(name,reuse=reuse):
        layer1 = conv(img, 32, 3, pad='SAME')  # 32*32*1 --> 32*32*32
        layer1 = batchnorm(layer1)
        layer1 = tf.nn.relu(layer1)
        pool1 = tf.nn.max_pool(layer1,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer2 = conv(pool1, 64, 3, pad='SAME')  # 16*16*32 --> 16*16*64
        layer2 = batchnorm(layer2)
        layer2 = tf.nn.relu(layer2)
        pool2 = tf.nn.max_pool(layer2,[1,2,2,1],[1,2,2,1],padding='SAME') 
        
        layer3 = conv(pool2, 128, 3, pad='SAME')    # 8*8*64 --> 8*8*128
        layer3 = batchnorm(layer3)
        layer3 = tf.nn.relu(layer3)
        pool3 = tf.nn.max_pool(layer3,[1,2,2,1],[1,2,2,1],padding='SAME')
        # 4*4*128
        return pool3

# 双分支特征图结构，匹配网络
def matching_2(f1,f2):
    with tf.variable_scope('match_layers'):
        all_feature = tf.concat([f1, f2], axis=3)  # 两个特征连接 # 4*4*256
        layer4 = conv(all_feature, 256, 3, pad='SAME')
        layer4 = batchnorm(layer4)
        layer4 = tf.nn.relu(layer4)
        pool4 = tf.nn.max_pool(layer4,[1,2,2,1],[1,2,2,1],padding='SAME')
        # 2*2*256
        batch_size = int(pool4.get_shape()[0])
        dense = tf.reshape(pool4, [batch_size,-1])
        dense1 = tf.layers.dense(dense, 512)
        dense1 = tf.nn.relu(dense1)
        dense2 = tf.layers.dense(dense1, 128)
        dense2 = tf.nn.relu(dense2)
        dense3 = tf.layers.dense(dense2, 1)
        output = tf.sigmoid(dense3)   
        return output

# 双分支特征图结构
def features_matching_2(inputs_r, inputs_g):
    
    with tf.variable_scope("matching"):
        f1 = features_2(inputs_r, 'inputs_r')
        f2 = features_2(inputs_g, 'inputs_g')
        output = matching_2(f1,f2)   
        return output

        
def gd_model_s2o(inputs_s, inputs_o):
    # sar --> opt
#    with tf.variable_scope("sar2opt"):
        out_channels = int(inputs_o.get_shape()[-1])
        g_outputs = create_generator_1(inputs_s, out_channels)  # fake_optical
        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1]
            predict_real= create_discriminator(inputs_s, inputs_o)
        with tf.name_scope("fake_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1]
            predict_fake = create_discriminator(inputs_s, g_outputs, reuse=True)
    
        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
    
        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(inputs_o - g_outputs))
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight   # 为什么权重相差这么大
            
        return  gen_loss, discrim_loss, g_outputs

def gd_model_o2s(inputs_s, inputs_o):
    # opt --> sar
#    with tf.variable_scope("opt2sar"):
        out_channels = int(inputs_s.get_shape()[-1])
        g_outputs = create_generator_2(inputs_o, out_channels)  # fake_sar
        # create two copies of discriminator, one for real pairs and one for fake pairs
        # they share the same underlying variables
        with tf.name_scope("real_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1]
            predict_real= create_discriminator(inputs_s, inputs_o)
        with tf.name_scope("fake_discriminator"):
            # 2x [batch, height, width, channels] => [batch, 1]
            predict_fake = create_discriminator(g_outputs,inputs_o,  reuse=True)
    
        with tf.name_scope("discriminator_loss"):
            # minimizing -tf.log will try to get inputs to 1
            # predict_real => 1
            # predict_fake => 0
            discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
    
        with tf.name_scope("generator_loss"):
            # predict_fake => 1
            # abs(targets - outputs) => 0
            gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
            gen_loss_L1 = tf.reduce_mean(tf.abs(inputs_s - g_outputs))
            gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight   # 为什么权重相差这么大
            
        return  gen_loss, discrim_loss, g_outputs    

# 双分支特征向量
def gfm_sia_tensor(g_outputs, inputs_o, label):        
    with tf.name_scope("features_matching"):
         m_output = features_matching(inputs_o, g_outputs)
    with tf.name_scope("features_matching_loss"):
        match_loss =  tf.reduce_mean(-(label * tf.log(m_output + EPS) + (1 - label) * tf.log(1 - m_output + EPS)))
    return match_loss, m_output

# 双通道
def gfm_chan(g_outputs, inputs_o, label):         
    with tf.name_scope("features_matching"):
         m_output = features_matching_1(inputs_o, g_outputs)
    with tf.name_scope("features_matching_loss"):
        match_loss =  tf.reduce_mean(-(label * tf.log(m_output + EPS) + (1 - label) * tf.log(1 - m_output + EPS)))
    return match_loss, m_output

# 双分支特征图
def gfm_sia_map(g_outputs, inputs_o, label):  
    with tf.name_scope("features_matching"):
         m_output = features_matching_2(inputs_o, g_outputs)
    with tf.name_scope("features_matching_loss"):
        match_loss =  tf.reduce_mean(-(label * tf.log(m_output + EPS) + (1 - label) * tf.log(1 - m_output + EPS)))
    return match_loss, m_output







