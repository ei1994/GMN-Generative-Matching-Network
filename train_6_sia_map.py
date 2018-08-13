#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 19:37:06 2017

@author: wrj
"""
'''
直接输入光和SAR，对比实验
双分支特征向图网络
'''

import numpy as np
import math
import time
import model_patent as model
import tensorflow as tf
import os
from datetime import datetime
import logging

batch_size = 200
epoch = 40
learning_rate = 2e-4
image_width = 32
image_height = 32

checkpoint_dir = 'ckpt_6_sia_map'
checkpoint_dir_g = 'ckpt_g6'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
checkpoint_file_g = os.path.join(checkpoint_dir_g, 'model.ckpt')
train_dir='summary_fm3_sia_map'

def initLogging(logFilename='record_sia_map.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
initLogging()

def gfm_shuffle(epoch,batch,x_data,y_data,label):
    for i in range(epoch):
        shuffle_index=np.random.permutation(y_data.shape[0])
        x_data1, y_data1, label1 = x_data[shuffle_index], y_data[shuffle_index], label[shuffle_index]
        batch_per_epoch = math.ceil(y_data.shape[0] / batch)
        for b in range(batch_per_epoch):
            if (b*batch+batch)>y_data.shape[0]:
                m,n = b*batch, y_data.shape[0]
            else:
                m,n = b*batch, b*batch+batch

            x_batch, y_batch, label_batch = x_data1[m:n,:], y_data1[m:n,:], label1[m:n,:]
            yield x_batch, y_batch, label_batch
            
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:,:,0]
#        image = image[:,:,np.newaxis]
    return image

    
def gfm_train():
    
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    checkpoints_dir = 'checkpoints/{}'.format(current_time)
    try:
        os.makedirs(checkpoint_dir)
        os.makedirs(checkpoints_dir)
    except os.error:
        pass
        
    data1 = np.load('6_up_sift_harris_transform_train_test_data.npz')
    patch_train = data1['arr_0']
    patch_1_train = patch_train[:200000,:,:32,:]  # sar
    patch_2_train = patch_train[:200000,:,32:,:]  # opt
    y_train = data1['arr_2'][:200000,:]
    patch_test = data1['arr_1']
    patch_1_test = patch_test[:3000,:,:32,:]  # sar
    patch_2_test = patch_test[:3000,:,32:,:]  # opt
    y_test = data1['arr_3'][:3000,:]

    graph = tf.Graph()
    with graph.as_default():
        inputs_sar = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], name='inputs_sar')
        inputs_opt = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], name='inputs_opt')
        inputs_lab = tf.placeholder(tf.float32, [batch_size, 1], name='inputs_lab')
        # 训练 M
        match_loss,m_output = model.gfm_sia_map(inputs_sar, inputs_opt, inputs_lab)
        out = tf.round(m_output)
        correct,ram = model.evaluation(out, inputs_lab)
        m_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(match_loss)
        
        tf.summary.scalar('mathing_loss', match_loss)
        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=10)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            sess.run(init)
            try:
                  shuffle1= gfm_shuffle(epoch,batch_size,patch_1_train,patch_2_train,y_train)
                  for step, (x_batch, y_batch, l_batch) in enumerate(shuffle1):
                        start_time = time.time()
                        step = step + 1
                        
                        feed_dict = {inputs_sar:x_batch, inputs_opt:y_batch, inputs_lab:l_batch}
                        _, m_loss,m_output_, m_out_ = sess.run([m_train_opt, match_loss, m_output, out ], feed_dict = feed_dict)
                        duration = time.time() - start_time
                        summary_str = sess.run(summary, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()

                        if step % 100 == 0:
                            logging.info('>> Step %d run_train: matching_loss = %.2f (%.3f sec)'
                                          % (step, m_loss, duration))
                            
                        if step % 3000 == 0 :
                            logging.info('>> %s Saving in %s' % (datetime.now(), checkpoint_dir))
                            saver.save(sess, checkpoint_file, global_step=step)
#                            
                        if step % 500 == 0 :
                            # test
                            true_count = 0  # Counts the number of correct predictions.
                            num = np.size(y_test)
                            shuffle_test= gfm_shuffle(1,batch_size,patch_1_test,patch_2_test,y_test)
                            for step_test, (x_batch, y_batch, l_batch) in enumerate(shuffle_test):
                                feed_dict = {inputs_sar:x_batch, inputs_opt:y_batch, inputs_lab:l_batch}
                                result, p_out, p_r = sess.run([correct,out,ram], feed_dict=feed_dict)
                
                                true_count = true_count + result
                            precision = float(true_count) / num
                            logging.info('Num examples: %d  Num correct: %d  Precision : %0.04f' %
                                        (num, true_count, precision))
                        
            except KeyboardInterrupt:
                print('INTERRUPTED')

            finally:
                saver.save(sess, checkpoint_file, global_step=step)
                print('Model saved in file :%s'%checkpoint_dir)

def gfm_test():

    data1 = np.load('6_up_sift_harris_transform_train_test_data.npz')
    patch_test = data1['arr_1']
    patch_1_test = patch_test[:30000,:,:32,:]  # sar
    patch_2_test = patch_test[:30000,:,32:,:]  # opt
    y_test = data1['arr_3'][:30000,:]
    
    graph = tf.Graph()
    with graph.as_default():
        inputs_sar = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], name='inputs_sar')
        inputs_opt = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], name='inputs_opt')
        inputs_lab = tf.placeholder(tf.float32, [batch_size, 1], name='inputs_lab')

        match_loss,m_output  = model.gfm_sia_map(inputs_sar, inputs_opt, inputs_lab)
        out = tf.round(m_output)
        correct,ram = model.evaluation(out, inputs_lab)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
#            saver.restore(sess, tf.train.latest_checkpoint('ckpt_fm3'))
            saver.restore(sess, 'ckpt_6_sia_map/model.ckpt-4000')
            true_count = 0  # Counts the number of correct predictions.
            num = np.size(y_test)
            shuffle_test= gfm_shuffle(1,batch_size,patch_1_test,patch_2_test,y_test)
            for step1, (x_batch, y_batch, l_batch) in enumerate(shuffle_test):
                feed_dict = {inputs_sar:x_batch, inputs_opt:y_batch, inputs_lab:l_batch}
                result, p_out, p_ram, p_m = sess.run([correct,out,ram,m_output], feed_dict=feed_dict)

                true_count = true_count + result
                if step1 % 10 == 0:
                    print('Step %d run_test: batch_precision = %.2f '
                                      % (step1, result/batch_size))
            precision = float(true_count) / num
            print('  Num examples: %d  Num correct: %d  Precision : %0.04f' %
                    (num, true_count, precision))
            
                
                
if __name__ == '__main__':
  gfm_train()
#  gfm_test()

