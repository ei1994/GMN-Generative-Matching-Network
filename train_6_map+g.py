# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 19:14:10 2018

@author: DELL
"""

'''
双分支特征图结构
通过生成器进行数据扩充。

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

checkpoint_dir = 'ckpt_6_map+g'
checkpoint_dir_g = 'ckpt_g6'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
checkpoint_file_g = os.path.join(checkpoint_dir_g, 'model.ckpt')
train_dir='summary_map+g'

def initLogging(logFilename='record__map+g_6.log'):
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


def gfm_test():
    
    data1 = np.load('6_up_sift_harris_transform_train_test_data.npz')
    patch_test = data1['arr_1']
    patch_1_test = patch_test[:67000,:,:32,:]  # sar
    patch_2_test = patch_test[:67000,:,32:,:]  # opt
    y_test = data1['arr_3'][:67000,:]
    
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
            saver.restore(sess, tf.train.latest_checkpoint('ckpt_map+g_6'))
#            saver.restore(sess, 'ckpt_map+g_6/model.ckpt-12000')
            true_count = 0  # Counts the number of correct predictions.
            all_mout = np.array([])
            all_lab = np.array([])
            num = np.size(y_test)
            shuffle_test= gfm_shuffle(1,batch_size,patch_1_test,patch_2_test,y_test)
            for step1, (x_batch, y_batch, l_batch) in enumerate(shuffle_test):
                feed_dict = {inputs_sar:x_batch, inputs_opt:y_batch, inputs_lab:l_batch}
                result, p_out, p_ram, p_m = sess.run([correct,out,ram,m_output], feed_dict=feed_dict)
                if step1 == 0:
                    all_mout = p_m
                    all_lab = l_batch
                else:
                    all_mout = np.concatenate((all_mout, p_m), axis=0)
                    all_lab = np.concatenate((all_lab, l_batch), axis=0)

                true_count = true_count + result
                if step1 % 10 == 0:
                    print('Step %d run_test: batch_precision = %.2f '
                                      % (step1, result/batch_size))
            precision = float(true_count) / num
            print('  Num examples: %d  Num correct: %d  Precision : %0.04f' %
                    (num, true_count, precision))
            

def gfm_train():
    
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    checkpoints_dir = 'checkpoints/{}'.format(current_time)
    try:
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
        fake_opt = model.create_generator_1(inputs_sar, 1)
        gen_1 = [var for var in tf.trainable_variables() if var.name.startswith("generator_1")]
        fake_sar = model.create_generator_2(inputs_opt, 1)
        gen_2 = [var for var in tf.trainable_variables() if var.name.startswith("generator_2")]
        
        match_loss,m_output = model.gfm_sia_map(inputs_sar, inputs_opt, inputs_lab)
        out = tf.round(m_output)
        correct,ram = model.evaluation(out, inputs_lab)
        m_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(match_loss)
        
        tf.summary.scalar('mathing_loss', match_loss)
        summary = tf.summary.merge_all()
        saver_g_1 = tf.train.Saver(var_list=gen_1)
        saver_g_2 = tf.train.Saver(var_list=gen_2)
        saver = tf.train.Saver(max_to_keep=10)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            sess.run(init)
            saver_g_1.restore(sess, tf.train.latest_checkpoint('ckpt_g6_s2o'))
            saver_g_2.restore(sess, tf.train.latest_checkpoint('ckpt_g6_o2s'))
            try:
                shuffle1= gfm_shuffle(epoch,batch_size,patch_1_train,patch_2_train,y_train)
                for step, (x_batch, y_batch, l_batch) in enumerate(shuffle1):
                    start_time = time.time()
                    step = step + 1
                    
                    feed_dict = {inputs_sar:x_batch, inputs_opt:y_batch, inputs_lab:l_batch}
                    _, loss, m_output_ = sess.run([m_train_opt, match_loss, m_output], feed_dict = feed_dict)
                    
                    fake_opt_ = sess.run([fake_opt], feed_dict={inputs_sar:x_batch})
                    fake_sar_ = sess.run([fake_sar], feed_dict={inputs_opt:y_batch})
                    fake_opt_ = np.array(fake_opt_, np.float64)[0,:]
                    fake_sar_ = np.array(fake_sar_, np.float64)[0,:]
                    shuffle_index = np.random.permutation(batch_size)
                    shuffle_index = np.array(shuffle_index, np.int32)
                    fake_opt0 = fake_opt_[shuffle_index]
                    fake_sar0 = fake_sar_[shuffle_index]
                    X1 = np.concatenate((x_batch, x_batch, fake_sar_, fake_sar0),axis=0)
                    Y1 = np.concatenate((fake_opt_, fake_opt0, y_batch, y_batch),axis=0)
                    L1 = [1] * batch_size + [0] * batch_size +[1] * batch_size + [0] * batch_size 
                    L1 = np.array(L1, np.float64)[:, np.newaxis]
                    
                    shuffle0 = gfm_shuffle(1,batch_size,X1,Y1,L1)
                    for step1, (x_batch, y_batch, l_batch) in enumerate(shuffle0):
                        feed_dict0 = {inputs_sar:x_batch, inputs_opt:y_batch, inputs_lab:l_batch}
                        _, G_loss, m_output_ = sess.run([m_train_opt, match_loss, m_output], feed_dict = feed_dict0)
              
                    duration = time.time() - start_time
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    if step % 100 == 0:
                        logging.info('>> Step %d run_train: loss = %.2f G_loss = %.2f (%.3f sec)'
                                      % (step, loss, G_loss, duration))
                        
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

if __name__ == '__main__':
  gfm_train()
#  gfm_test()

