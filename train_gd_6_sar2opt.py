
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 16:55:26 2017

@author: DELL
"""

'''
用于生成器的训练
sar --> opt
'''

import numpy as np
import math
import time 
import model_patent as model
import tensorflow as tf
import os
from datetime import datetime
import logging
from scipy import misc

batch_size = 200
epoch = 100
learning_rate = 2e-4
image_width = 32
image_height = 32
checkpoint_dir = 'ckpt_gd6_s2o'
checkpoint_dir_g = 'ckpt_g6_s2o'
output_dir = 'out6_s2o'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
checkpoint_file_g = os.path.join(checkpoint_dir_g, 'model.ckpt')
train_dir='summary_gd'

def initLogging(logFilename='record_gd6_s2o.log'):
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


def gd_shuffle(epoch,batch,x_data,y_data):
    for i in range(epoch):
        shuffle_index=np.random.permutation(y_data.shape[0])
        x_data1, y_data1 = x_data[shuffle_index], y_data[shuffle_index]
        batch_per_epoch = math.ceil(y_data.shape[0] / batch)
        for b in range(batch_per_epoch):
            if (b*batch+batch)>y_data.shape[0]:
                m,n = b*batch, y_data.shape[0]
            else:
                m,n = b*batch, b*batch+batch

            x_batch, y_batch = x_data1[m:n,:], y_data1[m:n,:]
            yield x_batch, y_batch

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
    return image

def gd_train():
    
#    if FLAGS.load_model is not None:
#        checkpoints_dir = 'checkpoints/' + FLAGS.load_model
#    else:
    current_time = datetime.now().strftime('%Y%m%d-%H%M')
    checkpoints_dir = 'checkpoints/{}'.format(current_time)
    try:
        os.makedirs(checkpoint_dir_g)
        os.makedirs(checkpoint_dir)
    except os.error:
        pass
    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass
    try:
        os.makedirs(output_dir)
    except os.error:
        pass
    
    data2 = np.load('6_up_sift_harris_mapping_data.npy')  
    X_train = data2[:70000,:,:32,:]  # sar
    Y_train = data2[:70000,:,32:,:]  # opt
    X_test = data2[70000:70100,:,:32,:]
    Y_test = data2[70000:70100,:,32:,:]

    graph = tf.Graph()
    with graph.as_default():
        inputs_sar = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], name='inputs_sar')
        inputs_opt = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], name='inputs_opt')
        test_sar = tf.placeholder(tf.float32, [None, image_height, image_width, 1], name='test_sar')
        
        # 训练 G
        gen_loss, dis_loss, _ = model.gd_model_s2o(inputs_sar, inputs_opt)
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(dis_loss,var_list=discrim_tvars)
        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator_1")]
        g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss,var_list=gen_tvars)
        
#        with tf.control_dependencies([g_train_opt, d_train_opt]):
#             gd_train_opt = tf.no_op(name='optimizers')
        
        tf.summary.scalar('gen_loss', gen_loss)
        tf.summary.scalar('dis_loss', dis_loss)
        summary = tf.summary.merge_all()
        
        saver = tf.train.Saver()
        saver_g = tf.train.Saver(var_list=gen_tvars)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            sess.run(init)
#            saver.restore(sess, tf.train.latest_checkpoint('ckpt_gd'))
#            saver.restore(sess, 'ckpt_gd/model.ckpt-4000')
            try:
              shuffle1= gd_shuffle(epoch,batch_size,X_train,Y_train)
              
              for step, (x_batch, y_batch) in enumerate(shuffle1):
                    start_time = time.time()
                    step = step + 1
                    feed_dict = {inputs_sar:x_batch, inputs_opt:y_batch}
                    _, _, g_loss,d_loss = sess.run([d_train_opt,g_train_opt, gen_loss, dis_loss], feed_dict = feed_dict)
                    duration = time.time() - start_time
                                        
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
#                    
                    if step % 100 == 0:
                        logging.info('>> Step %d run_train: g_loss = %.2f  d_loss = %.2f (%.3f sec)'
                                      % (step, g_loss, d_loss, duration))
                    if step % 2000 == 0 :
                        logging.info('>> %s Saving in %s' % (datetime.now(), checkpoint_dir))
                        saver.save(sess, checkpoint_file, global_step=step)
                        saver_g.save(sess, checkpoint_file_g, global_step=step)
                        
                        gen = model.create_generator_1(test_sar, 1, reuse=True)
                        gen_out = sess.run(gen, feed_dict={test_sar:X_test} )
                        show_images=np.concatenate((X_test,gen_out,Y_test),axis=1)
                        result = combine_images(show_images)
                        result = result*255
                        misc.imsave('out6_s2o/{}.png'.format(str(epoch)+"_"+str(step)), result)
                        
            except KeyboardInterrupt:
                print('INTERRUPTED')

            finally:
                saver.save(sess, checkpoint_file, global_step=step)
                saver_g.save(sess, checkpoint_file_g, global_step=step)
                print('Model saved in file :%s'%checkpoint_dir)

if __name__ == '__main__':
  gd_train()

