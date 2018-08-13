# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:15:34 2018

@author: DELL
"""
'''
调用训练好的网络，
对测试图像提取的图像块对进行样本点匹配测试

'''

import numpy as np
import model_patent as model
import tensorflow as tf
import os
import math

import h5py
import scipy.io as sio 

batch_size = 100
image_width = 32
image_height = 32

checkpoint_dir = 'ckpt_fm1'
checkpoint_dir_g = 'ckpt_g'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
checkpoint_file_g = os.path.join(checkpoint_dir_g, 'model.ckpt')
train_dir='summary_fm1'

def gfm_shuffle(epoch,batch,x_data,y_data,label):
    for i in range(epoch):
#        shuffle_index=np.random.permutation(y_data.shape[0])
        x_data1, y_data1, label1 = x_data, y_data, label
        batch_per_epoch = math.ceil(y_data.shape[0] / batch)
        for b in range(batch_per_epoch):
            if (b*batch+batch)>y_data.shape[0]:
                m,n = b*batch, y_data.shape[0]
            else:
                m,n = b*batch, b*batch+batch

            x_batch, y_batch, label_batch = x_data1[m:n,:], y_data1[m:n,:], label1[m:n,:]
            yield x_batch, y_batch, label_batch

def gfm_test(Data_file_name):
    
    Data_file_name = Data_file_name
    Data_file = h5py.File(Data_file_name +'.mat')
    Data_file.keys() 
    patch_1_test = Data_file['test_sar_patch']  #sar
    patch_2_test= Data_file['test_opt_patch']   #opt
    
    patch_1_test = np.array(patch_1_test)  #sar
    patch_2_test= np.array(patch_2_test)   #opt
    num1 = int(patch_1_test.shape[0]/batch_size)*batch_size
    num2 = int(patch_2_test.shape[0]/batch_size)*batch_size

    graph = tf.Graph()
    with graph.as_default():
        inputs_sar = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], name='inputs_sar')
        inputs_opt = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 1], name='inputs_opt')
        inputs_lab = tf.placeholder(tf.float32, [batch_size, 1], name='inputs_lab')
        
#        g_outputs = model.create_generator(inputs_sar, 1)
        match_loss,m_output= model.gfm_modelb(inputs_sar, inputs_opt, inputs_lab)
        out = tf.round(m_output)
        correct,ram = model.evaluation(out, inputs_lab)
        
        saver = tf.train.Saver()
        with tf.Session() as sess:  
#            saver.restore(sess, tf.train.latest_checkpoint(('ckpt_fm1b_' + Data_file_name.split('_')[1])))
            saver.restore(sess, tf.train.latest_checkpoint('ckpt_fm3br_6'))
#            saver.restore(sess, 'ckpt_fm1b/model.ckpt-4000')
            all_mout = np.array([])

            for i in range(num2):  #opt
                opt1 = patch_2_test[i,:,:,:]
                opt = np.tile(opt1,(num1,1,1,1))
                sar = patch_1_test[:num1,:,:,:]
                print(i) 
                patch_1_sar = sar #sar
                patch_2_opt = opt  #opt       
                y_test = np.zeros((num1,1))
                
    #            all_lab = np.array([])
    #            num = np.size(y_test)
                shuffle_test= gfm_shuffle(1,batch_size,patch_1_sar,patch_2_opt,y_test)
                for step1, (x_batch, y_batch, l_batch) in enumerate(shuffle_test):
                    feed_dict = {inputs_sar:x_batch, inputs_opt:y_batch, inputs_lab:l_batch}
                    result, p_m = sess.run([correct, m_output], feed_dict=feed_dict)
                    if step1 == 0:
                        all_mout = p_m
                    else:
                        all_mout = np.concatenate((all_mout, p_m), axis=0)
#                    true_count = true_count + result
                    if step1 % 100 == 0:
                        print('Step %d run_test: batch_precision = %.2f '
                                          % (step1, result/batch_size)) 
                all_mout = all_mout.T
                if i == 0:
                    results = all_mout
                else:
                    results = np.concatenate((results, all_mout), axis=0)
            return results

if __name__ == '__main__':
      Data_file_name = 'data_6_test_sar3'
      results = gfm_test(Data_file_name)
      sio.savemat('test_results/' + Data_file_name + '_gmap_r', {'data': results})
      
      Data_file_name = 'data_6_test_sar-3'
      results = gfm_test(Data_file_name)
      sio.savemat('test_results/' + Data_file_name + '_gmap_r', {'data': results})
      
      Data_file_name = 'data_6_test_sar5'
      results = gfm_test(Data_file_name)
      sio.savemat('test_results/' + Data_file_name + '_gmap_r', {'data': results})
      
      Data_file_name = 'data_6_test_sar-5'
      results = gfm_test(Data_file_name)
      sio.savemat('test_results/' + Data_file_name + '_gmap_r', {'data': results})
    
      Data_file_name = 'data_6_test_sar8'
      results = gfm_test(Data_file_name)
      sio.savemat('test_results/' + Data_file_name + '_gmap_r', {'data': results})






  