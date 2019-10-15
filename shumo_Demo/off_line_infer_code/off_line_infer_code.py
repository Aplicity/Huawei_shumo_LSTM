# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 20:30:21 2019

@author: x00423910
"""


import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random


def preProcessDataForOffLineInfer(fileName):
    #导入测试数据
    with open(fileName, "r") as rf:
        file_data = pd.read_csv(rf)
        file_data = np.array(file_data.get_values(), dtype=np.float32)
        print("fileName:", fileName, "  shape of file data:", file_data.shape)
   
    #至此，我们把fileName文件里的数据都读到file_test_set_data中了
    #此时的file_test_set_data是个维度为[XXX, 18]的数组
    #开始对每个数据样本构造数据特征
    test_xs = []
    for i in range(len(file_data)):
        org_data = file_data[i]
        #和训练代码中预处理一致地，我们需要构建之前的3个数据特征
        feature_dis_2d = np.sqrt(np.power(org_data[12]-org_data[1],2)+np.power(org_data[13]-org_data[2],2)) 
        feature_RS_Power = org_data[8]
        feature_dis_3d = np.sqrt(np.power(org_data[12]-org_data[1],2)
                                 +np.power(org_data[13]-org_data[2],2)
                                 +np.power(org_data[14]-org_data[9],2))
        tmp_test_xs = [feature_dis_2d/100, feature_RS_Power, feature_dis_3d/100]
        test_xs.append(tmp_test_xs)
    #将test_xs转换为numpy数组类型
    test_xs = np.array(test_xs)  
    print("shape of test_xs:", test_xs.shape) 
    return test_xs


def infer(fileName):
    test_xs = preProcessDataForOffLineInfer(fileName)
    
    #从保存的模型文件中将模型加载回来
    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, ["serve"], "./model_0922")
      graph = tf.get_default_graph()
      x = sess.graph.get_tensor_by_name('haha_input_x:0')
      y = sess.graph.get_tensor_by_name('haha_output_y:0')
      infer_y_value = sess.run(y, feed_dict={x: test_xs})
      print("shape of infer_y_value:", infer_y_value.shape)
      #保存结果为csv文件      
      np.savetxt(fileName+"_infer_res.csv", infer_y_value, delimiter=',')
      

#test_xs = preProcessDataForOffLineInfer(os.path.join("test_set","test_112501.csv"))
infer(os.path.join("test_set","test_112501.csv"))
