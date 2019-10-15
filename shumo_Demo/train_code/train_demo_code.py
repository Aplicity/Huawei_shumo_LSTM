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



def preProcessDataForTraining():
    #导入训练数据集
    total_train_set_data = []
    for train_set_file_name in os.listdir("train_set"):
        with open(os.path.join('train_set',train_set_file_name), "r") as rf:
            file_data = pd.read_csv(rf)
            file_data = np.array(file_data.get_values(), dtype=np.float32)
            print("fileName:", train_set_file_name, "  shape of file data:", file_data.shape)
            total_train_set_data.extend(file_data)
    total_train_set_data = np.array(total_train_set_data)
    print("shape of total_train_set_data data:", total_train_set_data.shape)
    #至此，我们把“train_set”文件夹下的数据都读到total_train_set_data中了
    #此时的total_train_set_data是个维度为[15284, 18]的数组
    #开始对每个数据样本构造数据特征
    total_train_xs = []
    total_train_ys = []
    for i in range(len(total_train_set_data)):
        org_data = total_train_set_data[i]
        #假设我们觉得数据中的四个地理坐标可以构造一个距离特征
        feature_dis_2d = np.sqrt(np.power(org_data[12]-org_data[1],2)+np.power(org_data[13]-org_data[2],2))     
        #假设我们觉得RS Power这个特征也很重要
        feature_RS_Power = org_data[8]
        #假设我们觉得三维空间中的这个距离也很重要
        feature_dis_3d = np.sqrt(np.power(org_data[12]-org_data[1],2)
                                 +np.power(org_data[13]-org_data[2],2)
                                 +np.power(org_data[14]-org_data[9],2))
        #现在我们构造三个特征了
        tmp_train_xs = [feature_dis_2d/100, feature_RS_Power, feature_dis_3d/100]
        #我们期望预测的就是第18列的RSRP
        tmp_train_ys = [org_data[17]]
        total_train_xs.append(tmp_train_xs)
        total_train_ys.append(tmp_train_ys)
    #将total_train_xs、total_train_ys转换为numpy数组类型
    total_train_xs = np.array(total_train_xs)
    total_train_ys = np.array(total_train_ys)    
    print("shape of total_train_xs:", total_train_xs.shape)    
    print("shape of total_train_ys:", total_train_ys.shape)
    
    return total_train_xs, total_train_ys
    
    


def train():
    #******* 1. 构建网络 *******    
    INPUT_FEATURE_NUM = 3  #因为预处理中只构造了3个特征
    OUTPUT_FEATURE_NUM = 1 #因为要预测的只有1个值
    x = tf.placeholder(tf.float32, [None, INPUT_FEATURE_NUM], name = "haha_input_x")    
    #最简单的网络，就是一个矩阵乘法 y = x*W+b，
    #其中X是我们的Placeholder，用来接输入数据的
    #y是整个网络的输出节点
    W = tf.Variable(tf.random_normal([INPUT_FEATURE_NUM, OUTPUT_FEATURE_NUM]))
    b = tf.Variable(tf.random_normal([OUTPUT_FEATURE_NUM]))
    y = tf.add( tf.matmul(x, W), b, name = "haha_output_y")
    
    #构造输错节点的Placeholder,用来接收标准答案数据的
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_FEATURE_NUM])
    #构造训练网络的cost,就是看预期答案y_和实际推理出来的答案y的差距是多少
    cost = tf.reduce_mean(tf.sqrt(tf.pow(y-y_,2)))
    #选择优化的算法使最小化cost    
    train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cost)
        

    
    #开始训练
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        #导入预处理之后的训练数据
        total_train_xs, total_train_ys = preProcessDataForTraining()
        for i in range(300):
            #在total_train_xs, total_train_ys数据集中随机抽取batch_size个样本出来
            #作为本轮迭代的训练数据batch_xs, batch_ys
            batch_size = 1000
            sample_idxs = random.choices(range(len(total_train_xs)), k=batch_size)
            batch_xs = []
            batch_ys = []
            for idx in sample_idxs:
                batch_xs.append(total_train_xs[idx])
                batch_ys.append(total_train_ys[idx])
            batch_xs = np.array(batch_xs)
            batch_ys = np.array(batch_ys)            
            #喂训练数据进去训练
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            #看经过这次训练后，cost的值是多少
            cost_value = sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys}) 
            print("after iter:",i, " cost:", cost_value)
        #训练完成之后，调用函数将网络保存成tensorflow的SavedModel格式的模型
        #注意这里保存的模型签名，就是告诉别人我们模型的输入和输出
        #如下就是告诉别人我们这个模型只有一个输入，这个输入就是x节点，别人用这个模型的
        #时候就要准备一个输入的Map，这个Map里面有一个key为“myInput”的项，对应的值应该
        #是一个维度为[None, INPUT_FEATURE_NUM]的numpy数组;
        #同理，输出的签名就是在告诉别人调用我们这个模型后返回的数据是一个Map，其中有一
        #key为"myOutput"的项，该项的值是一个维度为[None, OUTPUT_FEATURE_NUM]的numpy数
        #组;
        #这个签名是通过x和y节点的来告诉别人这个模型的输入输出的数据格式和维度
        #其中x作为一个数据输入节点，必须是placeholder类型，从上面也可以看到其维度是我们
        #自己根据自己的数据特征设计的，而y的维度是由x维度是[None, INPUT_FEATURE_NUM]
        #又乘了一个[INPUT_FEATURE_NUM, OUTPUT_FEATURE_NUM]的W矩阵再加上一个[OUTPUT_FEATURE_NUM]
        #的b矩阵得出来的，所以y的维度是[None, OUTPUT_FEATURE_NUM];
        #注意上的y_也是一个维度是[None, OUTPUT_FEATURE_NUM]的placeholder，是为了用于喂y在训练集中的预期答案，
        #再跟预测出来的y进行计算得到优化目标cost函数的。
        #注意区分签名里的outputs里的是输出节点y,而不是用了喂预期数据的y_,因为在模型训练好，用于推理
        #的时候，你是无法得到测试集中的预期答案的。
        tf.saved_model.simple_save(
                sess,
                "./model_0922",
                inputs={"myInput":x}, #这个就是我们的模型签名，告诉别人我们模型输入是x节点
                outputs={"myOutput":y} #这个就是我们的模型签名，告诉别人我们模型输入是y节点
                )
        
        
train()