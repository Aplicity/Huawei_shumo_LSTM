# -*- coding: utf-8 -*-
import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd


class mnist_service(TfServingBaseService):

    def _preprocess(self, data):
        preprocessed_data = {}
        filesDatas = []
        #虽然这里有两层循环，实际中无论是在线推理还是批量推理，在这里都只能拿到一个Excel文件的数据
        #注意，虽然批量的输入是一个OBS里的文件夹，里面可能有多个文件，
        #但批量推理的输出也是一个文件夹，里面有各个文件对应的推理结果
        #理解了这个批量推理输入文件和输出文件一对一的关系，就能理解这里每次是只能拿到一个Excel文件的，
        #不能用来做多个Excel文件数据的合并操作
        for k, v in data.items():
            print("key:",str(k)," v:",str(v))
            for file_name, file_content in v.items():
                pb_data = pd.read_csv(file_content)
                file_data = np.array(pb_data.get_values(), dtype=np.float32)
                print("shape of "+ file_name +":", file_data.shape)
                filesDatas.extend(file_data)
                
                
        #开始对每个数据样本构造数据特征
        test_xs = []
        for i in range(len(filesDatas)):
            org_data = filesDatas[i]
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
        
        #整个预处理阶段就是为了将传入的Excel文件中的原始数据构建出喂给PB模型用来推理（预测）的数据。
        #所以这个构建出来的preprocessed_data数据必须和SavedModel模型的入口数据格式保持一致
        #这就我们在训练模型的代码中在保存模型的时候要特别用签名来指出该SavedModel模型的输入和输出
        #可以看到我们在开始的训练代码里面声明了该模型是有个名字叫'myInput'的输入，该输入是个[None,3]的numpy数组，数据类型是默认的float32
        #恰好我们在这里构造的filesDatas就是一个维度为[None,3]，数据类型为float32的numpy数组
        #满足了模型的签名要求        
        test_xs = np.array(test_xs, dtype=np.float32)  #将test_xs转换为float32的numpy数组类型
        preprocessed_data['myInput'] = test_xs        
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)
        return preprocessed_data


    #如果你的模型的输出的维度就是[None, 1]，这个函数就不用改了
    #如果你的模型的输出的维度不是[None, 1]，你可以通过这个后处理把他调整为[None,1]
    #不需要考虑怎么把结果存成CSV文件，部署批量推理的时候，会自动帮你把输出搞成文件
    def _postprocess(self, data):         
        infer_output = {}
        #在这里的data是SavedModel模型推理后返回的结果
        #由于在训练代码中我们在SavedModel模型的签名中说明么他是有一个名字叫myOutput的输出项
        #该项数据的维度是[None,1]
        for output_name, results in data.items():
            #按照模型输出的签名，这个output_name应该是“myOutput”，results的维度是[None,1]
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output