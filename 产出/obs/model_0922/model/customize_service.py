# coding=utf-8

import numpy as np
from model_service.tfserving_model_service import TfServingBaseService
import pandas as pd
import tensorflow as tf
import warnings
import os
warnings.filterwarnings('ignore')

class mnist_service(TfServingBaseService):

    def _preprocess(self, data):
        preprocessed_data = {}
        filesDatas = []
        for k, v in data.items():
            for file_name, file_content in v.items():
                pb_data = pd.read_csv(file_content)
                input_data = np.array(pb_data.get_values()[:,0:17], dtype=np.float32)
                print(file_name, input_data.shape)
                filesDatas.append(input_data)

        filesDatas = np.array(filesDatas,dtype=np.float32).reshape(-1, 17)
        preprocessed_data['myInput'] = filesDatas
        print("preprocessed_data[\'myInput\'].shape = ", preprocessed_data['myInput'].shape)

        return preprocessed_data


    def _postprocess(self, data):
        infer_output = {"RSRP": []}
        for output_name, results in data.items():
            print(output_name, np.array(results).shape)
            infer_output["RSRP"] = results
        return infer_output

    def load_data(self):
        DFs = []
        for fileName in os.listdir("train_set"):
            with open(os.path.join('train_set', fileName), "r") as fr:
                temp_df = pd.read_csv(fr)
                DFs.append(temp_df)
        df = pd.concat(DFs)
        return df

    def features_process(self, df):
        # 转换角度为弧度，方便计算机计算
        df['Azimuth'] = df['Azimuth'] / 180 * np.pi
        df['Electrical Downtilt'] = df['Electrical Downtilt'] / 180 * np.pi
        df['Mechanical Downtilt'] = df['Mechanical Downtilt'] / 180 * np.pi

        # 是否弱覆盖: is_PCR
        df['is_PCR'] = df['RSRP'].apply(lambda x: 1 if x < -103 else 0)

        # One Hot 地物类型: Clutter Index_x
        df['Clutter Index'] = df['Clutter Index'].astype(str)
        df = pd.get_dummies(df)

        # 目标与发射塔海拔高度差: d_A （米)
        df['d_A'] = df['Cell Altitude'] - df['Altitude']

        # 天线离目标有效高度 h_b （米）
        df['h_b'] = df['d_A'] + df['Height']

        # 目标与发射机水平距离 ： d （米）
        df['d'] = 5 * ((df['Cell X'] - df['X']) ** 2 + (df['Cell Y'] - df['Y']) ** 2) ** 0.5

        # 目标栅格与信号线相对高度 : d_h_v (米)
        df['d_h_v'] = df['h_b'] - df['d'] * np.tan(df['Electrical Downtilt'] + df['Mechanical Downtilt'])

        # 天线是否反射才能抵达目标上方 ：is_reflect
        df['is_reflect'] = df['d_h_v'].apply(lambda x: 1 if x < 0 else 0)

        # 信号线长度 ：L （米）
        df['L'] = (df['d'] ** 2 + (df['d'] * np.tan(df['Electrical Downtilt'] + df['Mechanical Downtilt'])) ** 2) ** 0.5

        # 目标栅格离天线距离 ：S (米)
        df['S'] = (df['d'] ** 2 + df['h_b'] ** 2) ** 0.5

        # 传播路径损耗：PL （dB）
        # PL = 46.3 + 33.9 * f - 13.82 * log10(h_b) + (44.9 - 6.55 * log10(h_b)) * log10(d)
        # h_b可能为负值，也就是天线的海拔低于目标海拔，取对数时报错。因此在计算时用绝对值计算；0值则用1替换，取对数后得值为0。
        # d可能为0，这时候目标与天线在同一栅格，同样用1替代，取对数后得值为0。
        df['temp_h_b'] = abs(df['h_b'])
        df['temp_h_b'] = df['temp_h_b'].replace(0, 1)
        df['temp_d'] = df['d'].replace(0, 1)

        df['PL'] = 46.3 + 33.9 * np.log10(df['Frequency Band']) - 13.82 * np.log10(
            df['temp_h_b']) + (44.9 - 6.55 * np.log10(df['temp_h_b'])) * np.log10(df['temp_d'] / 1000)

        df = df.drop(['temp_h_b', 'temp_d'], axis=1)

        # 理论RSRP ：my_RSRP (dBm)
        df['my_RSRP'] = df['RS Power'] - df['PL']

        # 发射机服务目标数量 ：N
        def get_number_for_CellToTarget(data):
            temp_df = data.groupby("Cell Index")['Cell Index'].count().to_frame()
            temp_df.columns = ['N']
            temp_df["Cell Index"] = temp_df.index
            temp_df = temp_df.reset_index(drop=True)
            data = pd.merge(data, temp_df, on='Cell Index', how='left')
            return data
        df = get_number_for_CellToTarget(df)

        # 小区栅格发射机数量 : N_c
        def get_NumberOfStation_forCell(data):
            temp_df = data[["Cell Index", "Cell X", "Cell Y"]]
            temp_df = temp_df.drop_duplicates()
            count_data = temp_df.groupby(["Cell X", "Cell Y"])["Cell Index"].count().to_frame()
            count_data.columns = ['N_c']
            count_data = count_data.reset_index()
            data = pd.merge(data, count_data, on=["Cell X", "Cell Y"], how='left')
            return data
        df = get_NumberOfStation_forCell(df)

    def clean_data(self, df):
        '''
        step 1: 剔除与数据定义相违数据。
            由于题目地物类型名称的编号对各种建筑有高度规定，
            比如地物类型编号为10的建筑高度定义为大于60米，
            因此在数据中可以把该类地物类型建筑高度小于60米的作为异常数据并进行剔除。
        step 2: 拉依达准则剔除极端数据
        '''
        mask_A = (df['Cell Clutter Index'] == 10) & (df['Building Height'] <= 60)
        mask_B = (df['Cell Clutter Index'] == 11) & (df['Building Height'] < 40)
        mask_C = (df['Cell Clutter Index'] == 11) & (df['Building Height'] > 60)
        mask_D = (df['Cell Clutter Index'] == 12) & (df['Building Height'] < 20)
        mask_E = (df['Cell Clutter Index'] == 12) & (df['Building Height'] > 40)
        mask_F = (df['Cell Clutter Index'] == 13) & (df['Building Height'] > 20)
        mask_G = (df['Cell Clutter Index'] == 14) & (df['Building Height'] > 20)
        mask = mask_A | mask_B | mask_C | mask_D | mask_E | mask_F | mask_G
        df = df[~mask]
        del mask_A, mask_B, mask_C, mask_D, mask_E, mask_F, mask_G

        # 拉依达准则剔除极端数据
        RSRP_mean = df['RSRP'].mean()
        RSRP_std = df['RSRP'].std()
        upper_value = RSRP_mean + 3 * RSRP_std
        lower_value = RSRP_mean - 3 * RSRP_std
        mask = (df['RSRP'] >= lower_value) & (df['RSRP'] <= upper_value)
        df = df[mask]


    def run(self):
        # 相关分析
        feature_to_corr = dict()
        features = list(df.columns)
        features.remove('RSRP')
        features.remove('Cell Index')

        for col in features:
            corr = df[[col, 'RSRP']].corr().get_values()[0, 1]
            feature_to_corr[col] = corr

        data = sorted(feature_to_corr.items(), key=lambda d: abs(d[1]), reverse=True)
        data = pd.DataFrame(data=data, columns=['feature', 'correlation'])
        print(data.head(20))

        ## 挑选特征
        features = ['Frequency Band', 'RS Power', 'Cell Clutter Index', 'Building Height',
                    'Clutter Index_10', 'Clutter Index_11', 'Clutter Index_12',
                    'Clutter Index_13', 'Clutter Index_14', 'Clutter Index_15', 'Clutter Index_2',
                    'Clutter Index_5', 'Clutter Index_8', 'd_A', 'h_b', 'd', 'd_h_v', 'is_reflect',
                    'L', 'S', 'PL', 'my_RSRP', 'N', 'N_c', 'is_PCR']
        Xs = np.array(df[features].get_values(), dtype=np.float32)
        Ys = np.array(df['RSRP'].get_values(), dtype=np.float32)

        del df
        # np.save('Xs', Xs)
        # np.save('Ys', Ys)

    def StandardSclaer(self, xs, ys):
        '''
        数据标准化
        '''
        x_mean = xs.mean(axis=0)
        x_std = xs.std(axis=0)
        y_mean = ys.mean(axis=0)
        y_std = ys.std(axis=0)
        ss_xs = (xs - x_mean) / x_std
        ss_ys = (ys - y_mean) / y_std
        return ss_xs, ss_ys, x_mean, x_std, y_mean, y_std

    def get_train_data(self, data_x, data_y, batch_size, time_step):
        '''
        获取训练数据
        :param data_x: RSRP数据
        :param data_y: 输入特征
        :param batch_size: batch大小
        :param time_step: 时步数
        :return:
        '''
        batch_index = []
        train_xs, train_ys = [], []
        n = len(data_x)
        for i in range(n - time_step):
            if i % batch_size == 0:
                batch_index.append(i)
            x = data_x[i: i + time_step]
            y = data_y[i: i + time_step, np.newaxis]
            train_xs.append(x.tolist())
            train_ys.append(y.tolist())
            #     batch_index.append((len(train_xs)-time_step))
        return batch_index, train_xs, train_ys

    def get_test_data(self, data_x, data_y, time_step):
        '''
        获取测试数据
        '''
        size = (len(data_x) + time_step - 1) // time_step
        test_xs, test_ys = [], []
        for i in range(size - 1):
            x = data_x[i * time_step:(i + 1) * time_step]
            y = data_y[i * time_step:(i + 1) * time_step]
            test_xs.append(x.tolist())
            test_ys.extend(y)

        test_xs.append((data_x[(i + 1) * time_step:]).tolist())
        test_ys.extend((data_y[(i + 1) * time_step:]))

        input_size = len(x[-1])
        row = time_step - len(test_xs[-1]) % time_step
        test_xs[-1] = np.concatenate((test_xs[-1], np.zeros((row, input_size))), axis=0)
        test_ys.extend([0] * row)
        return test_xs, test_ys

    def main(self):
        ## def param
        rnn_unit = 100  # hidden layer units
        input_size = 25
        output_size = 1
        lr = 0.001
        time_step = 3
        batch_size = 32
        EPOCHs = 100

        np.random.seed(1)
        np.random.shuffle(xs)
        np.random.seed(1)
        np.random.shuffle(ys)
        xs, ys, x_mean, x_std, y_mean, y_std = StandardSclaer(xs, ys)

        # 构建模型
        with tf.name_scope('placeholder'):
            X = tf.placeholder(tf.float32, shape=[None, time_step, input_size], name='X')
            Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size], name='Y')
            inputs = tf.reshape(X, [-1, input_size])  # 换成2D, [batch_size * time_step , input_size]

        with tf.name_scope('LSTM'):
            # 输入层权重
            W_in = tf.Variable(initial_value=tf.random_normal(shape=(input_size, rnn_unit), mean=0.0, stddev=0.1),
                               name='W_in')
            b_in = tf.Variable(initial_value=tf.zeros(rnn_unit), name="b_in")

            input_rnn = tf.matmul(inputs, W_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 换回3D ,[batch_size , time_step, rnn_unit]

            ## 添加lstm 单元
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, reuse=tf.AUTO_REUSE)
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
            outputs = tf.reshape(output_rnn, [-1, rnn_unit])

            ## 输出层权重
            W_out = tf.Variable(initial_value=tf.random_normal(shape=(rnn_unit, 1), mean=0.0, stddev=0.1), name='W_out')
            b_out = tf.Variable(initial_value=tf.zeros(1), name="b_out")
            predictions = tf.matmul(outputs, W_out) + b_out

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(tf.reshape(predictions, [-1]) - tf.reshape(Y, [-1])))

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

        # 训练模型
        # 训练LSTM
        with tf.Session() as sess:
            with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
                init = tf.global_variables_initializer()
                sess.run(init)
                batch_index, xs, ys = get_train_data(xs, ys, batch_size, time_step=3)
                for epoch in range(EPOCHs):
                    total_loss = 0
                    for i in range(len(batch_index) - 1):
                        batch_x = xs[batch_index[i]: batch_index[i + 1]]
                        batch_y = ys[batch_index[i]: batch_index[i + 1]]
                        _, curr_loss = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
                        #                 pred = sess.run(predictions , feed_dict = {X: batch_x})

                        #             print(epoch+1)

                        # 保存模型
                tf.saved_model.simple_save(sess, "./model/",
                                           inputs={"myInput": X},
                                           outputs={"myOutput": predictions})


