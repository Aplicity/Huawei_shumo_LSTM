# Huawei_shumo_LSTM
题目详细和数据请求请点击超链接：
[第十六届（2019）中国研究生数学建模竞赛-A题](https://developer.huaweicloud.com/competition/competitions/1000013923/introduction)

## 文件说明
* train_set -- 训练集数据（数据太大，请点击上面超链接下载或百度云：链接:https://pan.baidu.com/s/1b3NLZrpJJcl2t7ZHfDYibA  密码:dpob）
  -  train_597801.csv
  -  ...
  - train_598501.csv
* test_set -- 验证集数据
  - test_112501.csv
  - test_115001.csv
  
 * 论文及题目
  - 2019年A题  无线智能传播模型.pdf（pdf版题目）
  - 2019年A题  无线智能传播模型.docx（word版题目）
  - 基于LSTM无线智能传播模型.pdf（论文报告）
  
* ModelArts平台使用指导-“华为杯”第十六届中国研究生数学建模竞赛.pdf（华为云ModelArts平台使用指导）
* baseline -- 大赛需要提交的线上demo
  - variables -- 模型保存文件
  - config.json -- 环境配置文件
  - customize_service.py -- 模型代码脚本
  - saved_model.pb -- 模型保存文件
  
* shumo_Demo -- 华为为本次大赛提供的线下demo
* 产出 -- 个人本次大赛产出
  - 基于LSTM无线智能传播模型.pdf（论文报告）
  - obs -- 对应 ./shumo_Demo/modelarts_upload_pkg/model_0922
* 人家参考 -- 大赛其他队员赛后共享
  - utils.py -- 模型代码
  - mybaseline -- 对应baseline（大赛需要提交的demo）

以下为本项目代码脚本
* 基于tf的LSTM.ipynb
* LSTM.ipynb
* customize_service.py

## 注明
* train_set在百度云下载后，同目录有一个叫source_data.csv的文件，实则为train_set所有数据连接在一起，单独保存。
* 经大赛友人说明，本次数据取自成都。同时也提供了R语言绘图代码，见 plot_geo.Rmd



  - 2019年A题  无线智能传播模型.pdf（pdf版题目）百度云
  - 2019年A题  无线智能传播模型.pdf（pdf版题目）
