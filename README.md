# 数据挖掘大作业

## 选题介绍

- [题目介绍](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition)

- 参考链接：[猫狗大战](https://zhuanlan.zhihu.com/p/25978105)

- [数据下载](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## 工程部署

- 目前在python3.5.4，tensorflow1.3.0，keras2.0.6环境下能正常运行；

- 下载训练集和测试集，训练集解压到train文件夹下，测试集解压到test文件夹下；

- 运行resize_cp.py，对原始数据进行简单预处理；

- 运行write_fv.py生成特征向量文件fv_Xception.h5；

- 运行cat_dog.py生成预测表格pred.csv，可提交kaggle上检验结果。

- 需要安装的python库：pillow，Pandas
