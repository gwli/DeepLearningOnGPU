CNTK
****

https://github.com/Microsoft/CNTK
安装需要额外安装一个anaconda python 就行了。
同时用VS 来做开发，在 *tools.options>python* 添加新环境变量，同时在工程文件中使用右键选择新的环境变量就行了。

#. 安装VS2017 并选择 *Data Processing* 模块。
#. pip install cntk  from https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-Windows-Python?tabs=cntkpy22
#. pip install keras
#. pip install tensorflow-gpu
#. set the backend of keras :command:`set KERAS_BACKEND=cntk`
   
setup on Azure
==============

#. Deploy CNTK Azure web API
#. Data Science Virtual Machine
 


使用的基本步骤，那就是建立网络拓扑，指定参数训练，输入适配。

cntk的主要用途

#. Feed forward
#. CNN
#. RNN
#. LSTM
#. Sequence-to-Sequence

用法也应该类似于CUDA了，有自己的API来指定结构。

有两种方式，在python里，是靠重载几个子命令来实现网络拓扑的，例如linear_layer，input_variable,dense_layer,fully_connected_classifier_net等等。

采用方式类似于theano的方式，利用图来构造网络结构。
所以也就是构造图的过程。 利用input,parameter,以及基本函数来当做原语来指定输入输出。
