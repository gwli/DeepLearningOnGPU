https://github.com/Microsoft/CNTK
安装需要额外安装一个anaconda python 就行了。
同时用VS 来做开发，在tools.options>python 添加新环境变量，同时在工程文件中使用右键选择新的环境变量就行了。

使用的基本步骤，那就是建立网络拓扑，指定参数训练，输入适配。

cntk的主要用途
#. Feed forward
#. CNN
#. RNN
#. LSTM
#. Sequence-to-Sequence

用法也应该类似于CUDA了，有自己的API来指定结构。

有两种方式，在python里，是靠重载几个子命令来实现网络拓扑的，例如linear_layer，input_variable,dense_layer,fully_connected_classifier_net等等。
