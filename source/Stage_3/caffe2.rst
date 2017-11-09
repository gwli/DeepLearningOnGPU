******
Caffe2
******

.. image:: /Stage_2/caffe2/operator_funcctionality_comparison.png

Caffe2 采用算子符号流图来设计网络拓扑。 基本概念

#. Workspace, workspace 来管理这些blob
#. blob => tensor  *blob* 只是一块命名的内存空间。blob 包含一个tensor.
#. Nets
#. Operators 例如 FC,Sigmoid 这些原语。


http://caffe2.ai/
https://github.com/caffe2/caffe2

caffe2 的目标定位类似于tensorrt. 但是也能用于训练。并且

https://www.jiqizhixin.com/articles/2017-04-20-6，想要实现的一次编码，任意运行。

基本教程 https://caffe2.ai/docs/tutorial-basics-of-caffe2.html
