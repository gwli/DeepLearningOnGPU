*****
MxNet
*****

https://mxnet.incubator.apache.org/get_started/why_mxnet.html 同时提供了两种计算模式，基于 *mx.nd.array* 的命令式计算，以及 *mx.sym.Variable* 的符号式计算。
并且两者是相互交互的，在DL的网络拓扑是用 符号式计算来描述。 而train 更新 model 则是用 命令式计算。 同时提供一系列的DL的原语。例如

.. list-table::
    
   Varible
   FullyConnected
   Activation
   SoftmaxOutput

https://github.com/apache/incubator-mxnet, 这个是由appache 来支持的，基本上支持所有的脚本语言。包括perl

#. gluon MxNet 命令式接口，rtd http://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-gluon.html

#. 很好玩的Example https://github.com/apache/incubator-mxnet/tree/master/example
#. 安装教程 http://phunter.farbox.com/mxnet-tutorial1

如何扩展OP
==========

对于面向对象语言，无非就是继承与注册。
https://github.com/zhubuntu/MXNet-Learning-Note/blob/master/%E7%BC%96%E5%86%99%E8%87%AA%E5%B7%B1%E7%9A%84Operator.md


MXNet的framework
================


.. figure:: /Stage_3/mxnet/framework.png
   
   https://mxnet.incubator.apache.org/architecture/overview.html


内存管理，是各个系统设计头号问题。并行依赖执行，类似于fork-join这样执行系统。
是关键。 unreal Engine实现了一套，fork-join的机制。现在各个DL的Framework也都各个实现了一套。   
   
