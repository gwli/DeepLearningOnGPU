tensorflow
==========

R01_ 是Tensorflow的中文社区， tensorflow采用数据流图表来描述数学计算。

节点用来表示施加的数学操作，也可以表示数据输入的起点或输出的终点。或者是持久的变量，tensor.

而线用来输入输出之间的关系。


context -> ssession.
tensor 来表示数据，
通过变量 维护状态
用feed与fetch 可以为任意操作赋值 与取值。

Placeholder与tf.variable的区别

网络的基本组成
==============

现在基本上所有神经网络库都采用符号计算来进行拓扑的构造。所以要想研究新的网络拓扑，添加新原语，就得能够添加新的原语。
也就是能够扩展基本原语。

tf.Variable
   主要在于一些可训练变量（trainable variables），比如模型的权重（weights，W）或者偏执值（bias）；
   声明时，必须提供初始值；

tf.placeholder
  用于得到传递进来的真实的训练样本： 不必指定初始值，可在运行时，通过 Session.run 的函数的 feed_dict 参数指定；

tf.name_scope
   可以用来表示层的概念
tf.run 就相当于求值替换执行。用 eval 用这词就更容易理解了。

而矩阵乘法可以用来表征 n*m 的网络连接。
#. 初始化变量
#. 网络拓扑
#. Loss函数
#. 优化方法



基本组成
--------

#. 变量
  + tf.Variable  

用点
tensorflow与thenao基本是一致的，都是利用图来构建计算模型，这些在python里实现，而真正的计算独立来实现的。 python 只是相当于一个控制台而己。

这样结构有点类似于符号计算的味道了。
在tensorflow.

变量就相当于符号。 各种placeholader,以及各种运算都符号化了。

这也正是编程语言的下一个趋势，算法的描述。

先构建computation graph,然后初始化，再开始运行。 

根据神经网络的结构来，


源码解读
========

http://www.cnblogs.com/yao62995/p/5773578.html

References
==========

.. _R01: http://www.tensorfly.cn/
