tensorflow
**********

很好的资源库: https://github.com/fendouai/Awesome-TensorFlow-Chinese

不错的基本原理的书: https://github.com/exacity/deeplearningbook-chinese
R01_ 是Tensorflow的中文社区， tensorflow采用数据流图表来描述数学计算。

节点用来表示施加的数学操作，也可以表示数据输入的起点或输出的终点。或者是持久的变量，tensor.

而线用来输入输出之间的关系。


context -> ssession.
tensor 来表示数据，
通过变量 维护状态
用feed与fetch 可以为任意操作赋值 与取值。

Placeholder与tf.variable的区别


神经网络库实现的难点，一个是计算的并行化，那另一个那就是variable空间的引用。也就决定了网络的拓扑的如何连线。

如何共享变量
============

就是通过变量的命名空间，来实现的,http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/variable_scope.html



tf.variable_scope 定义一个命名空间，tf.get_variable就在当前空间下搜索该变量。 如果想复用，就得scope.reuse_variables() 来实现

.. code-block:: python
   def conv_relu(input,kernel_shape,bias_shape):
        weights = tf.get_variable("weights",kernel_shape,initializer=tf.random_normal_initializer())
        biases  = tf.get_veriable("biases",bias_shape,initializer=tf.random_normal_initializer())
        conv = tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding='SAME')
        return tf.nn.relu(conv,biases)


   def my_image_filter(input_images):
       with tf.variable_scope('conv1'):
            # variable will be created with name "conv1/weights","conv1/biases"
            relu1 = conv_relu(input_images,[5,5,32,32],[32])
       with tf.variable_scope('conv2'):
            # variable will be created with name "conv2/weights","conv2/biases"
            return = conv_relu(relu1,[5,5,32,32],[32])


   result1 = my_image_filter(image1)
   result1 = my_image_filter(image2)
   #Raised varibleError(.. conv1/weights already existes)

   with tf.variable_scope("image_filters") as scope:
        result1 = my_image_filter(image1)
        scope.reuse_variables()
        result2 = my_image_filter(image2)

tf.variable_op_scope
tf.op_scope
tf.name_scope
tf.variable_scope


tf.concat 

.. code-block:: python

   t1 = [[1,2,3],[4,5,6]]
   t2 = [[7,8,9],[10,11,12]]
   #concat_deim 0 表示行，1表示列
   tf.concat([t1,t2],0) ==> [[1,2,3,],[4,5,6],[7,8,9],[10,11,12]]
   tf.concat([t1,t2],1) ==>[[1,2,3,7,8,9],4,5,6,10,11,12]]
   # < 1.0
   tf.concat(0,[t1,t2]) ==> [[1,2,3,],[4,5,6],[7,8,9],[10,11,12]]


各种矩阵的形式转换
==================

对于一个默认的列表 [1,2,3,5,6,7,8,9]，变成矩阵有几种变法:

#.    [[1,2,3,4,5,6,7,8,9]]  也就是1 * 9   tf.expand_dims(list,0)
#.    [[1],[2],[3],[4],[5],[6],[7],[8],[9]]  9* 1 tf.expand_dim(list,1)
#.    
要变成一个矩阵，就相当于列表的列表，[[1,2,3,4,5,6,7]] 这在tensorflow中叫expand_dims。 

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
tf.run 就相当于求值替换执行。用 eval 用这词就更容易理解了。 并且指定了返回值。 

tf.train.Saver 
   用于何存变量

而矩阵乘法可以用来表征 n*m 的网络连接。
#. 初始化变量
#. 网络拓扑
#. Loss函数
#. 优化方法

*global_steps*  用于全局的计数器

tensorboard 的用法
==================

http://ischlag.github.io/2016/06/04/how-to-use-tensorboard/

.. code-block:: python

   #Tensorflow summaries are essentially logs. And in order to write logs we need a log writer (or what it is called in tensorflow) a SummaryWriter. So for starters, we’ll add the following line before our train loop.
   
   writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
   #This will create a log folder and save the graph structure. We can now start tensorboard.
   tensorboard --logdir=run1:/tmp/tensorflow/ --port 6006

TensorFlow四种Cross Entropy算法实现和应用
=========================================

http://www.weibo.com/ttarticle/p/show?id=2309404047468714166594

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
