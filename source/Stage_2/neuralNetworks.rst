神经网络笔记
************

神经网络框架：
=============

.. graphviz::
   
   digraph{
      a [label="初始层"]
      b [label="隐层1（中层特征：边缘？？）"]
      c [label="隐层2（高层特征：shape？）"]
      d [label="决策层"]
      a->b  [label="卷积层"]
      c->b  [label="auto-coder"]
      b->d  [label="跨层网络实现多尺度特征"] 
      b->c  [label="使用sparse coding层层抽象"]
      c->d  [label="全局特征决策"]
   }

工具：
=====

#. 神经网络设计工具：（CAFFE，theao）。
#. 可视化：信息流的可视化（编码），训练时的可视化(过拟合，迭代下降)。
#. 网络拓扑本身的可视化。

feedback：
=========

#. 从整个学习上说，就是一个偏导函数量。要解决的一个问题包括两方面：第一个是学习速度问题，第二个是防止震荡。目前用的都是基于导数的优化。
#. 受到cost函数影响是约束问题的松和紧。 cost ， activate MSE(最小二乘），线性的max 函数 cross-entropy, sigmoid函数 log-likelihood, softmax函数。
#. 超参数的估计，目前是过拟合产生一个主要原因。
#. 具体采用哪一种组合呢，就看你采用哪一种解析了，如果想用要概率模型就要用softmax组合。

神经元结构：
============

#. 神经元从结构功能上来说：包括感知神经元和激活神经元。
#. 从感知神经元包括：线性的，卷积，限制玻尔兹曼机。
#. 激活函数包括（只要是连续即可）：sigmod，tanh，softmax，retified linear 。

info flow：
==========

auto coder， sparse coding。中层打label。

容量度量：
=========

深度网络能够识别分类多少信息？也就是如何度量一个网络的识别能力。



