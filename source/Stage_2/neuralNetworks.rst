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

工具：神经网络设计工具：（CAFFE，theao）。可视化：信息流的可视化（编码），训练时的可视化(过拟合，迭代下降)。 网络拓扑本身的可视化。





 


