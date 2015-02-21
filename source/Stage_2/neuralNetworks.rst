神经网络框架：
=============

.. graphviz::
   
   digraph{
   a [label="初始层"]
   b [label="隐层1（中层特征：color？？）"]
   c [label="隐层2（高层特征：shape？）"]
   d [label="决策层"]
   a->b  [label="卷积层"]
   c->b  [label="auto-coder"]
   a->c  [label="跨层网络实现多尺度特征"] 
   b->c  [label="层层抽象"]
   }

