迁移学习
========

https://github.com/gwli/transferlearning

小王爱迁移 https://zhuanlan.zhihu.com/p/30685086

解决的是如何复用知识。现在的深度学习模式，每一次都要从零开始。这样重复的工作量太大。
如何像传统的软件编程那样，实现知识代码框架的复用。这样可以大大加块深度学习的速度。
另外是现在已经积累了大量在的model,现在已经开始研究onnx,下一步自然是如何复用的问题了。

常见方法，在更一层的维度，抽象出更加commont的模型来进行复用。
另一种那就是两个模型领域也是可以相互转化的，转化公式。

#. 从模拟中学习
#. 适应新的域
#. 跨语言迁移知识

用深度网络，+输入生成识别网络，从而形成多任务网络。

.. figure:: /Stage_2/tl/tl_drive.png

迁移学习的简介: https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650724800&idx=2&sn=0d5e47e071c346eb4a485980deee5744&chksm=871b1dbeb06c94a81b74afcde32d759b7118d60e60b710570a2d6cf53fbe2a9badaed44c5f05#rd

一个重要的应用，那就是从模拟中学习，这样可以大大加快研发的速度。
例如一个仿真系统与真实系统还是有差别的，但是仿真系统中我们方便控制，并且快速的可重复。
这也是各家厂商开放自动驾驶模拟器的的原因。

另外一个就是机器人的学习训练，用实际环境的训练的模本会非常的高。

还有一个方向，那就是通用人工智能的开发。

提高模型的泛化能力
------------------

让模型更加稳健

多任务学习

持续学习

zero-shot学习


meta learning
=============

https://github.com/gwli/supervised-reptile



reptile
=======

怎么感觉像是在batch 之间又做了一次更新，并且用泰乐极数来解释。 

.. code-block:: python

   for in epoch:
       for i in k:
         backup=before_batch
       for in batch:
            W=updateOnLoss
         bacckup = backup +(w-backup)*outerstepsize

1. agent 是如何产生，固定的从一个备用池中选一下，还动态生的，
2. 参数是根据什么生成。 


meta学习也是进步一步把loss，W的更新再一步神经网络化。是不是直接用RNN来生成W呢。
提高模型的泛化能力
------------------
