deep learning 新文章：
*********************

Deep Learning Shape Priors for Object Segmentation
==================================================

形状，这个是在模拟人的情况吗？
modeling high-dimensional richly
structured data

deep learning 为什么能建模高纬度数据？？


在网络中使用deeplearning 还是有些不太方便的，低层次特征描述一些特征，高层次特征描述另一些内容特征

The Shape-Time Random Field for Semantic Video Labeling


这个文章中建立的是shape time model。我想这个必须是  暂时先不看。


Switchable Deep Network for Pedestrian Detection
================================================

这个是使用deep做行人探测，进一步得到一些高层次信息。可是这些高层次信息是什么？在隐藏层中怎样体现，是不是一些抽象信息。


扩展阅读：
=========

Learning and selecting features jointly with point-wise gated Boltzmann machines

利用特征对物体进行分类，前景特征和背景特征。


Facial Expression Recognition via a Boosted Deep Belief Network
===============================================================

早期的分类都是基于特征学习


这个是不是首先使用多个DBN，然后使用多个神经网络，使用弱分类器进行判决。

L层的联合概率密度函数可以表示为：

.. math::

   \Prob(H^0,H^1,...,H^L)=\prod_{l=0}^{L-2}\Prob(H^l|H^{l-1})\Prob(H^{L-1},H^L)

:math:`\Prob(H^{l-1},H^l)` 和:math:`\Prob(H^{l},H^{l-1})` 分别表示为：

.. math::

   \Prob(H^{l-1}|H^l)=\frac{1}{1\exp((W^{l,l+1}H^l+b_h^{l+1}))}

在最高层次的输出:mat:`H^L` 可以表示为：

.. math::

   H^L=  W^{L-1,L}H^{L-1}

其中:math:`W^{L-1,L}H^{L-1}` 是第最上层的权值矩阵。

形成弱分类器：

.. math::

   \xi_{strong}=\sum_{i=1}^{N_I}\beta_i[\frac{1}{1+\exp(-\sum\alpha j(Wj^(L1,L)H{i,j}^{L1}Tj) )}-E_i]^2

其中:math:`\alpha` 和 T是若分类器的门限。使用梯度下降方向。

使用联合梯度下降方向：

.. math::

   \xi\lamba\xi{strong}\xi{weak}


top 两层使用boosting 结构， 0，L2层使用 后向反馈算法。

这里DBN中输出的是什么？

使用卷积神经网络学习low-level feature。

这个网络中的switch指的是什么？

这个算法中使用部分面部图像，比如nose，eye and mouth。

算法整个流程：

.. graphviz:

digraph G {
   edge [fontname="FangSong"];
   node [shape=null, fontname="FangSong" size="20,20"];
   {
   a [label="图像"];
   b [label="特征"  ];
   c [label="分类器（强分类器和弱分类器）"];

   }

   a->b   [label="1.图像分块"];
   b->c    [label="2.学习层级的特征"];
   c->b [label="3.根据反馈调整前向特征"];
   }





疑问：
======

这里为什么要在输入的时候使用相互交叠的patches那？这样计算量不是很大吗？

算法创新点：
===========

#. 以前的算法都是基于特征学习、特征选择、分类约束，这三个过程是顺序并且是独立的，缺少全局反馈，此方法中形成整一个系统，使用全局微调，交替估计这三个状态的值，做出最优的学习


扩展阅读：
==========

#. On deep generative models with applications to recognition  
#. Disentangling factors of variation for facial expression recognition.

这两个文章使用层级的特征表示和logistic 回归交替估计用于目标分类，但是这个文章使用整张脸来做面部表情识别。

#. Facial action unit recognition with sparse representation. 
#. Sparse coding for flexible, robust 3d facial-expression synthesis.
这两篇文章使用sparse coding 呈现轮廓特征。

可能的创新点：
==============

#.  这个算法中全部使用强分类器，若分类器效果那？


Switchable Deep Network for Pedestrian Detection
================================================

这个文章中使用限制玻尔兹曼机，分出人的不同块。但是是怎样分块的？

Pedestrian Parsing via Deep Decompositional Network
===================================================

首先这个网络中使用的是什么？


全局调整是什么？

Discriminative Deep Metric Learning for Face Verification in the Wild
=====================================================================

Mahalanobis Distance Metric Learning

传统的Mahalanobis 距离试图找到方矩阵 :math:`M\in R^{d\times d}`


Hybrid Deep Learning for Face Verification
==========================================

在deep learning 中一直都使用卷积，怎样能确定那？哪里是特征，其实现在特征就是不明显了。

这里建立两张脸的特征，是什么意思？


扩展阅读：
==========

Deep convolutional network cascade for facial point detection  使用卷积神经网络来探测脸部区域。


这个文章看起来真是费劲，
 

DeepFace: Closing the Gap to Human-Level Performance in Face Verification
=========================================================================

使用三维脸建模，使用deep learning。


