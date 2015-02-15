deep learning 新文章：
*********************

Deep Learning Shape Priors for Object Segmentation
==================================================

形状，这个是在模拟人的情况吗？
modeling high-dimensional richly
structured data

deep learning 为什么能建模高纬度数据？？


在网络中使用deeplearning 还是有些不太方便的，低层次
Switchable Deep Network for Pedestrian Detection
================================================

创新点：
=======
#. 这个是使用deep learning 做行人探测，首先使用卷积神经网络得到低层次特征，进一步使用限制玻尔兹曼机得到中层和高层特征, 用logistic 回归得到高层次特征。
#. 加入更多实际的因素，比如前景和背景，特征有无，从而建立如下模型：

.. math::

   E(x,y,h,s,m;\Theta) = \sum {k=1}^K s_kh_k^T(W_k(x\circ m_k)+b_k\sum{k=1}^Ksa_k^T(x\circ m_k) y^TU\sum{k=1}^Ks_kh_k- d^Ty

其中K是元素个数,在实际中表示有些人坐着，有些人站着，就是不同的K。:math:`\Theta= \{W,b,c,U,d\}` 是所有未知的待估计的参数。其中U是全连接矩阵。:math:`s_k` 判断某个元素是否激活；:math:`m_k` 是saliency map 代表前景还是背景。比如，0 代表背景，1代表目标行人。


算法步骤：
=========

#. 首先是卷积神经网络，得到轮廓信息。
#. 采取kMeans 算法对训练数据进行分组。 
#. 采用EM算法，估计S和参数 :math:`\Theta` 。
#. 使用 logistic 回归得到labels信息。
#. 使用误差熵进行反馈微调。

可能的创新点：
=============

#. 在人脸识别中遮挡部分也可以switchable 变量来表示遮挡。但是人脸是一个相对简单的问题。

疑问：
=====

#. 在算法预训练中S 和其他参数是怎样估计的，为什么首先估计S? 是不是使用决策树会更好一点。

#. 怎样区分前景和背景？

#. 这里的head  shoulder， upper body， 这些都是怎么来的信息？

扩展阅读：
=========

Learning and selecting features jointly with point-wise gated Boltzmann machines

利用特征对物体进行分类，前景特征和背景特征。


Facial Expression Recognition via a Boosted Deep Belief Network
===============================================================

早期的分类都是基于固定特征的分类，特征没有为具体的分类进行定制，这篇文章首先使用多个DBN，最顶层使用分类器进行判决。

L层的联合概率密度函数可以表示为：

.. math::

   Prob(H^0,H^1,...,H^L)=\prod_{l=0}^{L-2}Prob(H^l|H^{l-1})Prob(H^{L-1},H^L)

:math:`Prob(H^{l-1},H^l)` 和 :math:`Prob(H^{l},H^{l-1})` 分别表示为：

.. math::

   Prob(H^{l-1}|H^l)=\frac{1}{1+\exp((W^{(l,l+1)}H^l+b_h^{l+1}))}

在最高层次的输出 :math:`H^L` 可以表示为：

.. math::

   H^L=  W^{L-1,L}H^{L-1}

其中 :math:`W^{L-1,L}H^{L-1}` 是第最顶层的权值矩阵。

最后形成强分类器：

.. math::

   \xi_{strong}=\sum_{i=1}^{N_I}\beta_i[\frac{1}{1+\exp(-\sum_{j=1}^M\alpha_j sgn (W_j^{(L-1,L)}H_{i,j}^{L-1}-T_j) )}-E_i]^2

其中 :math:`\alpha` 和 T是弱分类器的门限，使用梯度下降方向。

弱分类器：

.. math::

   \xi_{}= \sum_{j=1}^M\alpha_j \sum_{i=1}^{N_I}\beta_i[\frac{1+ sgn (W_j^{(L-1,L)}H_{i,j}^{L-1}-T_j}{2}-E_i]^2

最后判决使用分类器，并使用联合梯度下降方法更新权值W：

.. math::

   \xi=\lambda\xi_{strong}+\xi_{weak}


top 两层使用boosting 结构， {0，L-2}层使用后向反馈算法。


算法整个流程：
=============

.. graphviz::

   digraph G {
      a [label="图像"];
      b [label="特征"  ];
      c [label="分类器（强分类器和弱分类器）"];
      a->b   [label="1.图像分块"];
      b->c    [label="2.学习层级的特征"];
      c->b [label="3.根据反馈调整前向特征"];
   }

疑问：
======

这里为什么要在输入的时候使用相互交叠的patches那？这样计算量不是很大吗？

算法创新点：
===========

#. 以前的算法都是基于特征学习、特征选择、分类约束，这三个过程是顺序并且是独立的，缺少全局反馈，此方法中形成整一个系统，使用全局微调，交替估计这三个状态的值，做出最优的分类。

#. 这个算法中使用局部面部图像，比如nose，eye and mouth，达到更好的面部特征识别。

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

Pedestrian Parsing via Deep Decompositional Network
===================================================


#. 遮挡估计，估计那个部位被遮挡了？
#. 

首先这个网络中使用的是什么？

算法步骤：

1. 估计只等当

.. graphviz::

   digraph G {
      a [label="图像"];
      b [label="特征"  ];
      c [label="判断是否遮挡？"];
      d [label="使用降噪autocoder补全component"];
      e[label="判断是否是背景？"];
      a->b->c->d->e->f
   }


全局调整是什么？

扩展阅读：
=========

Body part detectors trained using 3d human pose annotations.

利用人类3d骨骼轮廓探测身体每个部分。

Robust boltzmann machines for recognition and denoising.

使用RBM学习遮挡部分和非遮挡部分。

The shape boltzmann machine: a strong model of object shape

generative model for partsbased object segmentation

利用玻尔兹曼机不全离散数据。

Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion

使用降噪aucoder来恢复破坏的数据。我想这个可以用于人职业规划中。



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



