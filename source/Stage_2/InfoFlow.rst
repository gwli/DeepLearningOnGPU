*********
info flow
*********

观察一个系统的核心，那就是看其信息流是怎样的，信息发生了哪些transform.

.. image:: /Stage_2/autoencoder.png

auto coder， sparse coding。中层打label。


深度学习读书笔记之 `AE（自动编码） <http://blog.csdn.net/mytestmy/article/details/16918641>`_ 
=============================================================================================

并且用 AE来自动学习一个物体的表征方式，是一个不错方向。
`Representation Learning: A Review and New Perspectives <https://arxiv.org/abs/1206.5538>`_


#. Sparse AE
#. Denoise AE
#. contractive AE
#. Stacked AE
#. Deep AE    

用于特征抽取特别有效。


.. code-block:: python

   model.parameters
   optimizer.RMSprop(model.parameters,lr=0.1,weight_decay=0.1)

AE对图形不同位置和方向进行边缘检测。另外可用于检测图像隐藏的相关性，和PCA类似。autoencoders  利用稀疏性来对规则化。

只是da的多层堆在一起，每一层算完之后，再整体就像MLP一样计算一遍。autoAE要利用约束防止训练单位阵。

Denoising Autoencoders 原理：

使用code和decode 来求解 :math:`w_{ij}` .

具体如下：

对于输入x建立神经网络：

.. math::
 
   y=s(Wx+b)


其中s是非线性函数：期望得到输出：

.. math::
 
   z=s(W^{T}y+b)


最后使用不同的reconstruction error 作为约束函数：

均方误差（square error ） 和交叉熵

最后使用均方误差作为约束函数：

.. math::
 
   L(x,z)=||x-z||^2


或者使用 `交叉熵(cross-entropy) <http://zh.wikipedia.org/wiki/%E7%9B%B8%E5%AF%B9%E7%86%B5>`_ 作为约束函数：

.. math::
 
   L_H(x,z)=-\sum_{k=1}^d[x_klog{z_k}+(1-x)log(1-z_k)]

square error 只适用于高斯误差，所以cross-entropy 更加鲁棒些。



L1,L2正则化

我自己的理解就是约束优化函数出现一些没有意义的解。常规的主要L2正则化:

.. math::
 
   J_R(w)=\frac {1}{n}||y-xw||^2+\lambda ||w||^2

但是如果对于高维数据一般存在稀疏性，一般加入L1正则化：

.. math::
 
   J_R(w)=\frac {1}{n}||y-xw||^2+\lambda ||w||^1

2006年tao证明L1正则化等价于0 范数，说明其具有稀疏性。 而稀疏就意味着80/20原则，也就是起关键作用就那少数几个参数。同时也就意味着降维。

另外在实现的时候，在层与层之间传输的数据一般是用张量来表示，如何化信息流，在实现一方便数据的输入适配，同时也是信息抽象的过程。例如在caffe里采用blob的概念来打包这个数据格式，并用protobuf来实现。



AE与GAN的同与不同
=================

https://zhuanlan.zhihu.com/p/27549418

#. 一个是AE可以多层的，GAN现在还相当于两层的，相当于是把feedback放到foward中。
#. AE更像一个压缩感知,把一个高维数据压缩到低维，而GAN更相反，是从低维数据生成高维数据。
   而这个本质就在于层层之间转换的这个基是什么。 GAN相当于只是找到 :math:`y=F(x)G(X)` 找到 :math:`G(x)` 
#. Vairational AE 也是一个生成模型，而原始版本的 AE只能重构原始数据。
   是在编码过程给它增加一些限制，迫使其生成的隐含向量能够粗略的遵循一个标准正态分布。
   每一次生成两处向量，一个表示均值，一个表示标准差，然后通过两个统计量来合成隐含向量。 

目前问题
=========

#. 如何构造每一个感知器，层与层之间如何连接，需要多少层？最简单的方法，每一层之间都是全连接，通过增加层数，来解决所有问题，这样的计算太大。因此如果全联接，要尽可能用剪枝算法，来减少不必要的连接。并且到底需要多少层都是根据实际的情况来的。

#. 另外一部分那就是如何反馈，现在看到的都是利用的梯度，建立一个cost函数，然后把所有的参数都放进去，然后求梯度，theano采用链式求导，也就是复合函数求导。只要都是表达式，就可以求导，一次更新所有参数。所以反馈机制，是整体的cost,还是每一层都可以有一个cost,并且反馈采用梯度，还是牛顿法等。

#. 多层之间是可以混合的，例如一层采用卷积，减少到一定程度，然后采用自动编码，最后是隐藏层等。另外神经元之间的横向连接如何建立，也就是层内部关联。


根据目前的论文来看，浅层更多的是通用的feature,深层是一些高阶的feature。 如果想迁移学习复用的话，就要根据相似度来选择到底需要复用几层。 

