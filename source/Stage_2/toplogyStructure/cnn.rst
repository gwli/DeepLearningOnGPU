卷积神经网络
============

.. graphviz::
   
   digraph Flow {
      Norm->Filter Bank->Non-Linar-> feature Pooling;
   }


卷积神经网络里，简单的理解那就是每一层的Filter如选，以及Pooling的大小如何选。最后要这些每一层的信息都要传递给决策层。


不管什么神经网络最后都链接上一个MLP来做判决的。

Normalization
-------------

各种各样的白平衡，对比度，去除噪声等等。来解决各种光照的影响。

Filter Bank
-----------

来解决尺寸的问题， 投影变换

Non-Linearity
-------------

主要是稀疏化，饱和，latel inhibition. 

Pooling
-------

实际就是聚合的过程， aggregation over space or feature type.

就是分块，然后用一点来代表这一块。什么样的映射更好，有什么更好的抽样理论或者压缩理论在这里。


神经网络知识本身就那些参数空间，其实就也是各种映射，如果找到有逆射那就更好了。

算法优点：

#. 针对图像中的像素点进行操作，通过卷积和下采样交替进行，在图像分类和识别中有重要应用。

#. 采用感受野和权值共享达到减小隐藏层的目的，同时起到旋转不变的作用。

#. down-sampling 达到减小分辨力的作用，同时也减小运算量。

#. 最后在经过 logistic regression 判断求所有layers的parameters。  %RED% 不难，建立一个cost函数，然后直接梯度计算%ENDCOLOR%


除了卷积网络本身还有什么方法可以来减少的连接数的。

.. math::

   \begin{figure}
     \centering
     \includegraphics[width=4cm]{CNN.jpg}\\
     \caption{卷积神经网络}
   \end{figure}

.. math::
 
   x_j^l = f(\sum_{i\in M_j}x_i^{l-1}*k_{ij}^l+b_j^l)

其中 :math:`M_j` 表示选择的输入maps的集合。（对于图像处理，是获取边缘信息。）

此时的灵敏度可以表示为：

.. math::
 
   \delta_j^l = \beta_j^{l+1}(up(\delta^{l+1})\circ f\prime(u_j^l))

up(.)表示上采样操作。

Sub-sampling Layers 子采样层

.. math::
 
   x_j^l=f(\beta_j^l down (x_j^{l-1})+b_j^l)

其中 :math:`down(.)` 表示下采样函数。

.. graphviz::

    digraph CNN{
   rankdir=LR
   node[shape=box]
   subgraph clusterA {
   
   x_1->y_1 [label="w_11"]
   x_2->y_1  [label="w_21"]
   x_2->y_2  [label="w_22"]
   x_3->y_2  [label="w_32"]
   label="layer1"
   subgraph clusterB {
    y_1
   
   y_2
   label="layer 2 maxpooling"
   }
   }
   y_1->y
   y_2->y
   }


照片实别的过程
==============

.. graphviz::

   digraph flow {
      Pixel->edge->texon->motif->part->object;
   }


组成
=====

卷积类型，每一次池化的大小，都是要原因的。池化的大小是不是可以用采用定理来决定 。

每一部都是这么实现了。

