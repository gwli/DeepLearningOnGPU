基本概念
========


softmax
-------

`逻辑函数 <https://zh.wikipedia.org/wiki/%E9%82%8F%E8%BC%AF%E5%87%BD%E6%95%B8>`_ 也是S函数，也就是sigmoid function.  :math:`P(t)=\frac{1}{1+e^{-t}}`


.. image:: /Stage_1/Logistic-curve.svg.png

softmax 函数就是归一化指数函数，是逻辑函数的一种推广，它可以任意维的函数类型映射变成(0,1)的S函数，并且所有函数元素的和为1。

.. math::
   
   softmax_i(a)=\frac{\exp{a_i}}{\sum\exp{a_i}}


用python进行函数计算的示例

.. code-block::

   import math
   z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
   z_exp = [math.exp(i) for i in z]  
   print(z_exp)  # Result: [2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09] 
   sum_z_exp = sum(z_exp)  
   print(sum_z_exp)  # Result: 114.98 
   softmax = [round(i / sum_z_exp, 3) for i in z_exp]
   print(softmax)  # Result: [0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]


Cross-Entroy
------------

这个概念是由香农信息量 :math:`log\frac{1}{p}` 。 cross-Entory 又叫相对熵或者相对熵。 

https://hit-scir.gitbooks.io/neural-networks-and-deep-learning-zh_cn/content/chap3/c3s1.html

.. math::

   \sum_{k=1}^N p_k\log_2 \frac{1}{q_k}

交叉熵可在神经网络(机器学习)中作为损失函数，p表示真实标记的分布，q则为训练后的模型的预测标记分布，交叉熵损失函数可以衡量p与q的相似性。交叉熵作为损失函数还有一个好处是使用sigmoid函数在梯度下降时能避免均方误差损失函数学习速率降低的问题，因为学习速率可以被输出的误差所控制。

交叉熵越低说明越相似。主要是度量两个序列相似度或者相关性大小的一种测量。


交叉熵与熵相对，如同协方差与方差.

熵考察的是单个的信息，交叉熵考察的是两个的信息（分布）的期望。

张量
====

`张量 <https://zh.wikipedia.org/wiki/%E5%BC%B5%E9%87%8F>`_ 在数学里，张量是一种几何实体，或者说广义上的“数量”。张量概念包括标量、矢量和线性算子。张量可以用坐标系统来表达，记作标量的数组，但它是定义为“不依赖于参照系的选择的”.

我们简单的理解例 C/C++中的类对象在数学上可以用张量表示。 即有数据结构，同时自带运算。

.. code-block:: c
   
   Tensor(
       Allocator *a,
       DataType type,
        const TensorShape & shape
  )


https://eigen.tuxfamily.org/dox/unsupported/classEigen_1_1Tensor.html
