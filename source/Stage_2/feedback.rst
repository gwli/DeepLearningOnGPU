Backpropgation
==============

整个就是一个链式求偏导的过程。 :math:`\frac{{\PartialC}{W}}` 另外就是图论中所有路径最短的问题。

#. 从整个学习上说，就是一个偏导函数量。要解决的一个问题包括两方面：第一个是学习速度问题，第二个是防止震荡。目前用的都是基于导数的优化。
#. 受到cost函数影响是约束问题的松和紧。 cost ， activate MSE(最小二乘），线性的max 函数 cross-entropy, sigmoid函数 log-likelihood, softmax函数。
#. 超参数的估计，目前是过拟合产生一个主要原因。
#. 具体采用哪一种组合呢，就看你采用哪一种解析了，如果想用要概率模型就要用softmax组合。

单层神经网络(前向传播)
========================

假设C类，N个训练样本的结果。

.. math::
 
  E^N=\frac{1}{2}\sum_{n=1}^{N}\sum_{k=1}^C(t_k^n-y_k^n)^2

这里 :math:`t_k^n` 表示第n个样本对应的标签的第k维。 :math:`y_k^n` 表示第n个样本对应的网络输出的第k 个输出。

对于样本n的误差可以表示为：

.. math::
 
   \begin{array}{l}
        E^n=\frac{1}{2}\sum_{k=1}^C(t_k^n-y_k^n)^2=\frac{1}{2}||\textbf{t}^n-\textbf{y}^n||_2^2\\
        \end{array}

那么l层的误差可以表示为：

.. math::
 
   \begin{array}
    E^n=\frac{1}{2}\sum_{k=1}^C(t_k^n-y_k^n)^2=\frac{1}{2}||\textbf{t}^n-\textbf{y}^n||_2^2\\
   \end{array}


对于传统的神经网络需要计算网络关于每一个权值的偏导数。我们用l表示当前层，那么当前层的输出可以表示为：

.. math::
 
   \begin{array}
   x^l=f(u^l)\\
   s.t.\; u^l =W^lx^{l-1}+b^l
   \end{array}


这里  :math:`x^l` 是下一层的输入，这一层的输出。


输出激活函数 :math:`f(.)` 可以有很多中，一般是sigmoid函数或者双曲线正切函数。意思是把他们进行分类。

.. graphviz:: 

   digraph logistic_regress {
   node [shape = box]
   rankdir=LR;
   {node [shape=circle, style=invis]
   1 2 3 4 5
   }
   { node [shape=point,width=0]
   input
   dummy1
   dummy2
   dummy3
   }
   { rank=same;
   posibity cost
   }
   {1 2 3 4 5}-> input-> function -> posibity -> dummy1 -> prediction -> output [weight=8];
   dummy1->dummy2 [weight=8]
   { rank=same;
   dummy2 -> cost  [splines="ortho"]
   cost -> dummy3 ;
   }
   dummy3-> input [weight=8]
   }

后向传导算法（BP算法）
=====================

每一层采用最小均方误差（LMS）模型，采用梯度下降法得到梯度，进而逐层传播到前向网络中去。

.. math::
 
   \frac{\partial E}{\partial b}=\frac{\partial E}{\partial u}\frac{\partial u}{\partial b}=\delta


因为 :math:`\frac{\partial u}{\partial b}=1`, 所以 :math:`\frac{\partial E}{\partial b}=\frac{\partial E}{\partial u}=\delta`, 得到后向反馈的灵敏度： 

.. math::
 
   \delta^l = (W^{l+1})^T\delta^{l+1}\circ f\prime(u^l)

这个模型在无限次迭代中趋于0，也就是没有价值。


输出层的神经元的灵敏度是不一样的：

.. math::
 
   \delta^L= f\prime(u^L)\circ(y^n-t^n)


神经网络就是利用多层信息进行非线性拟合。

权值更新可以表示为：

.. math::
 
   \frac{\partial E}{\partial W^l}=X^{l-1}(\delta^l)^T

.. math::
 
   \Delta W^l=-\eta\frac{\partial E}{\partial W^l}


就是首先求最后一层的误差，逐步扩展到前一层。

实际中对数据训练都是首先前向传导求出实际输出Op,然后和理想输出做对比。得到对比函数，最后使用后向传导调整权值。


随机梯度下降法
==============

mini-batch的原理，是把矩阵变小，这样不需要一次计算整个输入梯度，只用计算部分。 一次一点的来计算。 
​http://neuralnetworksanddeeplearning.com/chap1.html 公式18. 
随机的指的就是那个mini-batch, 正常每次全局的WB来算。 再来计算cost函数，每一次同时算，计算量太大没有办法算，只能每一次算抽取样本来模拟总体cost.
其实就是求平均值的问题，1/3(a+b+c) 与1/2(1/2(a+b)+c) 是不是趋于相当，或者相当于同阶无穷小。
这就是在第5章为什么提到sgd的噪声的原因。
