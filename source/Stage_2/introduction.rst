神经网络基础
************


神经网络起源
============

神经网络起源于对大脑皮层的研究，神纤细胞的轴突末梢（也就是终端）->神经细胞=处理端f(x)。

.. figure:: images/neutral.jpg
    :width: 400px
    :align: center
    :height: 200px
    :alt: alternate text
    :figclass: align-center

    GPU软件结构


大脑特征
======== 

#. 大脑皮层具有 分布式 并行的特点，
#. 最后的输出具有归纳和推广功能。
#. 神经输出按照激活（fire）和不激活两种选择。



.. figure::
.. math::

   \begin{figure}[htp!]
     \centering
     \includegraphics[width=6cm]{NN.jpg}\\
     \caption{神经网络基本结构}
   \end{figure}

虽然神经网络具有复杂的结构，但是只是简单人脑的模拟，成为人工智能（ANN），利用f(x) 模拟人脑思考中的非线性。

大脑模型：
一个是人类有几十万年的进化基础，并且沉淀下来。特别是那些人类看起来很容易，但是计算机却很难的事情。 另外那就是计算量的问题，现在超算的计算量已经超过人脑了。

.. graphviz::

   digraph G {
   rankdir=LR
      
   Memory1->Predict[label="feature1:Color"]
      
   Memory2->Predict [label="feature2:Construct"]
      
   Memory3->Predict [label="feature2:3D information"]
      
   Memory4->Predict [label="feature3:spatial and time seires information"]
      
   Predict->Output
   
   }

*这里的time series 着的是多层之间吗？input*



   

人工智能的未来
===============

大脑是使用记忆来创造的世界，大脑用记忆模型来预测未来，目前的深度学习也体现了这一点。

大脑和计算完全不同，大脑不是靠计算来解决掉问题，而是通过记忆来解决问题。


现在的神经网络分类是单层的分类，层次化的分类，现在还做不到。

参考
=====

#. http://www.cnblogs.com/daniel-D/archive/2013/06/03/3116278.html BP 算法之一种直观的解释
#. `深度学习wiki <http://deeplearning.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_  

#. `神经网络基础<ttp://blog.csdn.net/zouxy09/article/details/9993371>`_
#. `蜜蜂能够认出你 <http://www.huanqiukexue.com/html/newqqkj/newsm/2014/0409/24296.html>`_  蜜蜂在如此脑容量小的情况下能够认出人脸，有什么启发？

#. `L1,L2 正则化<http://freemind.pluskid.org/machine-learning/sparsity-and-some-basics-of-l1-regularization/>`_

#. `SDA <http://deeplearning.net/tutorial/SdA.html#sda>`_
#. `人工智能的未来 <http://blog.csdn.net/zouxy09/article/details/8782018>`_

#. `L1 Norm 稀疏性原理 <http://blog.sina.com.cn/s/blog_49b5f5080100af1v.html>`_
#. `import gzip 模块 压缩文件 <http://docs.python.org/2/library/gzip.html>`_  
#. `拉格朗日乘数 <http://zh.wikipedia.org/wiki/&#37;E6&#37;8B&#37;89&#37;E6&#37;A0&#37;BC&#37;E6&#37;9C&#37;97&#37;E6&#37;97&#37;A5&#37;E4&#37;B9&#37;98&#37;E6&#37;95&#37;B0>`_
#. `LDA-math-MCMC 和 Gibbs Sampling <http://cos.name/2013/01/lda-math-mcmc-and-gibbs-sampling/>`_  

#. `卷积神经网络: <http://blog.csdn.net/zouxy09/article/details/8775360>`_  
#. `LDA-math-MCMC 和 Gibbs Sampling <http://cos.name/2013/01/lda-math-mcmc-and-gibbs-sampling/>`_  gibbs 采样
