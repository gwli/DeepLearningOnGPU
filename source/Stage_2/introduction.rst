************
神经网络基础
************



.. graphviz::

   digraph G {
       rankdir=LR
          
       Memory1->Predict[label="feature1:Color"]
          
       Memory2->Predict [label="feature2:Construct"]
          
       Memory3->Predict [label="feature2:3D information"]
          
       Memory4->Predict [label="feature3:spatial and time seires information"]
          
       Predict->Output
   
   }

基本概念
========

激活函数 
   主要的错误，就像一个哈希函数，把一个任意大小的输入映射到固定区间的大小的数据
   常见的函数有: ReLu ,Sigmoid,Binary,Softplus,SoftMax,Maxout 等以及相关的变型，差不多20多种。

后向传播
   主要是反馈网络的计算，这包含两个LossFunction的计算，以及如何用LossFunction来更新W参数。
   主要有梯度法。
   Cost/Loss 函数 主要来度量预测值与标准值之间测度。单个值对的比较有有时候没有意义。
   例如二值判断，是没有办法来用距离来判断的，怎么办呢，其实是采用集合的正确率，这样一个统计值来计算出
   可以量化的距离值来计算loss.

   - MLE(Maximum LikeHood Estimation). 
   - Cross-Entory
   - logistic
   - Quadratic
   - 0-1 Loss
   - Hinge Loss
   - Expontenial
   - Hellinger Distance
   - Kullback-Leibler Divengence
   - https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
   - https://en.wikipedia.org/wiki/Loss_functions_for_classification

learningRate
   参数的步进的速度，来解决振荡的问题。最好用的应该是AdaGrad,自适应调整。

前向传播
   也就是正常推理计算，基本上者就是矩阵乘

训练的优化
==========

#. Gradient Descent
#. SGD 来解决计算量太大的问题，每一次随机抽取一块来更新
#. Momentum 根据上次的变量来加权更新当前的值 
#. Adagrad 也就是适应调整每一个参数


参数初始化
==========

#. 常量初始化，例如全0初始化，全1的初始化。
#. Small Random Numbers
#. calibrating the Variances 最好可以根据输入的结构来调整 
   
神经元
======

:math:`y=f(\sum{W}*X +b)`

输入层
======

把各种样的输入，映射到神经网络。当然各个输入之间相互独立是最好的。
一般都是 :math:`(X,y)` X是多维的，y是一维的，也可能 y也是多维的。

Batch Nomalization
==================

输入正则化，并且每一次正则化一部分，也可以提前预处理全部的数据。

隐藏层
======
   
输出层
======



Regularization
==============

#. L1 norm
#. L2 norm
#. Eearly Stopping
#. Dropout
#. Sparse regularization on columns
#. Nuclear norm regularization
#. Mean-constrained regularization
#. Cluster mean-constrained regularization
#. Graph-base similarity 



网络结构
========

#. Forward
#. LSTM
#. GAN
#. Auto-Encoders
#. CNN
#. RNN(Recurrant)
#. RNN(Recursive) 


如何开始
========

#. 针对问题，选一个合适的网络结构
#. 看看这个framework的实现有没有bugs 在梯度检查时。
#. 参数初始化
#. 优化
#. 检验模型的有效性
   
   - 如果无效，改变model structure 或者改大网络拓扑
   - overfit, Regularize to prevvent overfitting
      
     * Reduce modle size
     * l1/l2 on weights
     

参考
====

#. https://github.com/dformoso/deeplearning-mindmap
#. http://www.cnblogs.com/daniel-D/archive/2013/06/03/3116278.html BP 算法之一种直观的解释
#. `深度学习wiki <http://deeplearning.stanford.edu/wiki/index.php/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C>`_  

#. `神经网络基础 <ttp://blog.csdn.net/zouxy09/article/details/9993371>`_
#. `蜜蜂能够认出你 <http://www.huanqiukexue.com/html/newqqkj/newsm/2014/0409/24296.html>`_  蜜蜂在如此脑容量小的情况下能够认出人脸，有什么启发？

#. `L1,L2 正则化 <http://freemind.pluskid.org/machine-learning/sparsity-and-some-basics-of-l1-regularization/>`_

#. `SDA <http://deeplearning.net/tutorial/SdA.html#sda>`_
#. `人工智能的未来 <http://blog.csdn.net/zouxy09/article/details/8782018>`_

#. `L1 Norm 稀疏性原理 <http://blog.sina.com.cn/s/blog_49b5f5080100af1v.html>`_
#. `import gzip 模块 压缩文件 <http://docs.python.org/2/library/gzip.html>`_  
#. `拉格朗日乘数 <http://zh.wikipedia.org/wiki/&#37;E6&#37;8B&#37;89&#37;E6&#37;A0&#37;BC&#37;E6&#37;9C&#37;97&#37;E6&#37;97&#37;A5&#37;E4&#37;B9&#37;98&#37;E6&#37;95&#37;B0>`_
#. `LDA-math-MCMC 和 Gibbs Sampling <http://cos.name/2013/01/lda-math-mcmc-and-gibbs-sampling/>`_  

#. `卷积神经网络: <http://blog.csdn.net/zouxy09/article/details/8775360>`_  
#. `LDA-math-MCMC 和 Gibbs Sampling <http://cos.name/2013/01/lda-math-mcmc-and-gibbs-sampling/>`_  gibbs 采样
