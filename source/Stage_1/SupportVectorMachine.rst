Introduction
============


.. graphviz::

   digraph flow {
      SV [label= "support vector"]
   }
   

`SVM python 解释 <http://blog.csdn.net/zouxy09/article/details/17291543>`_   这个里面那个距离公式是错误的。

硬间隔最大化
============

支持向量机首先通过最最大化samples间隔得到如下约束函数：

.. math::
   min \frac {1}{2}||w||^2 \\
   s.t. y_i(w\cdot x_i+b)\geq 1  \qquad \forall x_i


这是一个`二次规划 <Quadratic Programming>`_ 问题，通过转换为对偶优化问题，可以找到更加有效的方法 。

对上式引入拉格朗日乘子，就可以得到以下拉格朗日函数：


.. math:: L(w,b,\alpha)=\frac {1}{2}w^Tw-\sum_{i=1}^{N}a_i y_i(w\cdot x_i-b) 

上式分别对w和b进行求导，分别得到：

.. math::
   w=\sum_{i=1}^{N}\alpha_i y_i x_i

.. math::
   \sum_{i=1}^{N} \alpha_i y_i=0

然后带入拉格朗日函数得到：

.. math::
   max W(\alpha)=\sum_{i=1}^{N}\alpha_i -\frac{1}{2}\sum_{i=1,j=1}^{N}\alpha_i \alpha_j y_i y_j x_i^T x_j \\
   suject to \alpha_i \geq 0, \sum_{i=1}^{N}\alpha_i y_i =0

软间隔最大化
============

松弛向量于软间隔最大化

当一些变量可以超出理想的边界一点的时候，使用软间隔最大化。在目标函数加入一个惩罚项，惩罚outlier， 得到新的最小化问题：

.. math::
   minimize_{w,b,\xi} \frac{1}{2}w^Tw+C\sum_{i=1}{N}\xi_i \\
   subject to y_i(w^Tx_i -b)+\xi_i-1\geq 0, 1\leq i \leq N \\
   \qquad \qquad  \xi \geq N  1\leq i\leq N

同样转换为对偶问题变为：

.. math::
   max \quad W(\alpha)=\sum_{i=1}^{N}\alpha_i -\frac{1}{2}\sum_{i=1,j=1}^{N}\alpha_i \alpha_j y_i y_j x_i^T x_j\\
   subject to \quad C\geq \alpha_i \geq 0, \sum_{i=1}{N}\alpha_i y_i =0

核函数
======


当即使不考虑outlier因素时，还是非线性曲线，就是需要把数据映射到高维，得到线性超平面，根据著名的cover定理：将复杂的模式分类问题非线性地投射到高维空间将比投射到低维空间更可能是线性可分的。但是转化到高维之后，数据是计算量增加。由考虑到我们在优化时要求是内积，和具体数据无关，因此我们值需要数据转换到高维空间的内积就可以了。核函数就是完成这个任务：


.. math:: K(x_i,x_j)=\phi(x_i)^T\phi(x_j) 

其中非常实用的径向基RBF函数：


.. math:: K(x,y)=exp(-||x-y||^2/(2\sigma^2))

此时约束函数转化为：

.. math::
   max W(\alpha)=\sum_{i=1}^{N}\alpha -\frac{1}{2}\sum_{i=1,j=1}^{N}\alpha_i \alpha_j y_i y_j K(x_i, x_j) \\
   suject to \quad C\geq \alpha_i \geq 0, \sum_{i=1}^{N}\alpha_i y_i =0


`SVM python <http://tfinley.net/software/svmpython1/#overview>`_ 


#. `CUDA SVM <http://patternsonascreen.net/cuSVM.html>`_  

