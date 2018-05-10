`深度学习 <DeepLearning>`_ 
===============================

-`神经科学 <Neuroscience>`_ 
================================

`theano <theano>`_ 
===================

-`EM算法 <EMAlgorithm>`_ 
===========================

-`高斯过程 <GaussianProcess>`_ 
===================================

+`稀疏编码 <SparseCoding>`_ 
================================

-+`条件随机场 <ConditionalRandomFields>`_ 
===============================================

+第二集梯度下降法
=========================

我想就是求导，我想和最小二乘本质上都是一样的。只不多最小二乘是参数向量的方法。
迭代函数：

.. math:: {\theta _{i + 1}} = {\theta _i} - \alpha \nabla {J_i}

+`第三集线性回归 <LogisticRegression>`_ 
===============================================


Durbin-Watson （DW）检验 
------------------------

是在回归模型的残差分析中被广泛地应用的一种技术。

假设我希望用最小二乘罚处理观察值，得到线性模型:

.. math::
   y_i =\beta+ \sum_{i=1}^k \beta_i X_{it}+ \xi

其中: :math:`\xi_t` 为第t期的残差。要是这个模型成立，:math:`\xi_t` 应该是独立的，因此:

.. math::
   \xi_t =P \xi_{t-1} +U_t

应该有P=0， 这样就可以检验模型是否正确。
   


http://wenku.baidu.com/view/18a1b2e8b8f67c1cfad6b84e.html


+第四集牛顿法
===================


原理： 并不是所有方程都可以求解，因此使用一阶泰勒展开：:math:`f(x)=f(x_0)+(x-x_0)f^(x_0)` ,  求解 f(x)=0， 得到：:math:`x=x_1=x_0-f(x)/f(x_0)` ， 虽然x1并不能让f(x1)=0, 但是只是更接近f(x)=0, 这样产生了迭代的思想。
牛顿法：
.. math:: {\theta _{i + 1}} = {\theta _i} - \alpha {J_i}/\nabla {J_i}
相比梯度下降法速度更快。
然后又将了GLM，说明最小二乘和 logistic 回归都是GLM 指数分布的特殊情况。是不是我们现在的所有问题都是指数分布？
主要是通过把某个问题划归为某个分布，然后求出期望
.. math:: \mu，从而得到h(x)函数。

梯度法和牛顿法的对比
--------------------

梯度法，是从导数的角度考虑，只是一个向量，而牛顿法是从泰勒分解的角度去考虑，是使用一个曲面去拟合。
那三阶导数又代表什么？
牛顿法相当于考虑梯度的梯度，因此收敛速度更快

梯度法和牛顿法在构造上的不同参考:http://blog.csdn.net/dsbatigol/article/details/12448627

http://www.zhihu.com/question/19723347

http://www.math.colostate.edu/~gerhard/classes/331/lab/newton.html  Newton's Method in Matlab

拟牛顿算法 -BFGS 算法
=====================

.. math::
   min f(x)  x\in R^n


BFGS算法的迭代公式为:math:`x_{k+1}=x_{k}+\alpha_kp_k` ，其中:math:`p_k` 为 :math:`x_k` 的下降方向满足:math:`B_kp_k+\deltaf(x_k)=0`, :math:‘`a_k` 为步长，:math:‘`B_k` 的校正步长为

这里使用局部一次近似，g(x)代替f(x), 开始g(x) 找到，并保证。

这里代替f(x)之后， 就可以使用拟牛顿方法

参考:非凸函数极小问题的BFGS算法结合线性搜索一般模型的局部收敛性

+第五集第六集朴生成学习算法素贝叶斯算法
==========================================================

  * `贝叶斯推断及其互联网应用（一）：定理简介 <http://www.ruanyifeng.com/blog/2011/08/bayesian_inference_part_one.html>`_ 
生成学习算法和logistic 回归的区别在于是否对p(x|y)建模，
也就是需要估计p(y)和p(x|y) ，利用贝叶斯等式求p(y|x) ，对于后来的0概率事件，提出laplace平滑，估计为小概率事件。

就是原来学的全概率时间。 算是后验概率。


-+第七集第八集`SVM <SurportVectorMachine>`_ 
==================================================


SVM的约束函数是通过最小几何间隔实现的，在具体优化的时候使用对偶原理求另一个问题，
在出现outliner使用，加入惩罚项； 使用kernel 把低纬非线性问题转化为高维线性问题。
在优化的时候对于多个位置变量使用sequential minimal optimization 来解决多个变量问题，速度比较快。
#. `Kernel-machines <http://www.kernel-machines.org/>`_ 
#. `jason_svm_tutorial <http://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf>`_ 

而这个与深度学习神经元方式类似，只是神经元两个反向的，对于SVM是要先核函数转化，然后再KKT求最优解。而SVM直接就是线性叠加，然后经过激活函数。

但是从大的层面来看，SVM相当单层的神经网络。
在svm中有不同的非线性函数，但是DL能够用sigmod function 实现对问题的很好刻画。
使用卷积可以模拟任何的函数。

第九集 经验风险最小化
=====================

这一节主要讲的是在我用
.. math:: \hat h 估计h时的误差，从概率上来说可能成功的概率，
比如在每次估计正确和错误，总有一个概率，这个服从伯努利分布，从而误差小于多少，从而推导出Hoeffding不等式，得到误差范围，同样也说明需要的
数据采样长度。

第十集feature selection  and feature selection
===============================================

-+`交叉验证 <crossValidation>`_ 
====================================

`交叉验证 <http://blog.csdn.net/liurong_cn/article/details/10516521>`_ 根据总数据的多少，不同的数据大小,选择不同的验证方法。

模型选择 怎样选择 模型大小，保证不再过拟合和欠拟合情况。

就是怎样使用数据的过程，比如使用70% 训练，30%验证， 对于少量数据使用交叉验证，
充分利用数据，

2、特征选择，选择相似度高的特征。然后使用交叉验证

如果我使用80% 用于训练，20% 用于测试，如果误差很大的说明（多大？） 是数据的异常检测造成的，要么是数据的造成的。

-+第十一集贝叶斯统计正则化
======================================

贝叶斯正则化： 也就是加入先验概率p(\theta),其实想加入惩罚项一样，可以用与减少过拟合问题。

比如原来的约束函数表示为:

.. math::
   J(\theta) = \frac{1}{2m} [\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2 ]

加入贝叶斯正则化后表示为:

.. math::
   J(\theta) = \frac{1}{2m} [\sum_{i=1}^m(h_{\theta}(x^{(i)})-y^{(i)})^2 + \lambda\sum_^{j=1}^n \theta_j^2]

其中 :math:`\lambda` 称为正则化参数，

而如果我们将:math:`\lambda` 设置的很大，则会出现欠拟合.

这里的贝叶斯指的是什么？

http://52opencourse.com/133/coursera%E5%85%AC%E5%BC%80%E8%AF%BE%E7%AC%94%E8%AE%B0-%E6%96%AF%E5%9D%A6%E7%A6%8F%E5%A4%A7%E5%AD%A6%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AC%AC%E4%B8%83%E8%AF%BE-%E6%AD%A3%E5%88%99%E5%8C%96-regularization


lasso算法
---------

http://blog.csdn.net/pi9nc/article/details/12197661

将Lasso应用于回归，可以在参数估计的同时实现变量的选择，较好的解决回归分析中的多重共线性问题，并且能够很好的解释结果。本项目注重对实际案例中的共线性问题利用Lasso的方法剔除变量，改进模型，并将其结果与以往变量选择的方法比较，提出Lasso方法的优势。
将Lasso应用于时间序列。将Lasso思想应用于AR(p)、ARMA(p)等模型，利用Lasso方法对AR(p)、ARMA(p)等模型中的变量选择，并给出具体的算法，随后进行模拟计算，阐明AR(p) 、ARMA(p)等模型的Lasso方法定阶的可行性。


debug方法
training error 和 test error 可以参看`模型调试 <http://blog.sina.com.cn/s/blog_62f4f5d30101dyk9.html>`_ 

是算法选择不合适还是优化问题，通过一些特性，分析

比如当人类飞行员去的更小的J(\theta),说明算法的问题，
 反之人类飞行员更大的J(\theta),说明J(\theta) 设计的不对

-+第十二集  K-Means 和EM算法
===================================··

这主要都是非监督学习，
K-means 首先假设 K中心点， 然后对中心最小化，可以得到局部最优点，相当于 坐标上升法

EM算法和高斯分类算法差不多，只是这里我们不知道y，因此EM算法假设Z，首先假设Z=P(Z|X;\theta)
然后估计\theta.


K-SVD
*****

这是k-svd，直接使用原图像。

K_SVD算法模型
=============

原始k-SVD模型同时估计稀疏字典和稀疏系数，也就是包含两个未知矩阵：

.. math::

   \min\limits_{D,X}{||Y-DX||_F^2}\;  \;\;subject \;to \forall i, x_i =e_k for some k

其中 :math: `e_k` 是单位阵，这个模型中，每个test图像中向量，只有一个1，其它全为0。但是在实际中，这个约束过分苛刻，为了放松约束，只要求任意图像都可以表示为矩阵D的线性表示，并约束 :math:`x_i` 满足稀疏性，得到：

.. math::

   \min\limits_{D,X}{||Y-DX||_F^2}\;  \;\;subject \;to \forall i, ||x_i||_0\leq T_0

估计X
======

这里因为同时有两个位置矩阵，D 和 X.因此无法得到最优解，这里采用近似追踪方法（approximation pursuit method）。首先假设任意初始矩阵D,估计X. 比如使用OMP算法。

更新dictionary D
=================

因为D是一个矩阵，需要求的元素很多，所以这里采用每次只估计一个 :math: `D_j` 原子，其他 :math:`j\neq k`  不变， 这种被称为extreme sparse represention。用梯度迭代更新的方法表示为：

.. math:: D^{(n+1)}=D^{(n)}-\eta\sum_{i=1}^N(D^{(n)}x_i-y_i)x_i^T 

.. math::

   \begin{array}{l}
   ||Y-DX||_F^2=||Y-\sum_{j=1}^Kd_jx_T^j||_F^2\\
   =||(Y-\sum_{j\neq k}d_jx_T^j)-d_kx_T^k||_F^2\\
   =||E_k-d_kx_T^k||_F^2\\
   \end{array}

抛弃零项，减小SVD估计D和W的计算量，得到：

.. math::

   \begin{array}{l}
   E_k^R=E_k\Omega,\\
   x_R^k=x_T^k\Omega
   \end{array}

核心代码实现：
==============

.. code-block:: matlab

    noIt = 200
    [rows,cols]=size(y);
     r=randperm(cols); 
    A=y(:,r(1:codebook_size)); 
    A=A./repmat(sqrt(sum(A.^2,1)),rows,1); 
    D=A;
    X=y;
    K = 50;
    
    for it = 1:noIt
        W=OMP(D,X,4.0/5*rows); 
       % As finding the truly optimal X is impossible, we use an approximation pursuit method. 
        R = X - D*W; 
        %这里包含的应该是误差。  如果是真的，还是应该差不多，如果不是真的，下一次应该包含些。
        %%  这里采用分向量估计的方法.
        for k=1:K
            I = find(W(k,:));
            Ri = R(:,I) + D(:,k)*W(k,I);  % 构成一个虚的向量。
            [U,S,V] = svds(Ri,1,'L');
            %重新更新D和W.
            % U is normalized
            D(:,k) = U;
            W(k,I) = S*V';
            R(:,I) = Ri - D(:,k)*W(k,I);
        end    
    end

假设他对的，然后更新每一个原子D。因为D，W是相互依赖的，在后续更新中，要同时更新。


算法总结:
==========

#.  和k-means比较近似，首先假设随机矩阵D,估计x，然后由x又开始估计D。
#.  首先估计稀疏表示，然后估计字典，在字典估计的时候，采用每次只估计一个Dj，其他不变，交替迭代的方法。

算法存在的可改进空间：
=====================
#. 这里使用的原图像，如果y使用特征，应该能够减少计算量，能获得更好的结果。


算法缺点：
=========

#. 这个算法无法得到全局最优点，只能得到局部最优点。但是实际操作中，这个效果还不错。

#. 这个KSVD用在哪那？我想可以用在识别上。把所有每一副图像都拉成向量，但是这里是基于图像任意排列的情况，只是一些抽象的结果，没有实质性意义，比如无法得到人脑识别到的轮廓信息。这些是算法本身的一些缺点，我想怎样也无法克服。

参考：
====
#. http://en.wikipedia.org/wiki/K-SVD

#. 浅谈K-SVD http://www.cnblogs.com/salan668/p/3555871.html

#. K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation



第十三集 factor analysis 
=========================

这里使用z表示为x的降秩，
.. math:: x=/mu + \Lambda z + \epsilon

然后用`EM算法 <EMAlgorithm>`_ 实现：

E-step 中 估计 z的概率中的p(z|x)  
.. math:: \mu_i,\Sigma_i  

M-step 中估计
.. math::  \mu, \Lambda , \epsilon


第十四集第十五集PCA 算法 ICA算法
================================

把数据从高维压缩到低维

马尔科夫链过程：
===============

马尔科夫链是对复杂世界关系的模型的一种近似简化。

其难点在于模型的估计。查看<<数学之美>>关于马尔科夫链部分。


.. math:: R({s_0},{a_0}) + \gamma R({s_1},{a_1}) + \gamma R({s_2},{a_2}) + ...

Bellman equation:


.. math:: V_\pi (s)=R(s)+\gamma \sum_{s-^\prime\in S} P_{s\pi}(s^\prime)V^\pi (s^\prime)
`后面的还不是 很懂 <http://blog.csdn.net/dark_scope/article/details/8252969>`_ 


多层神经网络MLP
================


MLP模型：
.. math::  f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

第一层activation 一般是sigmod和tanh， 第二层activation一般选择softmax，输出多个离散值。

W和b是分别是两层的参数。
多个参数W是怎么调节的？


粒子群算法
==========

第一个就是粒子本身所找到的最好解，叫
做个体极值点(用i pbest 表示其位置)。另一个极值点是整个种群目前找到的最好
解，称为全局极值点(用i gbest 表示其位置)
PSO算法首先初始化一群随机粒子，在每次迭代中，粒子通过跟踪两个极值来跟踪自己。 假设问题的搜索空间是D维空间上的一个超立方体，用S表示，且 :math:`S=a_1,b_1\times ....\times a_D, b_D` ，在每一次迭代找到这两
个最好解后，粒子i x 根据如下式子来更新自己的速度和位置：

.. math::
   v_{i,j}^{(k+1)} =v_{i,j}^{(k)}+ c_1+r_1(pbest_{i,j}^{(k)}-x_{i,j}^{(k)})+c_2+r_2(pbest_{j}^{(k)}-x_{i,j}^{(k)})

.. math::
   x_{i,j}^{(k+1)}=x_{i,j}^{(k+1)}+v_{i,j}^{(k+1)}

式中， :math:`pbest_i` 为粒子i 经过的最好的位置(自身经验)， gbest 表示的是群体中所
有粒子经过的最好位置(群体经验)； :math:`c_1 、c_2` 为正常数，称为学习因子。 :math:`r_1 、r_2` 为
[0，1]区间的随机数。当粒子在某一维度上飞出问题的搜索范围时，令它在这
一维度上等于搜索区间的上界或者下界。
所有粒子都由被优化函数f (x) 评价其适应度值，对于求解 min f (x) 问题，
个体极值点 :math:`pbest_i` 根据 :math: `f(pbest_i)` 的大小按照下述规则更新：


全局极值点gbest 取所有个体极值点中的最好点进行更新。

.. math::
   pbest_{i}^{\left( k+1 \right)}=\left\{ \begin{align}
  & x_{i}^{\left( k+1 \right)}     \text{if}  \text{f}\left( x_{i}^{\left( k+1 \right)} \right)<f\left( pbest_{i}^{\left( k \right)} \right) \\ 
 & pbest_{i}^{\left( k \right)}    otherwise \\ 
   \end{align} \right.\]


DPSO，对速度和位置使用0，1 构成，全部规整， 这适合于什么场合那？

http://blog.csdn.net/ztf312/article/details/76974571?locationnum=3&fps=1

遗传算法
--------
http://blog.csdn.net/b2b160/article/details/4680853


求解入校二元函数的 最大值

.. math::
   max f(x_1,x_2) = x_1^2+x_2^2
   s.t. x_1 \in {1,2,3,4,5,6,7}
        x_2 \in {1,2,3,4,5,6,7

（1）首先对个体编码

遗传算法的运算对象是表示个体的字符串，所以必须把变量 :math:`x_1, x_2` 编码为符号串，本题中，用无符号二进制正数来表示。

因为 :math:`x_1, x_2` 是0~7之间的整数，所以分别用3位无符号二进制整数来表示，将他们连接在一起所组成的6位无符号二进制数就形成了个体的基因型，表示一个可解行

例如，基因型 X 基因型 X＝101110 所对应的表现型是：x＝[ 5，6 ]。

个体的表现型x和基因型X之间可以通过编码和解码程序转换。 （为什么要用编码方式？）

（2）初始群体的产生

遗传算法是对群体进行的进化操作，需要给其准备一些表示起始搜索点的初始群体数据。

本例中，群体规模的大小取为4，即群体由4个个体组成，每个个体可以通过随机方法产生。

          如：011101，101011，011100，111001


(3) 适应度计算

遗传算法中以个体适应度的大小来评定各个个体的优劣程度，从而决定其遗传机会的大小。

本例中，目标函数总是取非负值，并且是以求函数最大值为优化目标，故可直接李欧阳目标函数作为个体的适应度。

（4）选择运算

选择运算（或称为复制运算） 是把当前群体职工适应度较高的个体按照某种规则或者模型异常到一代群体中去。一般要求适应度较高的个体将有更多的机会遗传到下一代中。

本例中，我们采用与适应度成正比的概率来确定各个个体复制到下一代群体中的数量。

* 先计算出群体中所有个体的适应度的总和 

 * 其次计算出每个个体的相对适应度的大小 fi/\delta fi, 即为每个个体被遗传到下一代群体中的概率，

*  每个概率组成一个区域，全部概率值之和为1；
 
  * 最后在产生一个0~1的随机数，依据该随机数出现在上述哪个改了区域来确定各个个体被选中的次数。（不太理解）

(5) 交叉运算

交叉运算是遗传算法中产生新个体的主要操作过程，它以某种概率相互交某两个个体之间的部分染色体。
本例采用单点交叉的方法，其具体操作过程是：

* 先对群体进行随机配对
* 其次随机设置交叉点位置。
* 最后在相互交换配对染色体之间的部分基因。


(6) 变异运算

变异运算是对个体的某一个或者某一些基因座的的基因值按某以较小的概率进行改变，他也产生新个体的一种操作方法。

本例中，我们采用基本位变异的方法来进行变异运算，其具体操作过程如下:

* 首先确定各个个体的基因变异位置，下表所示为随机产生的变异点位置，其中的数字表示变异点设置在该基因座处；

* 然后依照某一概率将变异点的原有基因值取反。

从上表中可以看出，群体经过一代进化之后，其适应度的最大值、平均值都得
    到了明显的改进。事实上，这里已经找到了最佳个体“111111”。       


差分进化算法DE
--------------

http://blog.csdn.net/hehainan_86/article/details/38685231

差分进化算法DE与遗传算法GA类似，下面是差分进化算法的步骤:

a. 初始化

对于优化问题;

.. math::
   min f(x_1,x_2,...x_D)
   s.t. x_j^L < x_j <x_j^U, j=1,2,..,D
 
其中，D是解空间的维度，:math:`x_j^L, x_j^U` 分别表示第j个分量 x_j 取值范围的上界和下界，DE算法流程如下。

1) 初始化种群。 初始种群 :maht:`\{x_i(0)|x_{j,i}^L<x_{j,i}(0)<x_{j,i}^U, i=1,2,...,NP; j=1,2,...,D\}` 随机产生：
   x_{j,i}(0)= x_{j,i}^L +rand(0,1)x_{j,i}^U-x_{j,i}^L)

  其中 :math:`x_i(0)` 表示第0代的第i 条“染色体”（或个体）,:math:`x_{j,i}(0)` 表示第0代的第i条“染色体”的第j各“基因”，NP表示种群大小，rand(0,1) 表示在（0,1）区间均匀分布的随机数。






进化算法、遗传算法与粒子群算法之间的比较
----------------------------------------

http://blog.csdn.net/hehainan_86/article/details/38398031

遗传算法、粒子群算法，差分进化算法 这些算法都属于进化算法的分支，DE算法性能获得最优，而且算法比较稳定，反复都能收敛到同一个解，PSO算法收敛速度次之，但是算法不稳定, 最终收敛结果容易受参数大小和初始种群的影响，EA算法收敛速度较慢，但在处理噪声方面，EA能够很多的解决而DE算法很难处理这种噪声问题。DE算法收敛速度比较慢


协同学理论
==========

协同学： 指的是在一个处于远离平衡态时，系统内部各个子系统会通过非线性的相互作用产生协同，在系统内部各个作用的作用下，系统会达到临界态，从而使得系统通过自组织方式转变为有序，使旧系统发展成为诸多方面都发生质的新系统，导致新的空间结构，时间结构或功能结构的形成。系统从无序转变为有序的关键是系统内的各个子系统的非线性相互作用。协同学研究焦点是复杂系统宏观特征的质变。这种“由旧结构，到处于不稳定状态，到形成
新结构”的系统演变过程揭示了事物普遍的变化规律。

协同学认为, 系统演化的过程可以用方程来表示，其一般方程为：

.. math::
   q(x,t)=N[q(x,t),\delta,\alpha,x] +F(t)
 
其中，N(.) 是个函数向量，其参数包含状态向量q、微分算子 :maht:`\delta`, 控制参数 :math:`\alpha` 、空间向量x与时间向量t; 最后一项F(t) 表示系统的作用力，该作用力可能来自外界，
也可能是系统内部各子系统之间的相互作用力，在某些情况下该作用力可以忽略，
不影响系统的演变过程，而在某些情况下它对系统的演变起着决定性作用。显然，
式（2-1）是一个关于q 及其不同阶导数的动力学方程或方程组，即微分方程。



时序数据分析
============

ARMA模型
--------

基本原理:

将预测指标随时间推移而形成的数据序列看做是随机序列，这组随机序列所具有的依存关系体现着原始数据在时间的延续性，一方面，影响因素的影响，假定影响因素为 :math:`x_1,...,x_k` ，由回归分析

.. math::
   Y_t =\btea_1x_1+\btea_2x_2+...+\btea_px_p+Z

其中Y是预测对象的观测值，Z为误差，作为预测对象 :math:`Y_t` 收到自身变化的影响，其规律可又下式体现:

.. math::
   Y_t =\beta_0+\btea_1Y_{t-1}+\btea_2Y_{t-2}+...+\btea_pY_{t-p}+Z_t

误差项在不同时期存在依存关系，表示为:

.. math::
   Z_t =\xi_t+\alpha_1\xi_{t-1}+\alpha_2\xi_{t-2}+...+\alpha_q\xi_{t-q}

这个说明噪声也是跟时序有关。

由此，获得ARMA模型表示:

.. math::
   Y_t =\beta_0+\btea_1Y_{t-1}+\btea_2Y_{t-2}+...+\btea_pY_{t-p}+\xi_t+\alpha_1\xi_{t-1}+\alpha_2\xi_{t-2}+...+\alpha_q\xi_{t-q}

如果时间序列Yt 满足：

.. math::
   Y_t =\beta_0+\btea_1Y_{t-1}+\btea_2Y_{t-2}+...+\btea_pY_{t-p}+\xi_t
其中:math:`\xi_t` 是独立同分布的随机变量序列，满足：

.. math::
   Var_{\xi_t}=\delta_{\xi}^2>0

以及 :math:`E(\xi_t)=0`

则称时间序列Yt服从p阶的自回归模型.

移动平均模型(Moving-Average MA)

如果时间徐磊Yt满足:

.. math::
   Y_t =\xi_t+\alpha_1\xi_{t-1}+\alpha_2\xi_{t-2}+...+\alpha_q\xi_{t-q}


则称时间序列Yt服从q阶移动平均模型


移动平均模型平稳条件:任何条件下都平稳。

自回归滑动平均模型（ARMA）

如果时间序列 Yt 满足：

.. math::
   Y_t =\beta_0+\btea_1Y_{t-1}+\btea_2Y_{t-2}+...+\btea_pY_{t-p}+\xi_t+\alpha_1\xi_{t-1}+\alpha_2\xi_{t-2}+...+\alpha_q\xi_{t-q}

则称时间序列Yt为服从(p,q)阶自回归滑动平均混合模型。或者记为 :math:`\phi(B) y_t = \theta(B)\xi_t` .


时间序列混沌性的 嵌入维度和 李雅普诺夫指数，是什么？


怎样提高时间序列的局部波动？


Information Theoretic Learning
------------------------------

Renyi's quadratic entropy 表示为:

.. math::
   H_2(X) = -log(\int_{-\inf}^{\inf} f_X^2(x)dx)

其中 :math:`f_X(\dot)` 是 X的概率密度函数。

信道函数表示为:

.. math::
   \hat H_2(X) =-log(\frac{1}{T^2}\sum_{j=1}^T\sum_{k=1}^T G_{2\delta_{ker}^2}(x_j-x_k))


这样就是一个误差。 这里使用非线性核。因此被称为信息潜在。

A second important ITL measure is correntropy
[19]. It is a similarity metric which “generalizes”
correlation, containing second and higher-order moments
which are expressed by the kernel used in
its definition. If a Gaussian kernel is chosen, the
correntropy between two random variables X and
Y is defined by

.. math::
   c(X,Y) = E\{G_{\delta_{ker}^2}(X-Y)\} \\
         = \int_{-\inf}^{\inf}\int_{-\inf}^{\inf} G_{\delta_{ker}^2}(x-y)f_{X,Y}(x,y)dxdy
         = \int_{-\inf}^{\inf} G_{\delta_{ker}^2}(e)f_E(E=e)de

这个用来衡量信号的互相关

.. math::
   c(E) = \\int_{-\inf}^{\inf} G_{\delta_{ker}^2}(e)f_E(E=e)de

二阶标准最大负熵定义为:

.. math::
   max c(E) s.t. e(n) = d(n) - f(r(n),w)

The principle of MCC is that the maximization
of correntropy increases the similarity between the
desired signal and the adaptive system output, i.e.,
the error values become smaller, which correspond
JOURNAL OF COMMUNICATIONS AND INFORMATION SYSTEMS, VOL. 31, NO. 1, 2016. 4
to higher values of the Gaussian kernel, finally
leading to an error PDF more concentrated at the
origin to higher values of the Gaussian kernel, finally
leading to an error PDF more concentrated at the
origin.

这里使用一个相关熵，相当于



https://books.google.com/books?id=yqhPCwAAQBAJ&dq=%22A%20kernel%20based%20method%20uses%20a%20kernel%20function%20which%20is%20a%20similarity%20function%22&pg=PA165&v=onepage&source=ttb#v=onepage&q=%22A%20kernel%20based%20method%20uses%20a%20kernel%20function%20which%20is%20a%20similarity%20function%22&f=false



DTW 
----

.. math::
   A = \left[ {\begin{array}{{20}{c}}
   {d\left( {{t_1},{r_1}} \right)}&{d\left( {{t_1},{r_2}} \right)}&{...}&{d\left( {{t_1},{r_n}} \right)}\\
   {d\left( {{t_2},{r_1}} \right)}&{d\left( {{t_2},{r_2}} \right)}&{....}&{d\left( {{t_2},{r_n}} \right)}\\
    \vdots & \vdots & \ddots & \vdots \\
   {d\left( {{t_m},{r_1}} \right)}&{d\left( {{t_m},{r_2}} \right)}& \cdots &{d\left( {{t_m},{r_n}} \right)}
   \end{array}} \right]

这里是一个矩阵，说明其

计算所有相似位置总和的位置最小

两个时序时序数据的关联性:

.. math::
   DTW(x,y) = min_W\{ \sum_{k=1}^K W_k\}

实际中使用累积距离矩阵

实际计算距离使用遍历方法找到D


http://blog.csdn.net/zouxy09/article/details/9140207

R/S 分析
--------

用来找到时间序列的周期。

#. 给定时间序列， :math:`X:(x_1,x_2,..,x_n)`，:math:`x_i , i=1,...,n` 

#. 计算序列的均值:math:`x_m` 和方差 :math:`s_n`,

#. 然后计算这个时间序列的新值:

.. math::
   z_i = x_i -x_m

#. 获得序列的累积序列:

.. math::
   y_r = \sum_{i=1}^r z_i.

#. 计算y的范围:

.. math::
   R_n = max(y_1,y_2,...,y_n)- min(y_1,y_2,...,y_n)

#. 除以 方差：

.. math::
   (R/S)_n=frac{R_n}{s_n}

Hurst 指数 是一个H值：

.. math::
   (R/S)_n=cn^H

判断时间序列是随机游走还是有偏的随机游走过程。





https://yq.aliyun.com/articles/50420
自适应共振理论ART模型
---------------------

通过上面的分析发现，首先是


-+trace in machine learning
===========================





-+`交叉验证 <crossValidation>`_ 
====================================

`交叉验证 <http://blog.csdn.net/liurong_cn/article/details/10516521>`_ 根据总数据的多少，不同的数据大小,选择不同的验证方法。


d4c3b2a1

检查自变量和因变量的关系
-------------------------

#. 首先检验哪些变量能够剔除，包括因子分析方法，通过因子分析方法
#. 怎样检验变量对于自变量是否足够？  DW检验 和DH检验


D-W 检验作用

对于期望和预测之间做残差，发现残差是否还有可以提取的信息，如果有继续找相关的自变量，如果没有说明模型能有效。


 * `深度学习资料 <http://blog.sciencenet.cn/blog-830496-679604.html>`_  
#. `机器学习公开课汇总 <http://blog.coursegraph.com/tag/&#37;E6&#37;B7&#37;B1&#37;E5&#37;BA&#37;A6&#37;E5&#37;AD&#37;A6&#37;E4&#37;B9&#37;A0>`_  
#. `深度学习资料 视频 <http://blog.csdn.net/overstack/article/details/8935399>`_  
#. `公开课可下载资源汇总 <http://blog.csdn.net/ddjj131313/article/details/12658675>`_  
#. `机器学习斯坦福资料 <http://cs229.stanford.edu/materials.html>`_  
#. `科学美国人 中文版 <http://www.huanqiukexue.com/>`_  
#. `大数据时代的机器学习热点——国际机器学习大会ICML2013参会感想 <http://www.csdn.net/article/2013-09-05/2816831>`_  
#. `图片库 做测试用 <http://www.image-net.org/challenges/LSVRC/2012/browse-synsets>`_  
#. `深度学习资料1 <http://www.cnblogs.com/549294286/archive/2013/07/08.html>`_  
#. `图片库 测试用 <http://pascallin.ecs.soton.ac.uk/challenges/VOC/>`_  
#. `Deep learning  进阶分析 <http://blog.csdn.net/zouxy09/article/details/8782018>`_  
#. `深度学习lecture <https://class.coursera.org/neuralnets-2012-001/lecture>`_  
#. `深度学习向导 <http://deeplearning.net/tutorial/>`_  
#. `deep-learning-made-easy  fastml <http://fastml.com/deep-learning-made-easy/>`_  
#. `deeplearning software list <http://deeplearning.net/software&#95;links/>`_  
#. `machine-learning-in-ocaml-or-haskell <http://stackoverflow.com/questions/2268885/machine-learning-in-ocaml-or-haskell>`_  
#. `人工智能的新曙光 <http://songshuhui.net/archives/56314>`_  人工智能概率论介绍




大脑思考 数学模型



pydot 有下载的必要吗？




http://songshuhui.net/archives/76501>`_   正态分布的前世今生 <非常精彩的分析了正态分布，以及二项分布，以及极大似然=平均值，以及最小二乘的关系。



Baum-Welch Algorithm
维特比算法 有空看下




分析下航空订票的原理：`航空公司如何给飞机票定价？ <http://www.guokr.com/article/91252/>`_  



logit 模型还不是很清楚



在svn中整体修改，多个不同位置的不同文件件



probabilistic Latent Semantic Analysis  什么



交替方法一定收敛吗?



+`高斯过程 <GaussianProcess>`_ 
===================================

+`条件随机场 <ConditionalRandomFields>`_ 
==============================================

+`决策树 <DecisionTree>`_ 
=============================

+`张量分解 <TensorDecomposition>`_ 
=======================================

分类器
=========

KNN( Nearest Neighbor)
----------------------

该方法的思路是： 假设一个样本在特征空间的K个最相似的样本中的大多数属于某一类别，

http://www.cnblogs.com/blfshiye/p/4573426.html

串匹配
=========

退火算法
============


退火算法设计的本初：改进非凸问题中`爬山算法 <http://www.cnblogs.com/heaad/archive/2010/12/20/1911614.html>`_ 中的局部最小点。


方法：模拟自然中退火的方法，通过随机交换，求出任意两个点之间的传输距离。我想和便利算法有什么区别？ 还有什么缺点？并用`内循环终止准则，或称Metropolis抽样稳定准则（Metropolis 抽 样准则） <http://www.google.com.hk/url?sa=t&rct=j&q=+Metropolis+%E6%8A%BD+%E6%A0%B7%E5%87%86%E5%88%99&source=web&cd=4&ved=0CD8QFjAD&url=http%3a%2f%2fjingpin%2eszu%2eedu%2ecn%2fyunchouxue%2fziliao%2fkejian%2f%25E7%25AC%25AC10%25E7%25AB%25A0-%25E6%2599%25BA%25E8%2583%25BD%25E4%25BC%2598%25E5%258C%2596%2eppt&ei=JyHEUYOcMY_ZkgWPq4GYAw&usg=AFQjCNG1kEOdSgfjKesiOxiDiT8E4u4ZBQ>`_ 蒙特卡洛实验来是系统达到平稳。

解决问题： `货郎担问题 <http://www.vckbase.com/index.php/wv/1196>`_ ，每次只能选择一个地方，也就是交换一个地方。遍历算法在这个问题中式难以解决的。
`matlab 代码 <http://wenku.it168.com/d_000326627.shtml>`_ 。


优缺点：
#. 优点：计算过程简单，通用，鲁棒性强，适用于并行处理，可用于求解复杂的非线性优化问题。
#.  缺点：运算量较大，下降速度和收敛稳定性的矛盾，下降速度过大，可能出现震荡。下降速度过慢，运算量大。

`退火算法的改进 <http://baike.baidu.com.cn/view/335371.htm>`_  比如采用变化的熟练速度和刚开始收敛速度比较快，基本稳定后采用小收敛速度。

  * 思考：和图论中*最短路径算法* 剪枝算法* 最小逆序数*。


 
K-means 算法=expection maximum （EM）期望最大
=====================================================

就是一个分类器的设计
#. `深入浅出K-Means算法  <http://www.csdn.net/article/2012-07-03/2807073-k-means>`_  K means 中K的选择，初始值的选择，里面都有。

Clustering by fast search and find of density peaks
---------------------------------------------------

这是2014年 science 文章

这里有两个基本的标准


.. code-block::
  \pho_i = \sum_j \xi(d_{ij}-d_c)

这里的pho 是 局部密度，满足:

.. code-block::
   \xi(x) =\begin{aligned}
   1, x<0  \\
   0, otherwise \\
   \end{aligned}

dc 是截断距离，pho基本上等于与点i的距离小于dc的点的个数。算法只对不同点的:math:`\pho_i` 的相对大小敏感，这意味着对于大数据集，分解结果对于 dc的选择具有良好的的鲁棒性。

数据点i 的 :math:`\delta_i` 是点到任何比密度大的点的距离的最小值:

.. code-block::
   \delta_i =\min_{j: \pho_j>\pho_i} (d_{ij})

对于密度大的点，我们可以得到 :math:`\delta_i =\max_j(d_{ij})` .

http://blog.csdn.net/jdplus/article/details/40351541


作者主页
http://people.sissa.it/~laio/Research/Res_clustering.php


局部二次方法
------------

假设:math:`T(\lambda)` 是解析矩阵函数，:math:`\lambda_k` 是非线性特征值问题的一个近似特征值，由Taylor公式可得:

.. math::
   T(\lambda_k + h) = T(\lambda_k ) +h T'(\lambda_k ) +\frac{1}{2}h^2T''(\lambda_k )  +\frac{1}{6}h^3 R(\lambda_k,h)

其中  :math:`R_3(\lambda_k,h)`  是一个矩阵，是范数有界的。相应于主次线性近似，我们使用如下二次特征问题:

.. math::
   (T(\lambda_k ) +h T'(\lambda_k ) +\frac{1}{2}h^2T''(\lambda_k ))x=0

近似非线性特征问题（1.1）如果h是二次特征问题（2.2）模最小的特征值，则 :math:`\lmabda_{k+1} = \lmabda_{k} +h` 可作为非线性特征问题的新近似特征值。


mackey-Glass
------------

是时序混沌系统的典型系统。


参考
http://wenku.baidu.com/link?url=2NiPN6KK7Bwv4KPQNkErkPthO0saGQTwB-uGNWkLgx-e4pcQvM1q5fzKs3L8Fw3GEBY7cLFQQ6jNcA0MxSt-Ze2Xw98YD3AkPfI2K1lzvHCk

Lorenz 
------

.. math::
   \frac{dx}{dt} = a(-x+y), frac{dy}{dt}= bx-y -xz, dz/dt=xy-cz

其中参数选取 a=10, b=28, c =8/3 时，系统具有混沌特性。

embeding dimension 是什么
-------------------------------------

是时延，  tau是延迟参数


.. math::
   [x(t),x(t+\tau),...,x(t+m\tau)]

是用embeding 来估计的 他的相位空间，


See also
========

斜反射布朗运动
*************

布朗运动， 一维延拓


布朗运动对应是什么？ 可以用在哪？



粒子群算法，反射回来。



延拓的租用？


.. math::
   \int_0^\inf 1_D(X_s)dL_s =0


.. math::
   v(x)=n(x)+tahn\theta(x)t(x)



.. math::
   dX_t=dB_t + V(x)dL


什么是调和函数



斜反射布朗运动：可以用在哪？

可以用在PSO中
。。


未来的发展都是基于网络的分散式样的发展，这样的


在听讲座的时候，听不懂怎么办？

CDN是什么？


umind 现在还不成熟，应该怎么办？

自己做实验，测试。


现在因为高纬度的问题，目前还没有解决。


勒贝格测度，大数定理


布朗运动和概率密度函数


异常检测
========

异常时在数据集中与众不同的数据，使人怀疑这些数据并非随机偏差， 而产生完全不同的机制，聚类算法对于异常的定义：异常既不属于聚类也不属于噪声。主要方法是包括：基于统计的方法，基于距离的方法，和基于偏差的方法。基于密度的方法。

基于统计的方法

Knorr 提出基于距离的异常检测方法，数据集S中一个对象与O的距离大于距离D，简单地说，基于距离的异常点就是那些没有足够多的邻居的点的对象，采用DB(p,D)-outlier 可以表示所有的基于统计的异常。

基于偏差的方法

提出一种“序列异常”的概念，算法介绍给定n个对象的集合S,建立一个子集序列{S1,S2,...,Sm} 这里




高斯马尔科夫模型
----------------


速度可以描述为如下:

.. math::
   v_n = cv_{n-1}= v_0 c^n

所以高斯马尔科夫测量的总开销为：

.. math::
   c_{MALM} = Kv_n =K v_0 c^n

这里的K是什么？

在GMLM中，高斯马尔科夫模型中节点速度描述如下:

.. math::
   v_n = \alphav_{n-1} +(1-\alpha)\mu +\sigma\sqrt{1-\alpha^2}w_{n-1}

其中 mu 和 sigma 是 v_n 的期望和方差，


就算我现在有很多想法，但是还是其实都没有实现，我觉得还是



用来描述速度的一个变化
参考: 一种高斯-马尔科夫模型下的车辆位置管理策略


参考:
http://jiangshuxia.9.blog.163.com/blog/static/3487586020083662621887/

#. `基于生物进化的遗传算法概述 <http://www.zjubiolab.zju.edu.cn/lesson/userfiles/file/&#37;E4&#37;BA&#37;A4&#37;E5&#37;8F&#37;89&#37;E5&#37;AD&#37;A6&#37;E4&#37;B9&#37;A0&#95;&#37;E7&#37;94&#37;B5&#37;E6&#37;B0&#37;94&#37;E5&#37;AD&#37;A6&#37;E9&#37;99&#37;A2&#37;E6&#37;9E&#37;97&#37;E5&#37;B3&#37;B0.pdf>`_  这里面有遗传算法在各领域的应用
#. ` De Jong&#39;s function 1 <http://www.geatbx.com/docu/fcnindex-01.html>`_  这个函数有多个局部最优点，一般用来作为算法局部最小点的例子
#. `退火算法 <http://baike.baidu.com.cn/view/335371.htm>`_  
#. ` <http://blog.sciencenet.cn/blog-1225851-761882.html>`_  
#. `大数据时代的机器学习热点——国际机器学习大会ICML2013参会感想 <http://blog.sciencenet.cn/blog-1225851-761882.html>`_  看不太懂，有空看看
#. `最大似然参数估计 <http://nbviewer.ipython.org/github/unpingco/Python-for-Signal-Processing/blob/master/Maximum&#95;likelihood.ipynb>`_  

Thinking
========




*模式* 描绘子的组合。

-- Main.GangweiLi - 24 Sep 2013


基于决策理论，基于神经网络，基于机器学习。

-- Main.GangweiLi - 24 Sep 2013


`西安电子科技大学陈渤 <http://web.xidian.edu.cn/bchen/index.html>`_  我电子所的，主要做大数据分析和机器学习 和图像识别。

-- Main.GegeZhang - 04 Jun 2014


`高新波 <http://web.xidian.edu.cn/xbgao/>`_  他做的方向比较新，可以看一下。

-- Main.GegeZhang - 04 Jun 2014

机器学习算法汇总：人工神经网络、深度学习及其它，http://www.csdn.net/article/2014-06-27/2820429
=================================================================================================

学习方式，有监督，半监督，无监督。强化学习。 


元胞机
-----

元胞网络将更擅长于解决现实世界中具有不稳定、非线性、不确定性、非结构化以及病
态结构的复杂决策问题


L0 范数、L1范数和ridge 回归
----------------------------


ridge 能够实现最小化，但是不能产生稀疏值，也就是不区分稀疏值，但是实际中通常需要稀疏值，因此使用L1范数（Lasso 回归），但是L1相关的变量不起作用，因此有人提出使用L1范数和L2范数结合，即所谓的弹性网，但是弹性网还是有偏估计，因此使用SCAD和L2 范数，L2 起到合并组效应，SCAD满足稀疏估计。

递进过程也就是：
ridge-》L1-> 弹性网-》弹性 SCAD

因此还需要使用SCAD正则化，


李雅普诺夫指数
---------------

http://baike.baidu.com/link?url=vFl3c6SKfWdon5BCGxiOFsgIqiz8WyIlj5gKQdM1Gi9gYmjxHpN9QKK7hcEgvGKkkd1wvbK0v5OYziea0v6SWyPsHVq90zT9Cc3bcSzLBOQnr9y0mG4ohD0Malmtc-mzX36X9AL3oTQ-yC-RJX3ojoFLe66fKgnJne8cKMHrdkq

考虑两个系统：

.. math::
   x_{n+1}= f(x_n),  y_{n+1}=f(y_n)

设其初始值微小误差为：

.. math:: 
   |x_0-y_0|

经过一次迭代有：

.. math::
   |x_1-y_1|= |f(x_0)-f(y_0)|= \frac{|f(x_0)-f(y_0))|}{|x_0-y_0|}\aprox \frac{df}{dx}|x_0-y_0|

第二次迭代得到：

.. math::
   |x_n-y_n|=  |\sum_{n=0}^{n=1}\frac{df}{dx}||x_0-y_0|
   

可见两个系统对初始扰动的敏感度由导数|df/dx|在x0处的值有关，它与初始值有关。


说明两个原来差值的很小，后来差别比较大，说明初始值对他有影响，也就是混沌现象。

如果两个系统初始存在细小的差异，随着时间产生分歧，分离程度用李雅普诺夫指数来度量


Volterra series
----------------

https://en.wikipedia.org/wiki/Volterra_series


假设非线性离散动力系统的输入为 :math:`X(n)=[x(n),x(n-\tau),...,x(n-(m-1)\tau)]` , 输出 :math:`y(n)= x(n+1)`, 则非线性系统的 Volterra 展开式表示为：

.. math::
   x(n+1) =h_0 +\sum_{k=1}^P y_k(n)

其中 

.. math::
   y_n(n) = \sum_{i_1,...,i_k=0}^{m-1} h_k(i_1,...,i_k)   \prod_{j=1}^k x(n-i_j\tau)


这里的 :math:`h_k(i_1,...,i_k)`  称为 k阶段 Volterra 核，p 为 Volterra 滤波器阶数。在实际应用中，必须采用有限次求和的形式。以二阶截断m次求和为例，用于混沌时间序列预测的滤波器为：


.. math::
   x(n+1) =h_0 +\sum_{i_1=0}^{m-1} h(i_1)x(n-i_1\tau) + \sum_{i_1,i_2=0}^{m-1} h(i_1,i_2)x(n-i_1\tau)x(n-i_2\tau)


这说明这个信号是有时延和历史核组成，这样有什么意义。

可用于非线性时间序列预测。但是这个只是一个模型，应该怎样求解，才是关键。

摘自：

混沌时间序列的Volterra级数多步预测研究


拓扑学
******

拓扑学解决什么问题？

结合基于

七桥问题: 解决路径问题

四色定理

庞加莱猜想

相关学科：

图论和几何拓扑

笛氏积

解决空间多维扩展

笛氏积：包含空间中的方向


拓扑定义：
--------

拓扑就是在拉伸、旋转中相邻的关系还是不变的


官方： 研究图形在拓扑变换下不变性质的学科。


拓扑学经常被描述成 “橡皮泥的几何”，就是说它研究物体在连续变形下不变的性质。

盘带哪些物体是本质同一形态的？

莫比乌斯带

无法区分一个面无法区分它是正面还是反面？ 这个和拓扑有什么关系？

从扭结问题，谈到临近点的关系。

它是一种几何，但是，

扭结理论：

说明物体实际上以某种形态扭结在一起, 这和弦论有什么关系？

http://blog.sciencenet.cn/home.php?mod=space&uid=522561&do=blog&id=417414

这里的爬虫的几何很难理解。

拓扑和积分有什么关系？


http://songshuhui.net/archives/1633


拓扑空间
--------
开集是什么？

同一集合上的不同度量可能导致相同的开集。

这说明是相同的


“同调群” 与 “基本群” 是什么？

“微分拓扑学 什么？


（同调群，同伦群

拓扑学和图论什么关系


拓扑学研究的是 在拓扑变换下的不变理论。

图论主要研究顶点， 边

紧空间
------

连通性
------

基本群和覆盖空间
================


单纯同调性
----------

探索复杂性
++++++++++

A small-world network is a type of mathematical graph in which most nodes are not neighbors of one another, but the neighbors of any given node are likely to be neighbors of each other and most nodes can be reached from every other node by a small number of hops or steps

https://en.wikipedia.org/wiki/Small-world_network

小世界网络： 指的是和临近的关系不近，但是可以通过几步跳称为两一个节点的邻居。

https://en.wikipedia.org/wiki/Scale-free_network

A scale-free network is a network whose degree distribution follows a power law, at least asymptotically. That is, the fraction P(k) of nodes in the network having k connections to other nodes goes for large values of k as:

.. math::
   p(k) \prox k^{-\gamma}


节点度服从度指数介于（2,3）的幂律分布

混沌：初始值影响很严重，分形


耗散理论 和非线性关系
---------------------

http://baike.baidu.com/link?url=d9hUNIax8Z1NnBOxHlHENTxO5lztOndRiJ7rLedYIkwBdbU2mJW9B63kkKeAxi1M5hIt58_CJmO1V56d3VsOWcEdtafXb_mMTXIQWwx620B-Vj4d2cPNi9RT70v1k7Wn

耗散理论呈现出非线性

耗散结构的研究揭示了一种重要的自然现象，并对复杂系统的研究提出了新的方向。在数学上描述复杂系统的方程通常是非线性的，一般包括分岔现象。

正的Lyapunov指数表明该度量邻近的点之间发散的快慢，负的Lyapunov指数表明系统在受到扰动后多长时间能回复到初始状态。如果系统中最大的Lyapunov指数为正数，则表明系统是混沌的，

基于混沌理论的短时交通流量预测

这是不是说明贫苦家人的孩子不能嫁。


时间序列预处理
===============


不平稳数据分析

BOX-COX 变换



差分
----

对于有趋势性的数据，采用差分

.. math::
    \delta X_t  =  X_t -X_{t-1}


季节差分
--------

对于周期性时间序列采用季节差分

.. math::
   Y_t= Z_t- Z_{t-12}

对数变换和


https://wenku.baidu.com/view/4d375381ec3a87c24028c43a.html



多变量时间序列预处理
--------------------


The purpose of presample data is to provide initial values for lagged variables. When trying to fit a model to the estimation data, you need to access earlier times. For example, if the maximum lag in a model is 4, and if the earliest time in the estimation data is 50, then the model needs to access data at time 46 when fitting the observations at time 50. By default, estimate removes the required amount of observations from the response data to use as presample data. This reduces the effective sample size.
z
怎样设置 重采样时间。


偏最小二乘
----------

解决多变量预测中x变量存在的共线性问题。


https://wenku.baidu.com/view/fc2291a9d1f34693daef3eca.html


做下变量分析，包含更多的。

数电、模电、包括更多的信息。

技术的方面，我想还是。
方便更多的信息。方便的处理。
·
招学生，现在学生比较多的爱上课。

CSDN 中 的学生偏微分。

做大数据的

tensorflow

http://www.tensorfly.cn/

基于检测的分类
---------------

利用分类对目标进行检测，

机器学习的基石是 分类和预测。


one-hot 编码是什么
----------------------


如果所有的数据概率*10 要么概率非常接近1，要么为0.


softmax的作用是将数据转换成概率，然后再通过吧one-hot编码，把数据转换成分类。

并且通过交叉熵来看起距离，也就是分类



数据中误差被放大，因此要对数据做均值为0，方差为1的归一化。


kaggle 数据竞赛

随机梯度下降法对于大矩阵计算比较有效，用在深度学习中。

随机梯度 是梯度随机变化一下

权值使用：使用方差较小的权值，防止参数变化很大。

学习率的指数衰减



不平衡数据集分类
=================

欠采样方法
----------

从多数类中随机抽取样本而减少多数类样本的数量，使数据达到平衡。

处理非平衡数据常用的方法就是设置损失函数的权重，是的少数类别错误的损失大于多数类别错误的损失。




：

经济学：

cost-sensitive learning 


从贝叶斯敏感的角度来说：


从贝叶斯后验概率角度出发，以实现损失最小为目标，优化目标如下

..math::
    H(x)= arg min(\sum_{j\in {-,+}}P(j|x)C(i,j))

edited nearest neighbor （ENN）

如果他的大部分k临近样本都跟他自身的类别不一样，我们就将他剔除。（什么意思）

http://blog.csdn.net/a358463121/article/details/52304670

tomek link removal

如果有两耳不同类别的样本，他们的临近都是对方，就称为A，B的 Tomek link。

tomek link removal 就是将 组成tomek link的样本删除掉


代价敏感 ：http://www.jianshu.com/p/3e8b9f2764c

我真的到那种挥斥方遒的阶段了吗？



AdaCost算法
-----------

比如学习的分类是怎样的，

参考：

http://www.jianshu.com/p/3e8b9f2764c8

http://blog.csdn.net/a358463121/article/details/52304670

怎样添加规则，

比如添加关键字数仿真的长度，比如学习识别的长度，经济搜索的长度，包括时长，

利用生成网络，生成数据，

分为创业型的，科学家

focus 好奇心，对其进行分类，

还有一个应该是五个点。

过采样方法：

一个简单的方法，就是通过有放回的抽样，不断地从少数类抽取样本，不过这样很容易导致过拟合。

http://blog.csdn.net/heyongluoyao8/article/details/49408131


Smote 用来平衡算法
------------------

是属于过采样的方法


smote 是 合成新的少数样本，策略是每个少数类样本a，从它的最近邻随机选一个样本b。

考虑少数类的一个样板i，其特征向量为$x_i, i\in {1,...,T}$;

1、从少数类的全部T个样本找到样本 $x_i$ 的k个近邻 x_i(near),near \in{1...,k}
2、然后从k个近邻中随机选择一个样本x_i(nn),生成一个0-1之间的随机数：

.. math::

    x_{i1}= x_i+\sing_i\dot(x_{i(nn)}-x_i)


将步骤2 重复N次，从而合成N个新样本: x_{inew}, new\in 1,...,N.

对全部的T个少数类样本进行上述操作，即可为少数类合成NT个新样本。

伯努利方程


宁可错杀一千，不可放过一个。是个概率，可以通过实现。

竞争模型

今天这个模型：竞争合作模型。

增强学习
========

优达学城 增强学习
-----------------


学习生物的一种变化的状态，比如加入更多的思考。基于上一步的操作。时间变化。

比马尔科夫链考虑更多的问题。

马尔科夫链 加入更多：

时间序列分析：应该也可以用。

标注:数据是否需要标注，进行分

增强学习 为什么要使用每个状态。

大家都在做机器学习，我有什么特殊的？

为什么是博弈的状态？

这里我们成为物理学

markov 是什么？

下一个状态总是有概率的，转换为概率。

markov 可以记住过去吗

物理规则是不变的。

MDP ，markov decision propress

定义问题：


增强学习的概念

博弈论


http://open.163.com/movie/2016/7/A/I/MBR31E8RH_MBR31KRAI.html


最大似然：

怎样定义凸问题？


无穷范数在凸优化中的作用？



对偶问题怎么理解？

C 语言 MATLAB 

fuzzy nerual network （  ）  模糊神经网络
==========================================


拆成分式

.. math::
  A=\mu_1 x_1 +\mu_2 x_2+...., \mu_n x_n

:math:  是 grade of membership  of x_i 


x_i-> [0,1] 是一个membership

.. math::

    A=\sum+i\mu_i


贝叶斯信息准则（BIC）
======================

.. math::

   AIC(p)= log\frac{||Z-X_pW_p||_2^2}{n}+\frac{2p }{n}


.. math::

   BIC(p)= log\frac{||Z-X_pW_p||_2^2}{n}+p\frac{logn }{n}

就是看这两个信息的达到最大

oracle property 就是数据在无限大，情况下，满足真是估计。


权值最小二乘（weighted least squares）WLS
-------------------------------------

.. math::
   WLS function =\sum{w_i}(y_\beta_0-\beta_1x_{1i},....,\beta_px_{pi})^2

对于wi就是对于每一个点的权值。



对不同的连接矩阵，


Lyapunov exponents 计算 李雅普诺夫 指数

在时间中的两个点 ||\delta x_i(0)|| and |||\delta x_i(t)|| 验证i方向的距离：



通过平均增长速度

.. math::
   \frac{||\delta x_i(t)||}{|||\delta x_i(0)||}= 2_i^\lambda t

or 
.. math::
  \lambda_i = \frac{1}{t}log_2{\frac{||\delta x_i(t)||}{|||\delta x_i(0)||}}


||\delta x_i(0)|| and |||\delta x_i(t)||

Lorenz中的 三根线都是混沌的？

这里的lambda 怎样找到？

大矩阵分解方法

https://books.google.com/books?id=joO8BAAAQBAJ&dq="The%20fast%20matrix%20multiplication%20by%20Strassen%20187%201969%20is%20based%20on%20an%20algorithm%20for%20multiplying%202%202%20block%20matrices"&pg=PA137&v=onepage&source=ttb#v=onepage&q="The%20fast%20matrix%20multiplication%20by%20Strassen%20187%201969%20is%20based%20on%20an%20algorithm%20for%20multiplying%202%202%20block%20matrices"&f=false



这里通过分解成两个矩阵相乘，得到一个小矩阵，

strassen' algorithm, zhge de 


计算放量从  2n^3 到 4n^2.8


但是这个方法存在数值不稳定。


还有MATLAB程序




https://books.google.com/books?id=BIJZTGjTxBgC&dq="In%20order%20to%20avoid%20B%20X%200%200%20in%20computation%20it%20is%20suggested%20to%20use%20the%20following%20formula%20to%20compute%20the%20inverse%20matrix"&pg=PA41&v=onepage&source=ttb#v=onepage&q&f=false


通过化简，

距离函数，是误差函数，这里首先把Y进行特征值和特征向量分解。

这里在D

https://books.google.com/books?id=6ZBOA-iDviQC&dq="To%20calculate%20the%20inverse%20of%20a%20square%20matrix%20B%20the%20adjugate%20or%20adjoint%20matrix%20of%20B%20is%20first%20defined"&pg=PA83&v=onepage&source=ttb#v=onepage&q="To%20calculate%20the%20inverse%20of%20a%20square%20matrix%20B%20the%20adjugate%20or%20adjoint%20matrix%20of%20B%20is%20first%20defined"&f=false


这也是计算逆矩阵的方法。这个方法还比较小。 这个计算逆矩阵的方法，叫做计算量。

https://books.google.com/books?id=jtX5BwAAQBAJ&dq=%22In%20this%20solving%20procedure%20the%20inverse%20problem%20is%20recast%20as%20an%20optimization%20problem%20which%20is%20solved%20by%20using%20a%20conjugate%20gradient%20strategy%20for%20updating%20at%20each%20iteration%20the%20contrast%20source%20and%20the%20contrast%20function%20in%20an%20alternate%20way%22&pg=PA257&v=onepage&source=ttb#v=onepage&q=%22In%20this%20solving%20procedure%20the%20inverse%20problem%20is%20recast%20as%20an%20optimization%20problem%20which%20is%20solved%20by%20using%20a%20conjugate%20gradient%20strategy%20for%20updating%20at%20each%20iteration%20the%20contrast%20source%20and%20the%20contrast%20function%20in%20an%20alternate%20way%22&f=false

这里使用共轭梯度法进行求解。


https://books.google.com/books?id=9kB5jE2IjS4C&dq="Unweighted%20least%20squares%20ULS%20is%20a%20factor%20extraction%20method%20that%20minimizes%20the%20sum%20of%20the%20squared%20differences%20between%20the%20observed%20and%20reproduced%20correlation%20matrices%20ignoring%20the%20values%20on%20the%20diagonals"&pg=PA113&v=onepage&source=ttb#v=onepage&q="Unweighted%20least%20squares%20ULS%20is%20a%20factor%20extraction%20method%20that%20minimizes%20the%20sum%20of%20the%20squared%20differences%20between%20the%20observed%20and%20reproduced%20correlation%20matrices%20ignoring%20the%20values%20on%20the%20diagonals"&f=false


从目前的角度来看，很多人在时间序列，并且方法也比较成熟了，应该研究 多

智能交通  

矩阵求逆矩阵

为什么要使用共轭梯度法。


这里使用梯度投影法，解决非线性问题。。

https://books.google.com/books?id=_VwICgAAQBAJ&dq="The%20gradient%20projection%20method%20has%20been%20successfully%20implemented%20and%20has%20been%20found%20to%20be%20effective%20in%20solving%20general%20nonlinear%20programming%20problems"&pg=PA370&v=onepage&source=ttb#v=onepage&q="The%20gradient%20projection%20method%20has%20been%20successfully%20implemented%20and%20has%20been%20found%20to%20be%20effective%20in%20solving%20general%20nonlinear%20programming%20problems"&f=false


https://books.google.com/books?id=WyvVDAAAQBAJ&dq=%22The%20matrix%20inversion%20operation%20can%20be%20divided%20into%20three%20approaches%20explicit%20computation%20implicit%20computation%20and%20polynomial%20expansion%22&pg=PT399&v=onepage&source=ttb#v=onepage&q=%22The%20matrix%20inversion%20operation%20can%20be%20divided%20into%20three%20approaches%20explicit%20computation%20implicit%20computation%20and%20polynomial%20expansion%22&f=false
对于矩阵逆计算，主要有三种方法， 线性方法， 比如矩阵分解。    还有一种矩阵乘法。





https://books.google.com/books?id=dfa8BAAAQBAJ&dq="Several%20numerical%20methods%20generally%20available%20in%20numerical%20analysis%20libraries%20are%20available%20for%20such%20computation%20such%20as%20matrix%20inversion%20and%20relaxation%20techniques"&pg=PA14&v=onepage&source=ttb#v=onepage&q="Several%20numerical%20methods%20generally%20available%20in%20numerical%20analysis%20libraries%20are%20available%20for%20such%20computation%20such%20as%20matrix%20inversion%20and%20relaxation%20techniques"&f=false

这里的 relax 技术是什么？


https://books.google.com/books?id=wcS_DAAAQBAJ&pg=PA310&lpg=PA310&dq=An+iterative+method+for+matrix+inversion+such+as+the+conjugate+gradient+CG+algorithm+is+therefore+preferred&source=bl&ots=Hk7UyyEO5y&sig=QbROuRxJHUVYP62sBv2_kpiv1RM&hl=zh-CN&sa=X&ved=0ahUKEwi1neCmiPHaAhVQ9GMKHZGvAhMQ6AEIJzAA#v=onepage&q=An%20iterative%20method%20for%20matrix%20inversion%20such%20as%20the%20conjugate%20gradient%20CG%20algorithm%20is%20therefore%20preferred&f=false

这里讲的共轭，也许就是 F^H 和

共轭找到算法，


共轭梯度算法是收敛速度慢




The conjugate gradient method takes a combination of the previous direction and the new direction to approach the optimum more directly.

使用 以前的方向和 新的方向

https://books.google.com/books?id=o6aKDQAAQBAJ&dq="A%20model’s%20prediction%20is%20in%20the%20form%20of%20a%20predictive%20distribution%20which%20is%20an%20estimate%20of%20the%20underlying%20distribution%20that%20resulted%20in%20the%20observed%20data"&pg=PT348&v=onepage&source=ttb#v=onepage&q="A%20model’s%20prediction%20is%20in%20the%20form%20of%20a%20predictive%20distribution%20which%20is%20an%20estimate%20of%20the%20underlying%20distribution%20that%20resulted%20in%20the%20observed%20data"&f=false

使用 使用 jackknife ，反复测试模型，只留一个观察。

如果



当李雅普诺夫指数是正的时候，数据呈现成 混沌性。
reservoir connections
input-driven reservoir

怎样计算 李雅普诺夫指数？


BIC 越大，说明模型对于问题的拟合越好，












这个是不是可以作为创新点。实际中使用梯度。。


h



最后的误差分为两种，系统误差，和随机误差，什么是系统误差， 系统误差是在建模中遇到的，当观察的和计算的 和真实值在一致性的偏差。


https://books.google.com/books?id=o6aKDQAAQBAJ&dq="A%20model’s%20prediction%20is%20in%20the%20form%20of%20a%20predictive%20distribution%20which%20is%20an%20estimate%20of%20the%20underlying%20distribution%20that%20resulted%20in%20the%20observed%20data"&pg=PT348&v=onepage&source=ttb#v=onepage&q="A%20model’s%20prediction%20is%20in%20the%20form%20of%20a%20predictive%20distribution%20which%20is%20an%20estimate%20of%20the%20underlying%20distribution%20that%20resulted%20in%20the%20observed%20data"&f=false
























