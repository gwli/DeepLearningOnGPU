#. `当前较新的机器学习理论是:流形学习(manifold learning),高斯过程(Gaussian processes). <http://blog.csdn.net/ericgogh/article/details/7264996>`_  
#. `高斯过程 主页  各种代码 资源 <http://www.gaussianprocess.org/>`_  
#. `kernel cookbook <http://mlg.eng.cam.ac.uk/duvenaud/cookbook/index.html>`_  

高斯过程多噪声比较敏感。


.. graphviz::

   digraph GP {
        nodesep=0.8;
        node [fontname="bitStream Vera Sans",fontszie=8,shape="record"]
        GP[label = "Gussian Process"]
        SP [color=yellow,
   	    label = "{StochasticProcess | \
   	              function familiy | \
   				  rondom variable is range}"
   	 ]		
        DF [ color=green,
           label = "{Distribute function  | \
   		         integration calculus function | \
   				  order is by input number}"
        ]
        GP ->SP -> DF; 
   }
   

+`高斯过程 <http://blog.sciencenet.cn/blog-520608-718474.html>`_ 
=====================================================================



高斯过程与核函数紧密联系，定义在y上的高斯分布是通过核函数表示出来的，与线性回归相比，高斯过程通过通过核函数的方式把x和y建立联系。在线性回归中，我们假设yde值服从某一个高斯分布
.. math:: p\left( y \right){\rm{ = }}N\left( {y|{w^T}\phi \left( x \right),{\sigma ^2}} \right)，即y的均值是w的一个线性变换。

%\[\begin{array}{c}
{\mathop{\rm cov}} \left( {\rm{y}} \right) = E\left( {y{y^T}} \right) = E\left( {\phi w{w^T}{\phi ^T}} \right)\\
 = \phi E\left( {w{w^T}} \right){\phi ^T} = \phi {\mathop{\rm cov}} \left( w \right){\phi ^T}
\end{array}\]%

注意到上面\phi是一个N\times M的设计矩阵，

这里\phi\phi^T可以通过一个核函数表示：

%\[k\left( {{x_m},{x_n}} \right) = \phi {\left( {{x_m}} \right)^T}\phi \left( {{x_n}} \right)\]%

可以理解为两个函数的相似关系。

%\[{C_{N + 1}} = \left( {\begin{array}{*{20}{c}}
{{C_N}}&k\\
{{k^T}}&c
\end{array}} \right)\]%

已知分布
.. math:: p\left( {{t_{N + 1}}} \right) = N\left( {{t_{N + 1}}|0,{C_{N + 1}}} \right)， 因此协方差矩阵是核函数给出的，这样高斯分布的性质容易得到：

%\[p\left( {{x_{N+1}}} \right) = N\left( {{x_{N + 1}}|{k^T}C_N^{-1}t,c - {k^T}C_N^{-1}k} \right)\]%

%RED% 这里的期望怎么计算的?? %ENDCOLOR%
#. `高斯过程简单理解 <http://www.cnblogs.com/tornadomeet/archive/2013/06/14/3135380.html>`_  
#. `协方差定义 <http://zh.wikipedia.org/zh-cn/&#37;E5&#37;8D&#37;8F&#37;E6&#37;96&#37;B9&#37;E5&#37;B7&#37;AE&#37;E5&#37;87&#37;BD&#37;E6&#37;95&#37;B0>`_  
#. `pmk&#95;projectpage <http://www.cs.utexas.edu/~grauman/research/projects/pmk/pmk&#95;projectpage.htm>`_  
#. `active Learning <http://www.doc88.com/p-705867908985.html>`_  



