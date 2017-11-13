******
Theano
******

install note
============

#. pip install Theano 
   #. pip install numpy 
   #. apt-get install python-scipy
#. apt-get install libblas-dev
   #. apt-cache search blas
  
introduction
============

theano利用python的语法来实现了自己的语言。这个语言的特别之处，自己实现的用自己的，自己没有实现的就可以直接使用了python的，不过只是一个中间件语言，下层实现还是直接借用CPU与GPU的blas库。并且theano实现还是一个JIT编译器。它的主要特点那就是直接使用利用sympy以及自己的张量符号表达式来直接构造公式,在theano中公式与函数基本上可以直接化等号了。 并且theano还有自己的编译算法，它可以直接打印出依赖图，如何简化自己的逻辑图呢。

既然是语言，就要变量，循环，分支，函数。那theano中是如何实现这些的。

*变量* 类型两部分，基本类型本身另外那就是结构，例如tensor.dscalar,就是double类型的标量

.. csv-table:: 

   b,w,i,l,f,d,c,  byte,word,int,long int,float,double,compex ,
   scalar,vector,matrix , ,


循环，又分为map/reduce两种。而在theano中采用了scan 来描述。这样是为了能够让theano识别for,只是其返回值，其中update是要function来用，另外是做计算结果来使用的。

分支
对于计算模型循环要远远大于分支，并且在大量的matrix不存在分支模型。所以在计算型的语言中分支会用的比较少，例如CG编程中主要是计算。

   
*函数*
直接用表达式来构造函数，但是没有看到其作用域的使用，没有直接声明函数的{}范围，其实直接使用引用传递，与可以实现 多值反回，就可以实现n对n的输出。正常的函数输出是n对1，n对n的输出，那就可以引用传递。在python 可以使用global来实现。 theano中函数声明没有传统语言的方式，直接符号表达式使用来当做函数体。当然这个过程它也会做一些符号计算。帮你做公式推导。

theano.function这个是编译指令，然后返回一个编译后object,就像shaderobject.
函数输入，解决了python中没有引用传递的问题，如果你把变量声明成shared类型，那就是引用传递了。也可以用borrow 的属性来直接使用引用传递。但保存一些python中好的习惯，参数可以有默认的值。格式Param(y,default=1)格式，也可以name/value来赋值。
函数体的可以直接使用输出表达式，也可以用update,given来指定。update 是用来指定如何操作shared变量，而givens,用来变量替换的。  theano这种方式可以复用达到每一条指令的颗粒度，即因为在没有编译之前都是符号表达式。

函数输出可以是多值输出，这可以任意的中间状态输出。



theano 结构
===========

现在明白了theano中对于优化问题的固定的结构，首先declare the variable,  then compile the data ,input, outputs, then tain the data;
finally look at the results. 

declare variable->construct the expression graph-> compile->train->show the results

+theano graph
=============

在调试的时候，我们必须知道什么是graph，包产variable，op等，首先输出的操作数，y.owner.op.name
对于函数输出y.maker.fgraph.outputs，一个函数形式，y.maker.fgraph.inputs是[x, w, b]，输入tensorvariable。

好像整个都是以函数的输出就是函数 theano.function{outputs} 中输出的类型就是function，无法得到实际数据。

多次迭代共同改变的是w的值。调节的是x，y 不断的输入，现在明白了为什么w，b是全局变量。

这里的prediction 好像没有什么用，只是预测。


 * `Matlab vs. scipy &#38; numpy <http://blog.pluskid.org/?p&#61;71>`_  
#. `theano学习指南 <http://www.cnblogs.com/xueliangliu/archive/2013/04/03/2997437.html>`_  明天看一下
#. `Intel MKL基础(1)了解MKL、MKL资源  <http://blog.csdn.net/gengshenghong/article/details/7011939>`_  CUDA 是不是有类似的库
#. `matlab <https://www.google.com.hk/url?sa&#61;t&#38;rct&#61;j&#38;q&#61;&#38;esrc&#61;s&#38;source&#61;web&#38;cd&#61;6&#38;ved&#61;0CE4QFjAF&#38;url&#61;https&#37;3a&#37;2f&#37;2fmidas3&#37;2ekitware&#37;2ecom&#37;2fmidas&#37;2fdownload&#37;2f&#37;3fitems&#37;3d20143&#37;2c3&#38;ei&#61;JxgDU7P-KI2YiAepmoCwBA&#38;usg&#61;AFQjCNFsTRm3&#95;KE-Dou&#95;hmM2SpdybIgNlw>`_  



*math compiler* 公式编译器，原理你把公式写出来，编译器就会把公式化简然后转化为高性能代码来进行计算。因为各个平台都会有各自己的高性能计算库，例如blas,lapack,intel的mkl,还有nv的cuda等等都是有现成了运算库，然后我们需要把算法以及矩阵运算，然后把分块分工，然后分配给这些计算单元并行。看来编译语法分析是重点。自己一定要自己实现一个解析器。

-- Main.GangweiLi - 22 Jan 2014


*张量计算* 这个是theano这个库所主要实现的功能，这个是sympy所没有实现的功能。另外的那就是公式编译器。其通过实现一个图，采用利用图论的最短路径来进行化简。

-- Main.GangweiLi - 26 Jan 2014


theano利用自己的内存分配机制，例如share型，一个独立的类型，就像语言一样有register型，对于异构编程的时候，会有对应的内存，例如CPU内存，GPU内存，还有共享区域。

-- Main.GangweiLi - 26 Jan 2014


我是不是要实现一门自己的语言，然后把它转化成各种语言，以后我只写我自己的语言，然后再进行转换。

-- Main.GangweiLi - 26 Jan 2014


*NVCC*  这个封装与python自身的那个 destutil 差不多，自己封装一个toolchain for NVCC, 当然要写配置文件，分析配置文件直接使用的是python 的configparser 库，看来自己以后可以直接这个了。

-- Main.GangweiLi - 07 Mar 2014


*优化* 对于不认识的新op是不会进行优化的。

-- Main.GangweiLi - 07 Mar 2014
