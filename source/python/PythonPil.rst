[[Scipy]]
=========

PIL, !NumPy,opencv之间的相互转换
=======================================

转换两种一种是基于字节流的,from/tostring,还有一种那就是基于matrix,from/tomatrix.
#. `转换列表来自于新加坡大学的课程 <http://www.comp.nus.edu.sg/~cs4243/conversion.html>`_ 
#. `opencv,PIL,numpy 之间相互转换 <http://opencv.willowgarage.com/documentation/python/cookbook.html>`_  都是围绕矩阵来进行的

在numpy中都包含array和matrix，但是子array中”*“表示的是元素乘积。 "dot"表示的是内积和。

在matrix中 "*"表示矩阵相乘。“multiply”表示的是元素相乘。

这样记起来比较复杂，说明python中有些东西还是比较混乱的。

但是都可以通过np.inner and np.outter 表示内积和外积。但是在多维数组中很容易搞混。

matrix 和 array 互相转换
B是matrix  B.A 转换为 array.

a是array  np.matrix(a)是matrix。
See also
========

#. `python PIL Image模块 <http://hi.baidu.com/drunkdream/item/9c9ac638dfc46ec6382ffac5>`_  基本命令
#. `Python开发入门详解 <http://www.crifan.com/tutorial&#95;how&#95;to&#95;develop&#95;using&#95;python&#95;entry&#95;level/>`_  基础的东西，看一下
#. `Python 调试命令 <http://www.cnblogs.com/xuxm2007/archive/2010/09/08/1821524.html>`_  
#. `python 简单图像处理（11） 空间域图像锐化(边缘检测) <http://www.cnblogs.com/xianglan/archive/2010/12/29/1920490.html#header>`_  
#. ` 异常error处理 <http://woodpecker.org.cn/abyteofpython&#95;cn/chinese/ch13s02.html>`_  
#. `Python,PIL,ImageMagick,PythonCOM --我们能实现什么 <http://zoomq.qiniudn.com/ZQScrapBook/ZqFLOSS/data/20041216230103/>`_  PIL与 imagemagick  哪一个功能更强
#. `How to make List from Numpy Matrix in Python <http://stackoverflow.com/questions/5183533/how-to-make-list-from-numpy-matrix-in-python>`_  

#. `&#91;Numpy-discussion] IM &#61; Numeric + PIL + OpenCV <http://mail.scipy.org/pipermail/numpy-discussion/2002-March/013664.html>`_  有时间来看一下这个东东
#. `sympy 矩阵推导 <http://docs.sympy.org/0.7.2/modules/matrices/matrices.html>`_  有空看一下
#. `python while for循环 <http://biancheng.dnbcw.info/python/399154.html>`_  有空看下
#. `colorsys <http://docs.python.org/2/library/colorsys.html>`_  各种颜色系统之间转换公式
#. `conv,fft,常用函数 <http://www.physics.rutgers.edu/~masud/computing/WPark&#95;recipes&#95;in&#95;python.html>`_  
#. `constraint python 模块 <http://labix.org/python-constraint>`_  
#. `sympy输出公式 <http://hyry.dip.jp/tech/slice/slice.html/35>`_  
Thinking
========



__init__什么意思？文件是怎样调用的？怎样才能深入文件内部？

-- Main.GegeZhang - 25 Jun 2013


__init___ 是和类绑定在一起的。我想是相当于main函数吧，但是这快还不是很懂。

-- Main.GegeZhang - 28 Jun 2013


array 和 list 有什么区别？并且 array上面的transpose，conj， conjugate 好像都不能用，说明和list差不多？

-- Main.GegeZhang - 29 Jun 2013


list支持任意类型的数据，保存的是指针和数值，比较浪费内存和CPU时间。
array 是保存数值，但是python中数值运算比较少，因此PIL设计了Numpy支持多维。

-- Main.GegeZhang - 29 Jun 2013


array 中np.fromfunction(func2, (9,9))必须是数字？那怎样把字母转换为int型。

-- Main.GegeZhang - 29 Jun 2013


xv utility是什么

-- Main.GegeZhang - 30 Jun 2013


python中如何实现按任意键继续效果！


-- Main.GegeZhang - 11 Jul 2013


好像waitkey只是等待一个重画的图，并不是这个作用

-- Main.GegeZhang - 11 Jul 2013


rand和randn 区别？ 


rand 是0-1均匀分布， random是均值为0，方差为1 的正态分布。


-- Main.GegeZhang - 22 Jul 2013


list 怎样转换为array ？ 并且设置任意大小的？

1.  产生 a = []  list,
2.   a.append (object)  加入元素， 
3.   转换为 array     np.array(a,dtype=np.floate32)
list 本质和array一样，所以list 可以通过 np.array(list)转换为 array。



-- Main.GegeZhang - 30 Jul 2013


CSV文件时干什么用的？主要用在哪些方面？

-- Main.GegeZhang - 17 Aug 2013


通过 a = np.asarray(mat) 把 cv2.cv.cvmat 转换为 array

-- Main.GegeZhang - 27 Aug 2013



x = np.array([3,4,5])
if  (x>0).all()：
用于检测所有的x是否大于0，我想可能会有用

-- Main.GegeZhang - 30 Aug 2013


continue，pass, break目前看对于循环，continue 继续执行，break 就中止了这个循环，还要继续做测试。
k =0
for i in range(5):
    k =k+1
    if i<3:
        break

-- Main.GegeZhang - 03 Sep 2013


怎样从 np.array得到np.matrix??


-- Main.GegeZhang - 04 Sep 2013


多向老李学习下调试技巧。并且记下来自己有哪些问题

-- Main.GegeZhang - 04 Sep 2013


`使用python中的matplotlib 画图，show后关闭窗口，继续运行命令 <http://bbs.eetop.cn/viewthread.php?tid=382878>`_ 

plt.close() will close current instance.
plt.close(2) will close figure 2
plt.close(plot1) will close figure with instance plot1
plt.close('all') will close all fiures

-- Main.GegeZhang - 19 Apr 2014


`python setup 安装 <http://icereality.blog.china.com/201010/7121825.html>`_ 

-- Main.GegeZhang - 16 Jun 2014

