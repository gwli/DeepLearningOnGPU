:author: GangweiLi
:comment: No
:CreateDate: 14-07-10
:status: C
:name: Python Language

Introduction
------------

Until now, the python is very concise language. It have all the metrics of previous language. You can the shadow of every language in it. 
it use the space indent for delimiter. it adapt the function programming technique. it use the file name as module.  

It has the OOP and basic procedure language.  the data structure : list, array, dict(hash table).  the data has constant meta-list just the C/C++.
it use =global= instrument is just like the tcl's "global".  there is import is mix namespace import and export and source.
the class just like the java, the first parameter of method is self point.  there is also exec,eval assert raiseError just like tcl. and lambda in Scheme.

os and sys 对于系统的操作
-------------------------

可以考虑用python脚本来代替shell脚本。并且要考虑并行的问题。
#. `can-i-use-python-as-a-bash-replacement <http://stackoverflow.com/questions/209470/can-i-use-python-as-a-bash-replacement>`_ 这个讨论已经写的很详细了。
there is two core module: *sys* and *os*. just like *info* in tcl.  sys is most about python interpreter. for example, the sys.path is object search path for lib and module.  `here <http://woodpecker.org.cn/abyteofpython_cn/chinese/ch14s03.html][os ]] is most about operation system. for example, os.environ["XXX"],you can get the ENV VAR.   [[http://woodpecker.org.cn/abyteofpython_cn/chinese/ch14s02.html][sys.argv]] more details about sys see [[http://docs.python.org/2/library/sys.html>`_

并且以后要把print都要换成`logging  <http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python>`_来全用。 用python popen处理一些系统的命令它会返回一个对象，

.. code-block::

      import os
      print os.popen('ping g.cn')
      #!/usr/bin/env python
      import sys
      import os
      topdir = os.path.dirname(os.path.abspath(__file__))
      if sys.platform == 'cygwin':
      	topdir = os.popen('cygpath -a -m %s' % (topdir), 'r').read().strip()
      sys.path.append(topdir+"/lib")
      from nose import main
      
      if __name__ == '__main__':
          main()

它返回一个文件对象，你可以对这个文件对象进行相关的操作。

但是如果你想能够直接看到运行结果的话，那就要用到python os.system,用了以后，立竿见影！
还是上面的问题：
.. code-block::
        import os
        print os.system('ping g.cn')
        输出的结果是：
        64 bytes from 203.208.37.99: icmp_seq=0 ttl=245 time=36.798 ms
        64 bytes from 203.208.37.99: icmp_seq=1 ttl=245 time=37.161 ms

Python 内在函数
---------------

.. csv-table:: builtin namespace

    dir() , __name__ ,    ,__file__ 
     built-in command , http://docs.python.org/2/library/functions.html 
     `Tuple 介绍 <http://woodpecker.org.cn/diveintopython/native_data_types/tuples.html>`_ , 不可变的list 
     SymbolPy , 符号计算 
     MatPlotlib, 画图
     NumPy , 矩阵与线性代数
    
Python 包管理与开发环境
-----------------------

Python的包管理就像perl 的CPAN一样。 `easy_install 与pip 新替 <http://pypi.python.org/pypi][ pypi]] 就是python 的CPAN。[[http://jiayanjujyj.iteye.com/blog/1409819>`_ egg文件就是python的打包文件。
#. `打包机制 <http://woodpecker.org.cn/diveintopython3/packaging.html>`_ .egg文件
#. `perlbrew <http://blog.caokee.com/2012/02/pythons-virtual-environment-and-multi-version-programming-tools-virtualenv-and-pythonbrew.html][python virtual environment and multi-version]] ,[[https://github.com/gugod/App-perlbrew#readme>`_perl的虚拟环境。ruby也有。
#. `pythonbrew online document <http://pythonbrew.readthedocs.org/en/latest/>`_
#.
#. `VirtualEnv 和Pip 构建Python的虚拟工作环境 <http://www.v2ex.com/t/42760>`_ 这个写的不错，virtualEnv 解决的就是不同库依赖之间的问题。并且有实例。而pythonbrew主要解决了不同引擎之间切换。同是兼容了，virtualenv这样的样的环境。这样就可以在版本与库之间进行选择了。就像pyrobot一样，就可以选择环境，选择brain.
#.用pythonbrew指是采用哪一个python, 而virtualEnv 指的在哪一个环境下使用python. 其本质是与linux的chroot是一样的道理。
#.
#. `python configparser.py <http://docs.python.org/2.7/library/configparser.html>`_ 以后配制文件，可以使用它，而不用自己在写分析了，有了一个标准的分析库。它采用的是windows INI 文件格式。

`包的开发与目录结构 <http://www.math.pku.edu.cn/teachers/lidf/docs/Python/6.html>`_ import 可以是整个包也可以只是变量，函数。但是python把命名空间与import并且source的功能混在一起了。看起来有一些不舒服。
for install and manipulate the package of python, just like pkgIndex in tcl. there is `distutils.core <http://docs.python.org/2/distutils/apiref.html>`_.  which manage the preprocess, compiler,linker, verification, install.here has some useful command:

.. csv-table:: cmd

   setup , 
   distutils.ccompiler , set_libraries, add_library_dir,add_runtime_library_dir,define_macro ,dir_utils,file_utils (mkdir,rm,copy_tree) , ` distutils-simple-example <http://docs.python.org/2/distutils/introduction.html#distutils-simple-example>`_ , this is helpful when you are writing more code. 
   when you install some extention module written from C/C++. you and build environment. you can gcc or MSBUILD. `setup.py  compiler options manual  <http://docs.python.org/2/install/>`_
   如果要`打包成可执行文件 原理 <http://wiki.woodpecker.org.cn/moin/LeoJay/PyPackage>`_，
   pyInstaller ,  http://www.pyinstaller.org/ticket/512 opencv 好像还没有支持 
   `cx-freeze <http://cx-freeze.sourceforge.net/>`_ 
   `py2app <http://svn.pythonmac.org/py2app/py2app/trunk/doc/index.html>`_ , mac 平台 

还有setup的编译环境，其实就是一个makefile,就是类似于ndk的东东，只是写了一个类与配制文件，把这些手工的步骤给封装了起来，对于C/C++编译就是那几步了，同时会把编译，链接的库的路径等等都会设置好，系统默认值一些值都会自动搜索系统目录，例如对于VC就会使用注册表信息去找这些。而对于linux gcc,windows cygwin,mingw等等都是这样的。 对于python 自己distutil 包setup 相当于python的grudle一样，在里面把所的配置信息写好，扩展的化就类似distutil.compiler类来做了。
并且theano也是采用这样的方式来封装nvcc的。


以及各种开发模式，插件式与模块化的区别与连系是什么。
例如python ETS插件式开发，http://code.enthought.com/projects/ .

包管理的难点，在一个单一环境是容易的，难点各种包管理模式之间的冲突，但是apt-get 与pip 如何兼容的，包管理本身也需要一定的信息结构。例如依赖关系，linux讲究的相互共享，这就造成了，系统升级之后，就莫名其妙的不能用了，而windows采用的是自包含，所以现在windows会变非常大。`包管理的特点 <http://shzhangji.com/blog/2012/11/18/aosa-python-packaging/>`_  依赖关系，方便的查询操作，以及编译环境的准备。这个有perl，python,以及gentoo的包管理，都非常熟悉。
namespace
---------

#. `python学习笔记——模块和命名空间 <http://blog.sina.com.cn/s/blog_4b5039210100ennq.html>`_ 在python中是一切对象，这个与lisp一切结数据的模式是很像的，现在还不知道hackshell的编程模型是什么。python的几种命名空间，对于简单函数调用，python可以像传统的面过过程一样，直接调用其函数，也可以采用面向对象方式。python面向对象机制是不是有一点像perl,但是它的面过过程的调用是通过静态函数来变通的。还是本来两种方式都是可以的。
   #. build-in name space  
   #. global name space 
   #. local name space 


string,list,dict/hash and tuple
-------------------------------

*String* is Object itself, so when you manipulate string. you do it like this "XXXX".append("XXX"); one of important is regular expression. for python you use *`re <http://www.cnblogs.com/huxi/archive/2010/07/04/1771073.html>`_* 

+-------------+------------------------------------------------------------------------------------+
| u'a string' |  prefix u stand for unicode character                                              |
+-------------+------------------------------------------------------------------------------------+
| r'a string' |  prefix r  stand for original string   means     regular expression is object too. |
+-------------+------------------------------------------------------------------------------------+

.. code-block:: 

   pattern = re.compile(r'hello')
   match = pattern.match('hello world!')
   match.group()

dictionary{}必须是key-value对，核心是哈希，内容可以使任何元素，可是实现删除，del and
d.clear()。里面的key是虚幻的。


list中是有顺序的，因此可以insert, append, extend.
list中就是数组操作，比如插入，remove，她的所有操作都是基于index的。里面的index是顺序排列的，比如123.。
应该讲的有条理些，如果我现在不做，就找不到工作。

tuple （）就是不可以改变的。

为了能够确定对象的属性，python使用一些系统参数比如 str, callable, dir:

str 主要是字符串操作，可以帮我找到 modulate的位置，其他的有么用不太清楚。

callable 主要从对象中找出函数。

dir列出所有的方法的列表。

getattr()得到对象的属性。

doc string 可以打印方法函数的document。

Python vs.C/ Matlab

其智能化主要体现在 "+"可以同时实现 字符串连接和算术运算。

多变量赋值，简化操作，就像perl一样。

逻辑运算：and，or 相比更加容易理解。

很多格式都是规范的，比如 indent，list.

Python 中 的class
-----------------

什么是类，我想就是分情况，然后需要的初始化__init_，一个class定义一种__init__就是初始化函数，里面的self就是参数赋值，然后就是def各种方法，利用参数值。

各种字符串之间的转换（dictionary->str，list->str）
-------------------------------------------------

list->str 可以通过"".join(li)实现， 但是不要通过str(),这种属于硬转换(只是在外面加了一个“”).

str-》list， bd.split(",")好像不行，因为split适用于把有一定界限的str分离。
 
 dict-》str， str(dict) 我觉得不太行，还是硬转换。？？

 str->dict,   eval(str)很多网站说这个是字符串转换，但是我觉得并不能成为字符串转换吧。原意是evaluate。  


python  comments
----------------

comments is important part of an programming language. most of the document is generated from the comments in code.  One orient, is putting document into code, which can be easier to maintain and update. 
so structure and format is important for an programming language. take compare several language.

      +--------+-----------------------------------------------------------------+-------------------------------------------------+
      | perl   |   has pod document system, and << STRING, and format report     |  pod2tex,pod2man,pod2pdf                        |
      +--------+-----------------------------------------------------------------+-------------------------------------------------+
      | java   |  javadoc                                                        |                                                 |
      +--------+-----------------------------------------------------------------+-------------------------------------------------+
      | c /C++ |  if you adopt the C/C++ syntax, you can use doxygen to generate |                                                 |
      +--------+-----------------------------------------------------------------+-------------------------------------------------+
      | python |  __doc__ ,__docformat__,reStructuredText                        |  python has puts comments as variable of python |
      +--------+-----------------------------------------------------------------+-------------------------------------------------+


you can access the comments from in the code of __doc__.  one usage for this is just like CAS testcase steps:

.. code-block::

   def tounicode(s):
       """Converts a string to a unicode string. Accepts two types or arguments. An UTF-8 encoded
       byte string or a unicode string (in the latter case, no conversion is performed).
   
       :Parameters:
         s : str or unicode
           String to convert to unicode.
   
       :return: A unicode string being the result of the conversion.
       :rtype: unicode
       """
       if isinstance(s, unicode):
           return s
       return str(s).decode('utf-8')


http://docutils.sourceforge.net/docs/peps/pep-0257.html
也就是基本的原则，语法还可以用markdown以及sphinx,只是函数模块类等的一第一段注释会被处理成文档。并且支持中文用u"""就可以了，以及r"""
256，224，216，这几篇都看一下。


command line
------------
for python, you can process comand line options in three way:
#. sys.argv
#. getOption
#. plac module   `Parsing the Command Line the Easy Way <https://ep2013.europython.eu/media/conference/slides/plac-more-than-just-another-command-line-arguments-parser.pdf>`_ 
#. `argparse <http://docs.python.org/2/library/optparse.html>`_ this one looks good for me, it just like getOption, but stronger than her.


mutli-thread of python
----------------------

多线程与进程一样，可以动态的加载与实现，而不必须是静态。并且可以是瞬间的，还是是长时间的。之前的理解是片面的，这个受以前学习的影响，一个线程或者线程就像一个函数根据其功能的来，不是说是线程就要有线程同步。可以是简单的做一件事就完的。例如实现异步回调呢，就可以是这样的，把回调函数放在另一个线程里。用完释放掉就行了。`C#线程篇---Windows调度线程准则（3） <http://www.cnblogs.com/x-xk/archive/2012/12/03/2795702.html>`_ 如何让自己的程序更快的跑完，其中在不同提高算法性能的情况下，那就是占一些CPU的时间片，优先级调高一些，就像我们现在做事一样，总是先做重要的事情。然后按照轻重缓级来做。就像找人给干活的时候，你总经常会说把我的事情优无级高一些。先把我的事情做完。 这个应该可以用转中断来实现。
` Lib/threading.py <http://www.laurentluce.com/posts/python-threads-synchronization-locks-rlocks-semaphores-conditions-events-and-queues/][Python threads synchronization: Locks, RLocks, Semaphores, Conditions, Events and Queues]],[[http://docs.python.org/2/library/threading.html>`_

例如以前的，我都是利用傻等的方式，还有时间片或者用sleep,其实异度等待的机制可以用`线程事件来高效实现 <http://blog.csdn.net/made_in_chn/article/details/5471524>`_

把这些东西优化到编程语言这一层那就是协程了，python 中 yield就是这样的功能。通过协程就可以原来循环顺序执行的事情，变成并行了，并且协程的过程隐藏了数据的依赖关系。 对于编程语言中循环就是意味着顺序执行。如何提高效率，实别的计算中数据依赖问题，把不相关的代码提升起来用并行，采用协程就是这样的原理。 这也就是什么时候采用协同。什么时候采用协程了。这个优化是基于实现的优化是基于你的资源多少来的。所以在python对于循环进行了优化。所以写循环的时候就不要再以前的方式了，采用计算器了，要用使用yield的功能。来进行简化。`coroutine <http://blog.dccmx.com/2011/04/coroutine-concept/>`_, 线程就是它什么时候执行，什么开始都是由内核说了算的。你就控制不了。coroutine就是提供了在应用程序层来实现直接的资源调度，如果更直接控制调度，另一个就是采用CUDA这样更加直接去操作硬件资源。


对于状态进度的更新有了一个更好的方法，注册一个时间片的中断函数，每一次当一个时间片用完之后，就来打印一个进度信息就不行了。这样就可以实时的知道进度了。
`Linux环境进程间通信 <http://www.ibm.com/developerworks/cn/linux/l-ipc/part2/index1.html>`_  目前看来需要在进度的SWap时来做的，需要内核调度函数提供这样一个接口。那就是在线程切换的时候，可以运行自定义的函数。其实这个就是profiling的过程。在编译的时候，在每一个函数调有的前后都会加上一段hook函数。我们需要做的事情，把切换的过程也要给hook一下。这个就需要系统的支持了。`coroutine的实现 <http://blog.dccmx.com/2011/05/coroutine-implementation/>`_ linux下可以有libstack库来支持，当然 了可以直接在C语言中嵌入汇编来实现。用汇编代码来切换寄存器来实现。

现在对于C语言可以直接操作硬件，这种说法的错误。同为一种语言凭什么说C可以操作硬件。原因在于好多的硬件直接C语言的编译器而己尽可能复用以前的劳动成果而己。只要你能把perl,python,各种shell变成汇编都能直接操作硬件的。
 

 
现代语法
--------

`List comprehensions  <http://docs.python.org/2/tutorial/datastructures.html>`_ 也开始发展perl的各种符号功能

Ilterators generators   

.. code-block::
   a = [expression for i in xxx if condition]   //list comprehensions
   a = (expression for i in xxx if condition)   //list generator 
   a = [(x,y) for x in a for y in b] 这个不同于双层循环
   a = [expression for x in a for y in b ]这个相当于双层循环

再加上 http://stackoverflow.com/questions/14029245/python-putting-an-if-elif-else-statement-on-one-line 对了可以使用lamba来实现

 `Python yield 使用浅析 <http://www.ibm.com/developerworks/cn/opensource/os-cn-python-yield/>`_  原理也简单，既然可以lamba 可以部分求值。yield的机制也就是执行变成半执行。参加的功能那就是计录了前当前的状态。当下一次调用时候，就可以直接恢复当前环境。执行下一步了。yield的功能其实就是中断恢复与保存机制。每一次遇到就这样保存退出。并且也保证了兼容性。下面的例子也就说明了问题。其实就是集合的表达方式问题。我们采用列举式还是公式表达式。  数据的表达方式就是集合表现方式。研究明白了集合也就把如何存储数据研究明白了。列表相当于我们数据采用列举式，而生成式我们采用是公式表示。

.. code-block::

   range(6)  [1,2,3,4,5,6]
   xrange(6)   相当于定义了类，最大值是6，最小值是0，步长为1，当前值为0.每调用一次，更新一下当前。当然利用这个是不是可以产生更多数更加复杂表达方式。同时也解决了以前在CAS的那sendMutliCmd中循环，无法记录自身当前值问题，必须使用global去使上一层变量的方法，现在通过这个yield方法就会非常方便。这个其实编程语言中闭包问题，就是在子函数中调用复函数中局部变量，在tcl中可以使用upvar来实现。使用动态代码实现一个子函数来进行调用。而在python这里可以直接yield来产生。同样也可以自己实现。
   class repeater {
     init;
    step;
     current:
     next: 调用一次method
     reset:
     set:
     method{ output=current+step;current=output}
    
   }

这样就用计算代替了存储。并且解决吃内存的问题。
   `65285-looping-through-multiple-lists <http://code.activestate.com/recipes/65285-looping-through-multiple-lists/>`_  可以使用map,zip以及list来实现。
   `yield与labmda实现流式计算 <http://www.cnblogs.com/skabyy/p/3451780.html>`_

`itertools <http://docs.python.org/2/library/itertools.html>`_ 更多的迭代器可以采用这些，这些采纳了haskell中一些语法。


Descriptors properites

Decorators
   * `Python装饰器与面向切面编程 <http://www.cnblogs.com/huxi/archive/2011/03/01/1967600.html>`_ %IF{" '这个其实是perl那些符号进化版本' = '' " then="" else="- "}%这个其实是perl那些符号进化版本
其实本质采用语法糖方式 ，其实宏处理另一种方式。在C语言采用宏，而现代语言把这一功能都溶合在语言本身里了。decorator直接采用嵌套函数定义来实现的。最背后是用lamba来实现 的。`functools <http://www.cnblogs.com/twelfthing/articles/2145656.html>`_提供了对于原有函数进行封装改变的方便方式。也就是各种样的设计模式加到语言本身中。

python对于循环进行了优化。所以写循环的时候就不要再以前的方式了，采用计算器了，要用使用yield的功能。来进行简化。
map,reduce机制，例如NP就经常有这样的操作，例如

.. code-block::

   x,y,z=np.random.random((3,10) 每一个一行。



并行处理
--------

以后要把for循环升级到map,reduce这个水平，两个概念是把循环分成有记忆与无记忆，map就是无记忆，reduce是有记忆。 `Python函数式编程——map()、reduce() <http://www.pythoner.com/46.html>`_ 就是为了并行计算，但是内置的这两个函数并不是并行的，
可以使用  `multiprocessing <http://bubblexc.com/y2011/470/][PProcess.map/reduce]]来直接实现，并且是不是可以把列表中元素直接换成函数，不就可以直接实现任意事件的并行了。这个有点类似于cuda的并行计算了 另外那就是利用[[http://docs.python.org/3.3/library/multiprocessing.html>`_ 来进行。



C extending Python
------------------

`对象机制的基石——PyObject <http://www.ibm.com/developerworks/cn/linux/l-pythc/][用C语言扩展Python的功能]] just like SWIG for tcl. there is stand process for C on python.   The big problem is that data type converstion.    [[http://book.51cto.com/art/200807/82486.htm>`_ PyObject 本质就是结构体指针加一个引用计数。

test framework of python
------------------------

   * `使用再简短手册 <https://nose.readthedocs.org/en/latest/][nose]] NOSE 底层驱动unittest 来进行的。[[http://wenku.baidu.com/view/422b7585b9d528ea81c77967.html>`_最快的方法那就直接问Ryan.
   * `pexpect <http://www.ibm.com/developerworks/cn/linux/l-cn-pexpect1/index.html>`_ 我们的GDBtest 是采用pexpect来进行gdb交互的。 今天出现工作不稳定的问题，是因为python版本不高造成，直接在cygwin中升级一下python就行了。

Data structure
--------------

  embeded dict. `what-is-the-best-way-to-implement-nested-dictionaries-in-python <http://stackoverflow.com/questions/635483/what-is-the-best-way-to-implement-nested-dictionaries-in-python>`_ 其中一个方法hook __getItem__ 来实现，但是有一个效率问题，其实那种树型结构最适合用mongodb来实现了。并且搜索的时候可以直接使用MapReduce来直接加快计算。
  
 `High-performance container datatypes <http://docs.python.org/2/library/collections.html>`_  同时还支持 `ordered Dictionary <http://code.activestate.com/recipes/576693/>`_ `同时支持对基本数据结构进行扩展，利用继承 <http://woodpecker.org.cn/diveintopython/object_oriented_framework/special_class_methods2.html>`_ 。

embeded system
--------------
#. `python  单片机开发 <http://ikeepu.com/bar/10455735>`_ 
#. `基于arm-linux的嵌入式python开发 <http://jim19770812.blogspot.com/2011/06/arm-linuxpython.html>`_



python data analysis
--------------------
python主要用于大数据分析的比较多，大的数据分析主要包括三个方面:
数据本身的存储,分析，批量处理，以及可视化的问题

数据存储，关键是效率

#. csv 最简单直接，并且方便扩展
#. xml 机器交互性强，但是不算太方便
#. npz 最简单直接
#. python 本身的串行化，效率不高。
#. pyData/pyTable 对大数据的存储
#. h5py 这个压缩存储

`best way to preserve numpy arrays on disk <http://stackoverflow.com/questions/9619199/best-way-to-preserve-numpy-arrays-on-disk>`_ 

分析计算
#.numpy,pandas,`blaze 下一代的numpy,总结pyData,pyTable,pandas <http://blaze.pydata.org/docs/latest/overview.html>`_ 
例如优化算法，以及优化求解等，同样可以pyomo等之类的库来实现。


可视化:
pylab,VTX以及直接利用opengl来计进行。
以及reportLib 对于pdf的直接读写。以及使用pyplot来进行二维以及三维的画图。`pandas plotting <http://pandas.pydata.org/pandas-docs/stable/visualization.html>`_ .


正是由于python的一切对象机制，使其把投象与具体结合起来，可以很方便应用到各个学科与领域，其实这个本身就是一个知识库。现在需要一个快速推理管理工具。

专业领域的应用
--------------

.. csv-table::
  化学,` openbabel <://openbabel.org/docs/current/index.html>`_ ,
  仿人机器人实时建模,pyrobot,http://wenku.baidu.com/view/b643988484868762caaed542.html 并且代码在自己的 /home/devtoolsqa8/pyrobot
  信号与图像处理,sift,Signal and Image process
  音乐,https://code.google.com/p/pyfluidsynth/ https://wiki.python.org/moin/PythonInMusic

例如对编程本身的支持，

但是python本身也自身的缺点，一个方面那就是GIL，并且他的效率是依赖C或者其他。不过python的一切皆对象方式不是错。可以把python当做一个描述语言。
具体让编译器来做翻译。
一个软件好用不好用的关键，是不是大量相关的库，在科学计算领域python是无能比了。自己尺可能用高阶函数来表达核心的东东，而不必纠结实现细节，其实道理都是一样的。
对于python的扩展这里提到cffi来扩展。以及bitey. 以及用distutils功能完全可以用来实现gradle所具有一切功能。
例如强大的 c++ boost库，同样也有python的接口 见 http://www.boost.org/doc/libs/1_55_0/libs/python/doc/。

下一代了 `pypy <http://www.oschina.net/translate/why_pypy_is_the_future_of_python?print>`_ . 

ipython notebook
----------------

其实就相是CDF的一种形式，可计算文档的结构。特别适合写paper来用。并且也实现了文学编程的模式。

并且可以直接保存在github上然后直接用http://nbviewer.ipython.org/ 直接在线的显示，是非常的方便，自己只需要用就行了。然后干自己的主业就行了。并且其支持与sphinx的之间格式的转化。


python as shell
---------------
http://pyist.diandian.com/?tag=ipython
现在看来，自己想要常用功能都有，只要把find,与grep简单的整一下，再结合%sx,与%sc,就无敌了，并且也不需要每一次都写到文件里，可以放在python 的变量里，因为python的变量要bash的变量功能要强大的多。
支持用iptyhon，尽可能，只要离开就要提出一个bug.这样就可以大大的提速了。直接继承一个magic class就可以简单，然后直接loadext就可以了，实现起来简单。自己也慢慢往里边添加自己的东东。可以参考在python里直接执行c的插件。看来这个扩展还是很容易的，把知识代码化，而不再只是文本描述。

并且ipython提供了类似于tcl中多解释器的方式，来实现多进程与kernel的并行，可以让并行计算随手可得，并且解决了GIL的问题，并且能够与MPI直接集成。%px 这个插件，看来是要升级自己的shell从bash到ipython了。if expand("%") == """|browse confirm w|else|confirm w|endif"""

在ipython  中使用vim mode其实也很简单，直接配置readline这个库就行，正是因为linux的这种共享性，只要改了readline的配置文件，那么所有用到它的地方都会改变，一般情况下，默认的文件放在/usr/lib里或者/etc/下面。这里是全局的。
http://stackoverflow.com/questions/10394302/how-do-i-use-vi-keys-in-ipython-under-nix
http://www.linuxfromscratch.org/lfs/view/6.2/chapter07/inputrc.html

.. seealso::

#. `flask <http://flask.pocoo.org/>`_ %IF{" 'Flask is a microframework for Python based on Werkzeug,Jinja 2 and good intentions.' = '' " then="" else="- "}%Flask is a microframework for Python based on Werkzeug,Jinja 2 and good intentions.
#. `subprocess <http://docs.python.org/library/os.html][os]] %IF{" 'android my be use this module.' = '' " then="" else="- "}%android my be use this module. and [[http://docs.python.org/library/subprocess.html>`_ which just like system call of Perl or expect? which one?
#. `A Byte of Python <http://sebug.net/paper/python/>`_ %IF{" 'an introduction tutorial' = '' " then="" else="- "}%an introduction tutorial
#.  1. data structure  list, metalist, dict,class,module
#. `python PEP <http://www.python.org/dev/peps/pep-0405/>`_ %IF{" 'what is PEP' = '' " then="" else="- "}%what is PEP
#. `在应用中嵌入Python <http://gashero.yeax.com/?p&#61;41>`_ %IF{" '' = '' " then="" else="- "}%
#. `Python on java <http://www.java2s.com/Open-Source/Android/android-core/platform-sdk/com/android/monkeyrunner/JythonUtils.java.htm>`_ %IF{" '' = '' " then="" else="- "}%*Commute between Python and java* JythonUtils.java there use hash table to mapping the basic data element between java and python.
#. `org.python.core  <http://web.mit.edu/jython/jythonRelease&#95;2&#95;2alpha1/Doc/javadoc/org/python/core/package-summary.html>`_ %IF{" 'the online manual' = '' " then="" else="- "}%the online manual
#. `jython offical web <http://www.jython.org/>`_ %IF{" '' = '' " then="" else="- "}%
#.
#. `install sciPy on linux <http://www.scipy.org/Installing&#95;SciPy/Linux#head-fb320be917b02f8fbe70e3fb2c9fe6f5f5f06fc2>`_ %IF{" '科学计算' = '' " then="" else="- "}%科学计算
#. `python and openCV <http://www.opencv.org.cn/index.php/Python&#37;26OpenCV>`_ %IF{" '' = '' " then="" else="- "}%
#. `ipython <http://ipython.org/>`_ %IF{" '' = '' " then="" else="- "}%
#. `python for .net  CLR <http://pythonnet.sourceforge.net/>`_ Just like Java for JPython, anything in .net you can use via clr.
#. `Python之函数的嵌套 <http://developer.51cto.com/art/200809/90863&#95;4.htm>`_ %IF{" '' = '' " then="" else="- "}%
#. `简明 Python 教程 <http://woodpecker.org.cn/abyteofpython&#95;cn/chinese/index.html>`_ %IF{" '' = '' " then="" else="- "}%
#. `Python 中的元类编程，这才是python 所特有的东西。 <http://www.ibm.com/developerworks/cn/linux/l-pymeta/index.html>`_ 元类是什么，就是生成类的类。
#. `五分钟理解元类 <http://blog.csdn.net/lanphaday/article/details/3048947>`_ %IF{" '' = '' " then="" else="- "}%
#. `Python 描述符简介 <http://www.ibm.com/developerworks/cn/opensource/os-pythondescriptors/index.html>`_ %IF{" '还是不太懂' = '' " then="" else="- "}%还是不太懂
#. `Python 自省指南 如何监视您的 Python 对象 <http://www.ibm.com/developerworks/cn/linux/l-pyint/index2.html>`_ %IF{" '' = '' " then="" else="- "}%
#. `可爱的 Python: Decorator 简化元编程 <http://www.ibm.com/developerworks/cn/linux/l-cpdecor.html>`_ %IF{" '' = '' " then="" else="- "}%
#. `Python的可变长参数 <http://www.cnblogs.com/QLeelulu/archive/2009/09/09/1563148.html>`_ %IF{" '' = '' " then="" else="- "}%
#. `cuda support python <http://docs.continuum.io/numbapro/index.html>`_ %IF{" '' = '' " then="" else="- "}%
#. `cuda python <http://news.zol.com.cn/361/3610272.html>`_ %IF{" '' = '' " then="" else="- "}%
#. `欢迎使用“编程之道”主文档! <http://pythonhosted.org/daot/>`_ %IF{" '基于python更接近于自然语言' = '' " then="" else="- "}%基于python更接近于自然语言
#. `how-to-install-pil-on-64-bit-ubuntu-1204 <http://codeinthehole.com/writing/how-to-install-pil-on-64-bit-ubuntu-1204/>`_ %IF{" '' = '' " then="" else="- "}%
#. `marshal 对象的序列化 <http://blog.csdn.net/jgood/article/details/4545772>`_ %IF{" '' = '' " then="" else="- "}%
#. `python PIL <http://www.pythonware.com/products/pil/>`_ %IF{" '' = '' " then="" else="- "}%
#. %IF{" '' = '' " then="" else="- "}%
#. `sorted <http://docs.python.org/2/howto/sorting.html>`_ %IF{" 'key 与cmp到底有什么区别' = '' " then="" else="- "}%key 与cmp到底有什么区别
#. `python-convert-list-to-tuple <http://stackoverflow.com/questions/12836128/python-convert-list-to-tuple>`_ %IF{" '' = '' " then="" else="- "}%
#. `pygame <http://eyehere.net/2011/python-pygame-novice-professional-1/>`_ %IF{" '在研究游戏的时候来看一下' = '' " then="" else="- "}%在研究游戏的时候来看一下
#.
#. `python 图像应用实例 <http://scipy-lectures.github.io/#>`_ %IF{" '里面有很多代码，有空的时候要看一下' = '' " then="" else="- "}%里面有很多代码，有空的时候要看一下
#. `python 多继承 <http://christophor.blog.163.com/blog/static/16215437320107276239434/>`_ %IF{" '' = '' " then="" else="- "}%
#. ` windows7下使用py2exe把python打包程序为exe文件 <http://blog.csdn.net/xtx1990/article/details/7185289>`_ %IF{" '' = '' " then="" else="- "}%
#. ` 函数迭代工具 <http://www.cnblogs.com/huxi/archive/2011/07/01/2095931.html>`_ %IF{" '' = '' " then="" else="- "}%
#. `python 字节码文件（.pyc）的作用与生成 <http://hi.baidu.com/smithallen/item/fa2b77e5438908c5bbf37db4>`_ %IF{" 'python 可以把pyc 当做二进制发布，当然可以也可以自己加密使用' = '' " then="" else="- "}%python 可以把pyc 当做二进制发布，当然可以也可以自己加密使用
#.
#. `python-with-statement <http://effbot.org/zone/python-with-statement.htm>`_ %IF{" '这个要求你的类，自己有enter,exit函数，with 会自动调用这些。' = '' " then="" else="- "}%这个要求你的类，自己有enter,exit函数，with 会自动调用这些。


thinking
--------

*Jython embedded and extension with java*
just like right diagram, you there are three way call the jython, there an other way is extend the jython with the java. there are some interface to follow. and there is mapping between your jython data type and java data type. they provided some converting function.  java can use the jython installed on the PC.  
androidRobot reference the example `monkeyrunner.JythonUtils.java <http://www.java2s.com/Open-Source/Android/android-core/platform-sdk/com/android/monkeyrunner/JythonUtils.java.htm>`_  robot run on its base.

@MonkeyRunnerExported is used to generate _doc_ for python method,  _doc_ is built-in string for documentation.
JLineConsole(); Just support single line command? `PythonInterpreter source code <http://code.google.com/p/jythonroid/source/browse/branches/Jythonroid/src/org/python/util/PythonInterpreter.java?spec=svn30&r=30>`_   
<verbatim>
at ScriptRunner.java, via run.  bind the robot->RobotDevice.
 public static int run(String executablePath, String scriptfilename, Collection<String> args, Map<String, Predicate<PythonInterpreter>> plugins,Object object)
/*     */   {
/*  79 */     File f = new File(scriptfilename);
/*     */ 
/*  82 */     Collection classpath = Lists.newArrayList(new String[] { f.getParent() });
/*  83 */     classpath.addAll(plugins.keySet());
/*     */ 
/*  85 */     String[] argv = new String[args.size() + 1];
/*  86 */     argv[0] = f.getAbsolutePath();
/*  87 */     int x = 1;
/*  88 */     for (String arg : args) {
/*  89 */       argv[(x++)] = arg;
/*     */     }
/*     */ 
/*  92 */     initPython(executablePath, classpath, argv);
/*     */ 
/*  94 */     PythonInterpreter python = new PythonInterpreter();
/*     */ 
/*  97 */     for (Map.Entry entry : plugins.entrySet()) {
/*     */       boolean success;
/*     */       try { 
					success = ((Predicate)entry.getValue()).apply(python);
/*     */       } catch (Exception e) {
/* 102 */         LOG.log(Level.SEVERE, "Plugin Main through an exception.", e);
/* 103 */       }

				continue;

				/*if (!success) {
					LOG.severe("Plugin Main returned error for: " + (String)entry.getKey());
				}*/
/*     */     }
/*     */ 
/* 111 */     python.set("__name__", "__main__");
/*     */ 
/* 113 */     python.set("__file__", scriptfilename);
			  python.set("robot", object);
/*     */     try
/*     */     {
/* 116 */       python.execfile(scriptfilename);
/*     */     } catch (PyException e) {
</verbatim>
=Extendting=  see 9.4 P223. Jython for Java Programmers.

-- Main.GangweiLi - 29 Oct 2012


*pprint*
pretty print is better than print has more control and smart

-- Main.GangweiLi - 02 Jul 2013


怎样在python 中添加路径？

-- Main.GegeZhang - 19 Jul 2013


python 中怎样实现程序复用，我想很多文件人家都已经写好了，？？


-- Main.GegeZhang - 16 Aug 2013


安装python子包

目录 到某个目录下： 首先是 D: 然后是 cd /d D:\Program Files (x86)\imageAirport

然后是 python setup.py  install


-- Main.GegeZhang - 11 Jan 2014


python 逐层构成： list->array->matrix

-- Main.GegeZhang - 14 Jan 2014


*对于集合运算支持*
python 有一个专门的 set 与frozenset类型来进行集合运算，本质是通过哈希作为基础来实现的。例如交并补差还对称差集等等，都是可以计算的。既然有了这样的数据结构来支持这样的运算，对于blender,以及GIMP中图形的交并补差也就容易了很多了。首先是顶点交并补差，然后是线最后是面。

-- Main.GangweiLi - 02 Apr 2014


*如果用python来写后台程序* 可以参考 ndk-gdb.py 中的background Running. 其实写起来很容易，就是in,out,err的重定向问题。可以线程Thread或者subprocess.communicate等待退出并读取输出。



*GIL* 这里有两篇文章写的不错，
http://fosschef.com/2011/02/python-3-2-and-a-better-gil/
http://zhuoqiang.me/python-thread-gil-and-ctypes.html

欲练神功，挥刀自宫，
就算自宫，也未必成功，
不用自宫，也一样能成功！

这三句用到这里简值太精典了，由于GIL限制多线程，要解决这个问题就必须自宫了，但是２０多年的发展有太多的库依赖此，也就是就算自宫，也未必成功，但不是没有办法了，直接利用c扩展来做，也直接解决了这个问题，把多线程的东东都放在C语言里，并用ctype来引用就行了。也就是解决问题的思路了。

python的缺点，那就是对多线程以及效率本身不高，但是结构清晰简单。Go语言天生对并行应该支持的非常好。但是一些新的编程范式支持还不错，并且是除了perl之外库非常的语言了。

`python 与 asm <http://wdicc.com/asm-and-python/>`
非常生动解读了，各个层级的执行效果，为了通用性，人们写出各种各样的执行框架，其实所谓的ABI就是汇编二进制指令之间的复用机制。所谓的dll,以及elf,各个段的机制，其实与代码层面的机制是一样的。并且在elf为了尽可能节省空间，把程序所有重复的字符串symbol都直接全用符号表来压缩，当然如果小于指令地址长度的符号就没有意思了，一个
地址要32位是4字节，64就是8字节来表示的。所以就看你理解的深入程度，深入才能浅出。



*pygments* 支持各个应用平台，wiki,web,html以及latex,console等等。这样非常方便配色，尤其是代码与log的分析的时候就非常的方便了。为了各种利用生成语法文件是非常的方便。

与自己写一个语法文件一样，其实也是一个词法与语法分析，然后给出配色，并且我们还可以利用语法树直接做一些别的操作，因为它已经支持大部分的语言了。可以省去自己很大一部分时间。可以只加入一个hook就可以了。

python 非常适合做一个interface语言，在于它的简单与精练。然后是各种场景的应用，现在感觉python是可以与perl有一拼了。各种各样库非常全。以后的编程可能都会多层编程同时存在的问题。用来解决灵活性与效率的问题。


`LINQ in Python <http://sayspy.blogspot.com.au/2006/02/why-python-doesnt-need-something-like.html>`_ 

http://www.cnblogs.com/cython/articles/2169009.html
http://blog.csdn.net/largetalk/article/details/6905378
http://blog.csdn.net/largetalk/article/details/6905378
http://wklken.me/posts/2013/08/20/python-extra-itertools.html
http://stackoverflow.com/questions/5695208/group-list-by-values
并且数学上排列组合都实现了。
原来都是为实现http://blog.jobbole.com/66097/ 无穷序列，随机过程，递归关系，组合结构。 都是源于yield.
http://radimrehurek.com/2014/03/data-streaming-in-python-generators-iterators-iterables/

这种懒惰性求值，都是利用不yield这种方式产生，并且具有不回退性，那就不能求长度等操作了。每两次的调用不是同一个值了。
这个直接利用语言的高阶特性会非常的简单，例如列表推导，以及filter,map,reduce再加上lambda 的使用，以及sorted 再加上itetools中groupby,

另一大块那就是vector化的索引计算。其实就相当于数据组的sql语言了。
