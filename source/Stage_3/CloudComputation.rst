********
并行计算 
********

Introduction
=============

并行计算从适用范围分为微观的与宏观两个层面。微观的就是现在多核计算，主要是GPU、CPU以及DSP等。宏观的就是现在云计算，主要是google引领的类似于Hadoop的机制。 `计算能力需求评估 <ComputeCapacity>`_ 

如何提高并行速度，一个关键条件就是各个部分尽可能独立，没有依赖。其中一个方法是对于变量尽可能只写不改。对于函数尽可能顺序没有分支。如何分支考虑能否并行。分支是对流水线最大的杀手。流水线适用于顺序执行。流水线预存取，所以要进行分支预测。

#. Concurrency 
#. Load balancing
#. dependency 

并行计算，微观
===============

这个微观这个极别，有多种解决方案,

.. csv-table:: 

   Nvidia,  CUDA  ,
   IBM ,Power ,
   AMD , `AMD的CTM <http://gpgpu.org/static/s2007/slides/07-CTM-overview.pdf>`_   ,
   MS ,[[http://pcedu.pconline.com.cn/softnews/dongtai/0907/1714191_all.html微软的 Direct Compute]], 并且在win7的操作系统直接有GPU的API，并且会自动加速一些graphic的处理，包括视频，游戏等等。以及win7非常炫的3D界面。, 
   ARM , `Mali <http://zh.wikipedia.org/wiki/Mali_%28GPU%29>`_  ,
   open , `OPENCL <Work.OpenCL>`_  ,
   Intel , Phi 处理器 ,

当然还有一些都超级计算机架构例如 CRAY,PGP等等。其中Direct Compute,OPENCL 都是基于CUDA架构的。

 ` 区别 <http://hi.baidu.com/zhanghuisky365/item/147bdff520b74a49922af21b>`_   OPENCL is basing CUDA. OPENCL is compiled to PTX of CUDA. and then NVIDIA would convert the PTX to bin for GPU.

CUDA 是为计算本身设计，所以它的处理对象也是计算本身，因此要想对CUDA 有深入理解，首先要计算本身有深入的理解，尤其是矩阵的运算，比如矩阵的分块、分解、求逆以及矩阵的加乘运算。matlab来解决能不能问题。而并行计算是来解决快不快的问题。（这里好像联系不是很紧密）

通过对CUDA的学习对于并行计算有了全面的认识，这些年计算方法本身快速发展，整个编程环境也在发生着变化，从单核、单线程到多核多线程。特别是对于CUDA用到各个概念可以运用各个平台上（什么意思）。从指令级别的SIMD,再线程级别的SPMD,主要解决数据之间的依赖，还有各种硬件之间的速度差别，以及开发效率与维护效率。


从编程语言的发展看cuda编程
==========================

现在编程也开始慢慢分层结构化，并且扁平化，在尽可能解决复用性的同时又不丧失效率。（跟上面的有什么联系？）对于现在编程模式中如何快速在各个层次之间进行切换呢，就像CUDA的Thrust的结构一样。现在的编程语言可以轻松在各个层之间相互转换，就像TCP/IP的结构一样，指令级相当于物理层，.so层，以及命令层，对象层，逻辑算法层。每一层都有自己的描述语言，并且层与层之间可以相互转化。其实现在的编程语言正是没有各个层的混杂。一个关键那就是编译器的优化与编程体系的优化。(什么意思)

`CUDA <Work.CUDAProgrammingAndProfiling>`_ 的出现以及opengl,openCL的发生，更像是把硬件高层化，例如CUDA相当于直接要控制硬件了，例如opengl,就是汇编指令更高一层抽象，而CUDA则是把硬件的指令水平更加进一步提高了，可以直接针对硬件线程的操作。 

现在编程语言的发展都是走的大而全的路线，而不是分层合作的路线。每一家都在重复做着一些事情。CUDA提供了对于GPU的直接高层控制，例如可以C语言水平的直接控制。每一个原语都有对应的硬件单元。而OPENGL与DX都是某一领域的硬件语言。
而OPENCL则是考虑的硬件扩平台问题，而OPENMP与ＣＵＤＡ相对应解决ＣＰＵ的并行化。OPENACC 是用来解决语言扩平台，通过编译指令来实现类似于M4的方式这样产生CPU与GPU的代码，而C++ AMP则是直接针对高层OPENCL这种层API的编程。它的output直接是 OPENCL，当然可以直接支持CUDA。而theano这个更是直接从符号语言到CPU与GPU的产生，可以生成CPU可用的C代码，也可以生成GPU可以用CUDA代码，或者opencl代码。 那么是不是也可以直接从Haskell这样语言来产生CUDA代码。更好的那就是theano这种机制。来解决各种语言之间转换，并且保持可读性。

而另外一些语言要考虑的并行性与依赖关系。并行性例如CUDA对于线程三层模式，grid/block/thread三种模式。依赖的解决一方面依赖于手工解决，别一方面可以利用函数调用等依赖算法来自动管理。现在数据依赖主要是靠人来解决，例如CUDA就像你来实现内因分配方式然后通过数组下标的变化计算来解决并行，有点类于numpy与sql的模式。 Hadoop 采用的是map/reduce模型，这个模形解决了没有反馈一直往前走，对于反馈模型不太好解决了。而 `mpi <http://www.mcs.anl.gov/research/projects/mpi/>`_    , `pvm Parallel Virtual Machine <http://www.csm.ornl.gov/pvm/>`_   则是通过消息路由的机制透明的实现各个结点或者进程的同步。例如MPI的进程也CUDA一样是分组的，同样消息也分P2P与组播与广播等机制。

对于性能影响（这个题目好像不太好）
=====================================

数据传输，同步这种overhead会大大影响效率，例如CUDA发明了nvlink来取代PCIE，来得到更高传输速率，这也是为CUDA中在host memory 与device  memory （是什么）之间慢的原因，二者是通过PCIE接口传的，PCI接口的速度决定了传输速度，那是不是可以用网口或者光纤直接传输呢，对于集群之间也尽可能高速传输。
 the parallel coupute divided three layers.

.. csv-table:: 

   PHY layer ,  cluster of PC ,
   manage and comunication resource ,  ,
   compulate application , ,


#. `PVM编程指南 <http://www.longen.org/l-r/detaill-r/pvmprogamming.html>`_ 
#. `还驾驭不了4核？ 别人已模拟出百万核心上的并行 <http://www.csdn.net/article/2013-01-29/2814001-million-cores-parallel>`_ 
#. `Google Dremel 原理 - 如何能3秒分析1PB <http://www.yankay.com/google-dremel-rationale/>`_ 
#. `research of google <http://research.google.com/pubs/pub36632.html>`_ 

Evolution of The Architecture for CPU-GPU 
==========================================

从最初的CPU计算，到GPU的崛起，到未来CPU与GPU的融合。GPU一直饱受诟病传输速度一直又受限于PCI的慢速接口，现已经被解决了：

（下面说的是什么？）
#. 减少传输，例如使用CUDA的dynamic Parellel 
#. GPU Direct and  NVlink绕过PCI直接与GPU的高速互通，甚至CPU与GPU直接互联，这一点已经被NV与IBM联合开发的nvlink来实现了。
#. 另外ＣＰＵ与GPU直接共享内存与最后一级cache,现在已经开始实现了。例如Soc,Tegra 5,已经把GPU与ARM做在一起SOC了。并且现在已经有l4t的ubuntu系统。但是还是采用原有的内核调度方法。而在这一方面应该像微软的direct compute与win7学习。直接在内核直接直接CPU与GPU的调度。也就是把tk1中CUDA 内核直接与linux的kernel直接融合在一program  -
    
   .. graphviz::

      digraph evolution {
         rankdir=LR;
         "chip integrated" ->"holistic optimization" ->"Opportunistic optimization" ->"Tool/Emerging/Power/temparature";
      }
   

制约GPU的性能两个主要问题，传输速度与control flow divergence.
 
传输速度的问题已经有了很好的解决方案，直接绕开PCIe来通信的NVLink以及GPUDirect,另一个那就是Soc就像现在TK1直接CPU与GPU做在一起。其实这个概念也是AMD之前的APU的概念，只是AMD没有做好。

对于flow divergence的问题，现在还没有很好的方法。但是可以以下几种改进。

#. 通过算法本身改进以及通过编译器的优化来避免flow divergence.
#. Dynamic Warp Formation 或者 Large Warp Microchitecture and Two Level Scheduling.
#. 通过CPU配合来解决调度。另外对指令级SSE并行，以及thread level parallel等等，另外那就是剃除冗余。等等到了再进一步观察其演化。或者发一些文章。

Hadoop
======

#. `Apache Hadoop集群的安全性架构 <http://www.csdn.net/article/2013-01-28/2813973-Hadoop-Security-Architecture>`_ 
#. `大数据发展的5条趋势 <http://www.csdn.net/article/2012-11-08/2811632>`_ 

See also
========

#. `ownership of data in the cloud. <http://papers.ssrn.com/sol3/papers.cfm?abstract&#95;id&#61;1562461##>`_  You need first to contact legal, you cannot rely on what they say. Imagine it is a government that want to spy the telecom of others. There are so many “bad” potential business behind that (a company that provide security professional service and try to find how to breach your code to sell their services).

#. `apache hadoop <http://hadoop.apache.org/>`_  
#. ` Use Apache HBase when you need random, realtime read/write access to your Big Data. This project&#39;s goal is the hosting of very large tables -- billions of rows X millions of columns -- atop clusters of commodity hardware. Apache HBase is an open-source, distributed, versioned, column-oriented store modeled after Google&#39;s Bigtable: A Distributed Storage System for Structured Data by Chang et al. Just as Bigtable leverages the distributed data storage provided by the Google File System, Apache HBase provides Bigtable-like capabilities on top of Hadoop and HDFS. Features <http://hbase.apache.org/>`_  

#. `hadoop over cuda <http://www.cse.ust.hk/gpuqp/Mars.html>`_  
#. `hadoop over cuda on apache <http://wiki.apache.org/hadoop/CUDA&#37;20On&#37;20Hadoop>`_  
#. `openMP <http://blog.csdn.net/drzhouweiming/article/details/4093624>`_  多核，不仅仅是多线程就够了的
#. `通过 GCC 学习 OpenMP 框架 <http://www.ibm.com/developerworks/cn/aix/library/au-aix-openmp-framework/>`_  
#. `并行计算 youtube <http://www.youtube.com/watch?v&#61;zb49vDrOxgA>`_  
#. `函数式编程是一个倒退 <http://kb.cnblogs.com/page/154935/>`_ 

思考
======



*automation login the 1t* 
so that I accumulate the volume of the online store.

-- Main.GangweiLi - 04 Aug 2012


*CITRIX*  has become an new leading IT provider. it support the virtual computer. there is new technique revolution.  now the Minix architecture is getting the focus on the cloud compultation. the slim client machine is popular now.  And Now that I Find that the Business opportunity is different from the technique perspective.  the technique is not necessary to be so high to have a value of business. the business value is meeting the requirement of real requirement.
*personal+performance+function*  "distributive management -> centric delivery ->suffer from effective management -> transform to service*
the logical device is getting more and more popular in much area.  for computer architecture.
CISCO let the switch device recognize the package from which virtual OS. 
the apllication is analogized and grouped to different category. 
Cloud is basing on the virtual computer.
-- Main.GangweiLi - 24 Oct 2012


How can I makefile run parallel. 

-- Main.GangweiLi - 24 Mar 2013


*openacc* 对于已经有代码通过标定，让编译器自己来决定如何并行


-- Main.GangweiLi - 03 Aug 2013

