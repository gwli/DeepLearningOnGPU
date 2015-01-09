深度学习
========

深度学与前些年神经网络的区别，那就是网络层数要比以前多了。这样的话功能就大大加强了。

网络一组layer来组成，layer 又由一组node来组成，层与层之间联接是由forword与backwrod连接组成。每一层都solver，以及module. 而这里blobs 就是node. 
foward 用来把input -> output同时计算出 loss cost,而backword 而是根据loss cost来计算梯度来提取下一步w,b，优画成什么程度为止呢，取决于loss cost. 这个就像牛顿求根法一样，到多少步停止。 并且w,b的初值如何选择。

每一个小的神经网络连接再加上决策树形成大的神经网络，不就实现了脑的功能。

Blobs 是由包着数据本身，而数据本身四维的数组(Num,Channels,Height and Width) 有点类似于CUDA的Thread分配。Blobs更像是CUDA Array 的机制。Blobs统一管理CPU与GPU的内存。(n,k,h,w).正好每个线程计算输入都是Blobs来得。输入传输，利用google proto来进传输。 `Google Protocol Buffer 的使用和原理 <http://www.ibm.com/developerworks/cn/linux/l-cn-gpb/>`_  google 这种机制就其名子一样，当做协议包来解析，就像网络包一样，每一个包不需要通用只要自己能认识自己就行了。就像网络协议一样，只规定了协议包结构。就自然解析就容易了。无非两种TLV,并且进一步利用编码本身实现压缩，并且元编程实现类的自动化定义与实现。并且还支持动态编译JIT,这也就意味着没有只要有源码就行了。

这个是把编码与元编程技术结合起来的。 

caffe 这个神经网络还可以snapshotting与resume,这个就非常的方便。
