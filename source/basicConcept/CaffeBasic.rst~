深度学习
========

深度学习与前些年神经网络的区别，那就是网络层数要比以前多了。这样的话功能就大大加强了。

网络一组layer来组成，layer 又由一组node来组成，层与层之间联接是由forword与backwrod连接组成。每一层都solver，以及module. 而这里blobs 就是node. 
foward 用来把input -> output同时计算出 loss cost,而backword 而是根据loss cost来计算梯度来提取下一步w,b，优化到什么程度为止呢，取决于error 要求和iter numbers.

每一个小的神经网络连接再加上形成大的神经网络，就实现了脑的功能。

Blobs 是由包着数据本身（什么意思？），而数据本身四维的数组(Num,Channels,Height and Width) 有点类似于CUDA的Thread分配。Blobs更像是CUDA Array 的机制。Blobs统一管理CPU与GPU的内存。(n,k,h,w)正好每个线程计算输入都是由Blobs来的。输入传输，利用google proto来进传输。 `Google Protocol Buffer 的使用和原理 <http://www.ibm.com/developerworks/cn/linux/l-cn-gpb/>`_  google 这种机制就其名字一样，（把什么当作？）当作协议包来解析，就像网络包一样，每一个包不需要通用，只要自己能认识自己就行了。就像网络协议一样，只规定了协议包结构。就自然解析就容易了。无非两种TLV,并且进一步利用编码本身实现压缩，并且元编程实现类的自动化定义与实现。并且还支持动态编译JIT,这也就意味着没有只要有源码就行了。

这个是把编码与元编程技术结合起来的。 

caffe 这个神经网络还可以snapshotting与resume,这个就非常的方便。

深度网络
========
本质是一层一层向上抽象的过程。同时一个由无序到有序的过程。但是过程的路径又不指一条，就像人每个人的思维过程也是一样的。所以每一层的每个神经网络的抽象可以不同。并且再加上一顺序就会有各种各样的结果。到这个过程中就像需要有一个像分词库一样的东东。如果只是简单组合就能得到一属性，还是排列才能进行判定， 排列与组合是不同层次的判别路线。

深学习就是在解决如抽象概念这个难题。
caffe流程
=========
#. 数据格式的转换，目前支持 lmdb,leveldb.
#. 定义 network
#. 定义 solver
#. training
#. testing

开发流程添加几个头文件与与源文件。然后实现setup,initial, resharp,forward,backword,然后就注册就行了。
