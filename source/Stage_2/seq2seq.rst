seq2seq
=======

这是一个经典通用的matter,是一个翻译的过程，也是一个通过的编解过程。可以用于语言翻译，也可以用编程语言的汇编。
是不是可以用seq2seq来实现加密与解密。

Encoder-Decoder（编码-解码）是深度学习中非常常见的一个模型框架.

这个过程你可以套用任何的结构。 只要能这个理论来解释。

对于语言常用是LSTM,而对于图片大大家自然而然还是利用CNN来进行编码。
利用seq2seq的方式来实现结构化适配，目标是下层sensor的大小，输入当前的结构化信息。然后通过编码器来实现长短大小的转换。



基本模型
--------

.. graphviz::
   
   digraph G {
     Encoder->intermidiaState->Decoder;
   }

#. 要对字典中加入,<PAD>,<EOS>,<UNK>,<GO> 等标记符号。是不是可以冲过一些标点符号以及reserv关键字后效果会更好。
基本上都会加attention机制，其实是LSTM的扩展。


TVM+TensorFlow提高神经机器翻译性能
==================================

发现cuBLAS的批量matmul的性能问题，然后自身又采用TVM来cuda内核的更进一步优化。:w
https://mp.weixin.qq.com/s/HquT_mKm7x_rbDGz4Voqpw
https://github.com/Orion34C/tvm-batch-matmul-example


TVMlang
=======

http://tvmlang.org/


NMT
===

该论文论述的神经机器翻译（NMT）六大挑战：领域误匹配、训练数据的总量、生僻词、长句子、词对齐和束搜索（beam search）

LSTM 的引入解决了「长距离重新排序」问题，同时将 NMT 的主要难题变成了「固定长度向量（fixed-length vector）」问题：如图 1 所示，不管源句子的长度几何，这个神经网络都需要将其压缩成一个固定长度的向量，这会在解码过程中带来更大的复杂性和不确定性，尤其是当源句子很长时 [6]。

https://zhuanlan.zhihu.com/p/28701852

MCE
===

利用NMT的结构，建立一个external Memory,这样就解决记录长度的问题，并且每一次read/write 采用 attiontion机制来进行相当生成一个存取地址，以及做一次的内容聚合。

https://www.jiqizhixin.com/articles/2017-12-14-10

