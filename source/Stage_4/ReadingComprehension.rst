##############
机器阅读与理解
##############

R01_ 介绍了深度学习解决机器阅读理解任务的研究进展，提出三种内容表示的方法。

.. figure:: /Stage_4/ReadingComprehension/Model1.png
   
   模型1 用BIRNN 来表征文档 

.. figure:: /Stage_4/ReadingComprehension/Model2.png
.. figure:: /Stage_4/ReadingComprehension/Model3.png
   
   模型三是在模型1的基础上改进来的，使用双向RNN来表征每个单词及其上下文的语言信。
   但是RNN对于太长的内容表征能力不足，会存在大量的信息丢失。


机器阅读理解问题形式:  人工合成问答，cloze style queries以及选择题等方式。，也是文档内容本身的表示以及问题本身的表示。对于问题本身的理解，也要有一个推理过程。对于更加复杂的事情还需要世界知识库融合的一个过程。

三种模型
========

一维匹配模型
------------

.. figure:: /Stage_4/ReadingComprehension/1d_model.png

所以将这个结构为一维匹配模型，主要是计算问题Q和文章中单词序列的匹配过程形成了一维线性结构。
现在Attension Sum Reader, AS Reader, Stanford Attentive Reader. Gated-AttensionReader GA reader.
Attentive Reader,AMRRNN都是这种结构。

关键是匹配函数的定义 

#. AS Reader   :math:`F(D_i,Q)=D_iQ`
#. Attentive Reader  :math:`F(D_i,Q)=tanh(W_DDi,W_QQ)`
#. Stanford AR 采用又线性函数(Bilinear)
   :math:`F(D_i,Q)=D_iWQ`

二维匹配模型
------------

.. figure:: /Stage_4/ReadingComprehension/2d_model.png

Consensus Attension,Attention-over-Attention AOA,Math-LSTM都属于这种模型。

推理模型
--------

一般通过加深网络的层数来模拟不断增加的推理步骤。

.. figure:: /Stage_4/ReadingComprehension/3d_model.png
.. figure:: /Stage_4/ReadingComprehension/3d_model_GA.png

   GA Reader 的推理过程
.. figure:: /Stage_4/ReadingComprehension/3d_model_IA.png
   
   IA Reader 的推理过程
.. figure:: /Stage_4/ReadingComprehension/3d_model_AMRNN.png

   AMRNN Reader 的推理过程

.. figure:: /Stage_4/ReadingComprehension/3d_model_MemoryNetworks.png

   记忆网络  的推理过程


其它模型
========

主要是EpiReader,EpiReader和动态实体表示模型(Dynamic Entity Representation) DER模型

EpiReader是目前机器阅读理解模型中效果最好的模型之一，其思路相当于使用AS Reader的模型先提供若干候选答案，然后再对候选答案用假设检验的验证方式再次确认来获得正确答案。假设检验采用了将候选答案替换掉问题中的PlaceHolder占位符，即假设某个候选答案就是正确答案，形成完整的问题句子，然后通过判断问题句和文章中每个句子多大程度上是语义蕴含(Entailment)的关系来做综合判断，找出经过检验最合理的候选答案作为正确答案。这从技术思路上其实是采用了多模型融合的思路，本质上和多Reader进行模型Ensemble起到了异曲同工的作用，可以将其归为多模型Ensemble的集成方案，但是其假设检验过程模型相对复杂，而效果相比模型集成来说也不占优势，实际使用中其实不如直接采取某个模型Ensemble的方案更实用。

DER模型在阅读理解时，首先将文章中同一实体在文章中不同的出现位置标记出来，每个位置提取这一实体及其一定窗口大小对应的上下文内容，用双向RNN对这段信息进行编码，每个位置的包含这个实体的片段都编码完成后，根据这些编码信息与问题的相似性计算这个实体不同语言片段的Attention信息，并根据Attention信息综合出整篇文章中这个实体不同上下文的总的表示，然后根据这个表示和问题的语义相近程度选出最可能是答案的那个实体。DER模型尽管看上去和一维匹配模型差异很大，其实两者并没有本质区别，一维匹配模型在最后步骤相同单词的Attention概率合并过程其实和DER的做法是类似的。

https://evolution.ai/#technology  已经做到实用阶段。
例如国内在做让机器人高考也是朝着方面走。

现在内容表达式方法有三种。 一种那就是把文档本身当做单词的序列来这样理解。 由于每一个单词的权重也是不一样的。这样也可以再计算出来语言来。

整体分为一维的方法，二维的方法，三维的方法。

二维的方法
==========

也就是把整个文档与问题之间做一个二维mapping矩阵，相当于每一个单词的对最终的权重也都是不一样的。


数据集
======

#. bAbi
#. CNN
#. Daily Mail
#. SQuAD 数据集


当前的问题
==========

#. 更大难度的阅读理解数据集
#. 神经网络模型单一
#. 二维匹配模型需要做更深入的探索
#. 世界知识(World Knowledge)的引入
#. 发展更为完善的的推理机制,目前的推理还是停留在注意力焦点转移的机制。

自我理解的方向
==============

如何用神经网络表达一个知识库，并且随着知识的增长，如何扩展知识库。如何自动增加层数。
同时来了新的东东，如何实现与新旧知识之间的融合，也就不可避免添加适配层来进行适配训练融合。
如何用网络结构来实现迭代的符号化推导。而现在的神经网络是一个简单的强映射关系。

对于文档与内容的表示，一般用双向RNN来做。
`机器阅读理解中文章和问题的深度学习表示方法 <https://www.nytimes.com/2017/08/14/arts/design/google-how-ai-creates-new-music-and-new-artists-project-magenta.html?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly>`_

文章与问题的表示方法

reference
=========

.. _R01: http://www.36dsj.com/archives/63037
