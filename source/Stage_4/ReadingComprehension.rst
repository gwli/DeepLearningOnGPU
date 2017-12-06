**************
机器阅读与理解
**************


机器阅读的类型
==============

#.  基本的完形填空式
#.  cloze style queries.
#.  推理
#.  与外界知识库的融合
#.  SQuAD数据集这类没有候选项且答案可能是多个词的machine comprehension问题。


R01_ 介绍了深度学习解决机器阅读理解任务的研究进展，提出三种内容表示的方法。
R02_ 介绍了WordEmbed的用意义，同时　R03_ 知乎解释当前Word Represntation的方法，
R04_ 是stanford 开放的训练好的库，GloVe: Global Vectors for Word Representation. 
R05_ 知乎专栏对这个有专门的一期。

.. figure:: /Stage_4/ReadingComprehension/Model1.jpg

   模型1 用BIRNN 来表征文档 

.. figure:: /Stage_4/ReadingComprehension/Model2.jpg
.. figure:: /Stage_4/ReadingComprehension/Model3.jpg
   
   模型三是在模型1的基础上改进来的，使用双向RNN来表征每个单词及其上下文的语言信。
   但是RNN对于太长的内容表征能力不足，会存在大量的信息丢失。


机器阅读理解问题形式:  人工合成问答，cloze style queries以及选择题等方式。，也是文档内容本身的表示以及问题本身的表示。对于问题本身的理解，也要有一个推理过程。对于更加复杂的事情还需要世界知识库融合的一个过程。


:math:`{D,Q,A,a}`  D->document,Q->Questions,AnswerSet,a->right answer.

三种模型
========

一维匹配模型
------------

.. figure:: /Stage_4/ReadingComprehension/1d_model.jpg

所以将这个结构为一维匹配模型，主要是计算问题Q和文章中单词序列的匹配过程形成了一维线性结构。
现在Attension Sum Reader, AS Reader, Stanford Attentive Reader. Gated-AttensionReader GA reader.
Attentive Reader,AMRRNN都是这种结构。

rnn可以理解为一个人看完了一段话，他可能只记得最后几个词说明的意思，但是如果你问他前面的信息，他就不能准确地回答，attention可以理解为，提问的信息只与之前看完的那段话中一部分关系密切，而其他部分关系不大，这个人就会将自己的注意力锁定在这部分信息中。

关键是匹配函数的定义 

#. AS Reader   :math:`F(D_i,Q)=D_iQ`
#. Attentive Reader  :math:`F(D_i,Q)=tanh(W_DDi,W_QQ)`
#. Stanford AR 采用又线性函数(Bilinear)
   :math:`F(D_i,Q)=D_iWQ`


注意力机制分为 Complex attention 与simple attention模型。

.. figure:: /Stage_4/ReadingComprehension/complex_attention.png
.. figure:: /Stage_4/ReadingComprehension/simple_attention.png

卷积注意力模型（convolutional attention-based encoder），用来确保每一步生成词的时候都可以聚焦到合适的输入上,

二维匹配模型
------------

.. figure:: /Stage_4/ReadingComprehension/2d_model.jpg

Consensus Attension,Attention-over-Attention AOA,Math-LSTM都属于这种模型。

推理模型
--------

一般通过加深网络的层数来模拟不断增加的推理步骤。

.. figure:: /Stage_4/ReadingComprehension/3d_model.jpg
.. figure:: /Stage_4/ReadingComprehension/3d_model_GA.jpg

   GA Reader 的推理过程
.. figure:: /Stage_4/ReadingComprehension/3d_model_IA.jpg
   
   IA Reader 的推理过程
.. figure:: /Stage_4/ReadingComprehension/3d_model_AMRNN.jpg

   AMRNN Reader 的推理过程

.. figure:: /Stage_4/ReadingComprehension/3d_model_MemoryNetworks.jpg

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
#. 中文语料库 R08_
#. English Gigaword数据集，该数据集包括了六大主流媒体机构的新闻文章，包括纽约时报和美联社，每篇文章都有清晰的内容和标题，并且内容被划分为段落。经过一些预处理之后，训练集包括5.5M篇新闻和236M单词

CNN,daily Mail 数据集生成方法，见 R06_ 中文解读见 R07_




当前的问题
==========

#. 更大难度的阅读理解数据集
#. 神经网络模型单一
#. 二维匹配模型需要做更深入的探索
#. 世界知识(World Knowledge)的引入
#. 发展更为完善的的推理机制,目前的推理还是停留在注意力焦点转移的机制。
#. 常用评价指标 R10_

方向跟踪
=========

http://harvardnlp.github.io/

自我理解的方向
==============

如何用神经网络表达一个知识库，并且随着知识的增长，如何扩展知识库。如何自动增加层数。
同时来了新的东东，如何实现与新旧知识之间的融合，也就不可避免添加适配层来进行适配训练融合。
如何用网络结构来实现迭代的符号化推导。而现在的神经网络是一个简单的强映射关系。

对于文档与内容的表示，一般用双向RNN来做。
`机器阅读理解中文章和问题的深度学习表示方法 <https://www.nytimes.com/2017/08/14/arts/design/google-how-ai-creates-new-music-and-new-artists-project-magenta.html?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=Deep%20Learning%20Weekly>`_
文章与问题的表示方法

自动文摘的功能
==============

Summarization. 
R09_ 介绍了两种方法 抽取式，与摘要式。 现在还没有很好的解决方式，由于信息的过载。人们迫切有一个工具用最短的时间了解最多的最有用的信息。 根据人们的提出问题，来查询相关的论文，然后自动形成综述。 但是目前还没有很的解决方法。

machine translation是最活跃的一个研究领域，seq2seq框架就是从该领域中提炼出来的，attention model也是借鉴于soft alignment，对于文本摘要这个问题来说，套用seq2seq只能解决headlines generation的问题，面对传统的single document summarization和multi document summarization任务便束手无策了，因为输入部分的规模远大于输出部分的话，seq2seq的效果不会很好，因此说abstractive summarization的研究还长路漫漫。不过这里可以将extractive和abstractive结合在一起来做，用extractive将一篇文档中最重要的一句话提取出来作为输入，套用seq2seq来做abstractive，本质上是一个paraphrase的任务，在工程中可以试一下这种思路。在后续的研究中也可以尝试将extractive和abstractive的思路结合在一起做文本摘要

难点在于自动评价的标准建模。

#. MRT+NHG  这个效果目前是比较好的。

#. R11_  教机器学习摘要
#. R12_ 分析常用的方法与派系。 
#. R13_ 摘要系统的实现
#. [R13]_ 是由saleforce 实现的基于增强学习实现的摘要，是目前的最高水平

用seq2seq的思路来解决文本摘要问题仍停留在short text的生成水平上，最多到paragraph level。原因也比较简单，rnn也好，gru、lstm也罢，终究都面临着一个长程依赖的问题，虽然说gru、lstm等技术用gate机制在一定程度上缓解了长程依赖和梯度消失、爆炸的问题，但终究文本过长的话，神经网络的深度就会随之变得非常深，训练起来难度就会随之增加。所以，这也是为什么document level或者说multi document level的abstractive式的摘要生成问题至今都是一个难以解决的问题。确实，short text的理解、表示在一定程度上有了很大的突破，也可以在工程上有不错的应用，比如机器翻译。但text变了之后，一篇很长的文章如何更加准确地理解和表示是一个非常难的问题，attention是一个不错的解决方案，在decoder的部分不需要考虑encoder的全部，只需确定需要注意的几个点就可以了，其实人在看一篇长文的时候也是这样一种机制，从某种角度上来讲，attention在decoder时提供了一种降维的手段，让model更能捕捉到关键的信息。

那些老的基于统计的方法，只是基于词频的方法的实用性极差，google 搜索提供的textsum 看起来还不错。
最新的水平 

语料库可以在 nlpcn.org 上找不到少。
github.com/thunlp 清华大学的自然语言处理

对于字幕分析
============

字幕下载器
----------

.. code-block:: bash
   
   # https://github.com/Diaoul/subliminal
   $ docker run --rm --name subliminal -v subliminal_cache:/usr/src/cache -v /tvshows:/tvshows -it diaoulael/subliminal download -l en /tvshows/The.Big.Bang.Theory.S05E18.HDTV.x264-LOL.mp4

分析流程
--------

#. 爬取数据 
   - 要利用代理与匿名网络来解决IP被封的问题
   - 利用多线程并行，来加快速度 

#. 数据的清洗
   - 词性，格式
   - 中英文的问题
    
#. 分析

   - 词汇量
   - 用到的人名，地名
   - 用到颜色的名字
   - 心情的的词语
   - 车，酒，等等 
   - 分词,摘要  
     
     * yaha 具有自动摘要的功能
     .. code-block:: bash
        
        pip install jieba,yaha
    
   - 如何判断经典句子

#. 词云的生成
   - wordCloud see github
   - 中文的支持主要是字体的设置 

#. 每一次更新之后

   - 最热的词汇排行有什么变化
   - 有哪些新增，新增有哪些变化。
   - 读一批书，帮助自己快速的了解一个行业
   - 把一次把一千本书，压缩一下。看看哪些是重复的句子。也就是所谓的金句。

#. 如果一次一千本，那么重复1000次，然后可可视化看看一些参数的变化。

   - 就像我们的阅读用同样的方法，重复读几遍，就会有不一样的认识，那我如果用机器来重复这样的事情呢。   
 
python 任意中文文本的生成 http://blog.csdn.net/fontthrone/article/details/72988956
  

Char-RNN
========

对于特别常的文档，生成效果也不好，例如ku,用了115M,然而生成的效果很差。

reference
=========

.. _R01: http://www.36dsj.com/archives/63037
.. _R02: https://yjango.gitbooks.io/superorganism/content/shen-ceng-xue-xi-ying-yong/zi-ran-yu-yan-chu-li/word-embedding.html
.. _R03: https://www.zhihu.com/question/32275069 
.. _R04: https://nlp.stanford.edu/projects/glove/ 
.. _R05: https://zhuanlan.zhihu.com/p/22577648
.. _R06: https://github.com/deepmind/rc-data
.. _R07: http://rsarxiv.github.io/2016/06/13/Teaching-Machines-to-Read-and-Comprehend-PaperWeekly/
.. _R08: http://hfl.iflytek.com/chinese-rc/
.. _R09: http://rsarxiv.github.io/tags/%E8%87%AA%E5%8A%A8%E6%96%87%E6%91%98/ 
.. _R10: http://www.jianshu.com/p/60deff0f64e1
.. _R11: http://rsarxiv.github.io/2016/06/25/%E6%95%99%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%91%98%E8%A6%81/ 
.. _R12: http://bj.bcebos.com/cips-upload/cwmt2012/ymy.pdf
.. _R13: http://rsarxiv.github.io/2016/06/10/Neural-Network-Based-Abstract-Generation-for-Opinions-and-Arguments-PaperWeekly/
.. _R14: https://arxiv.org/pdf/1705.04304.pdf
