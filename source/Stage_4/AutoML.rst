***********
AutoML
***********

* NAS  (Neural Architecture Search)
* Hyper-parameter optimization
* meta-learning
自动特征工程、自动调参,元学习。 
https://zhuanlan.zhihu.com/p/42924585

现在自动AutoML 还处于非常初级的水平，例如把CNN的分成小块，例如各种卷积，操作。并且在小样本上学习，然后放在大样本上学习。 

采用搜索算法有NAS,ENAS等等。这个一方面可以降低入门的设计难度，并且还能找到不错的模型。但是也挺费资源的。现在实验表明，这个相比于随机搜索的效果差不多。

而真正的automl 是能够program sythesis相结合的automl,每一个基本的program-code可以传统的代码库，相当于基本的API，并且找到合适的解之后，可以采用kernel的
融合，进一步压缩，因为这样的代码可以直接与汇编级别的指令直接对接。 可以直接实现不同层级的演化。 元学习进一步也就是program sythesis. 





