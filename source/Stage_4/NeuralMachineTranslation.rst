********
机器翻译
********

[FPTG]_  解决了机器翻译中因为使用RNN产生的效率问题。
进一步实现从串行到并行的转换。 RNN 基本上都是 word by world.
而 [FPTG]_ 实现word->word 的并行。


ByteNet 采用并行村的卷积方法来取代RNN。

Saleforce 采用QRNN。

.. [FPTG] https://einstein.ai/research/non-autoregressive-neural-machine-translation
