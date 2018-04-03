视觉推理
========

推理有两个方向，一个symbols方向，另一个connectionist方向。
clevr测试集 https://cs.stanford.edu/people/jcjohns/clevr/

一种是分解一个成一个个小问题的组合，通过回答这些小问题，最终回答这个大问题。
另一种是把图形与问题放在一起，然后扔到高层join训练 。

如果能做到实时的话，就相当于自动驾驶中，那个路况的查看过程了.


tbd-nets
========

#. 即然是module化的开发，网络的结构也都大同小异，如何区分各个模块的功能。
   如何功能块的特征，靠经验的理解设计+cost函数的设计。    

   .. image:: /Stage_4/VQA/tbd-net_modules.png

#. relate 模块是如何定义

#. Or/And 是直接通过取Max/min来得到。

   .. code-block:: python
      
      def forward(self,attn1,attn2):
          out = torch.max(attn1,attn2)
          return out

#. Attention是如何体现的。通过mask来体现的,也就是直接通过W*X 来实现的。把其高亮为红色。
   .. code-block:: python
     
      def forward(self,feats,attn):
          attended_feats = torch.mul(feats,attn.repeat(1,self.dim,1,1))
          out = F.relu(self.conv1(attended_feats))
          out = F.relu(self.conv2(out))
          out = F.sigmoid(self.conv3(out))
          return out
#. QueryModule 基本结构与attentionmodule是一样的。没有看出来是如何实现编码的功能的。

#. RelateModule,主要是通过扩大 receptive field by dilated convolutions. https://zhuanlan.zhihu.com/p/27285612 
#. QueryModule是一样的。 生成attention的最后一层sigmoid,不然都是relu.
   编码是用Seq2seq来生成的把任意图的问题，变成一个原基本词汇组成的语法表，然后tbd-nets对于每一个元语都有一个元module来实现的。
   .. code-block:: python
      
      for module_name in vocab['program_token_to_idx']:
          # skip list
          # hook module      
          if module_name = "scene":
             module = None;
          elif module_name = "intersect":
             module = modules.AndModule()
          elif module_name = "union": 
             module = modules.OrModule()
          #....
          self.function_modules[module_name]=module
          self.add_module(module_name,module)

      def foward():
          #...
          #.就是一个传统的状态机,构造神经网络
          for i in reversed(programs.data[n].cpu().numpy()):
               module_type = self.vocab['program_idx_to_token'][i]
               #skip list
               module = self.function_modules[module_type]
               if module_type = "scene":
                  # do something
               if 'equal' in module_type or module_type in {'intersecct','union','less_than','greater_than']:
#. Comparison  Encoding*encoding
                output = module(out,saved_output)
#. reasoning chain 是如何体现的
   是通过那个programs chain来来体现用seq2seq生成的。
#. 是如何训练的与定义cost函数的。
   是用交叉熵来做cost的。 

   .. code-block:: python
      optimizer = torch.optim.Adam(tbd_net.parameters(),le-04)
      xent_loss = torch.nn.CrossEntropyLoss()
      
      for batch in train_loader:
          _,_,feats,answers,programs = batch
          #...
          optimizer.zero_grad()
          outs = tbd_net(feat_var,programs_var)
          _,preds = outs.data.max(1)
          mapped_ans= map_ans(answers)        
          num_corrects +=
          loss = xent_loss(outs,Variable(mapped_ans))
          loss +=tbd_net.attention_sum * 2.5e-07
          loss.backward()
          optimizer.step()
          
          #...
          #validate
          #save something



MAC 机器推理
============

MAC 模仿的的计算体系架构,Control,Memory,Attention. 其实也就是相当遗传进化选择的过程。 

它的做法相当于tbd-nets中人为设计的原语直接用相关性+attention 的Control给取代了。
https://mp.weixin.qq.com/s/sbV5SL6fAGad5KlBoqUKFQ

并且将推理过程放到Key-Value Memory Networks中，这里key-value也都是机器学习来的。

.. image:: /Stage_4/VQA/mac_fw.png
.. image:: /Stage_4/VQA/mac_cell_structure.png


#. 输入的信息流是怎么的。
#. 如何与其他框架模块接口
#. 每一子块是如何计算的
#. 如何演化以及停止
#. 都有哪些超参数
#. 如何可视化
#. 有保证的健壮性。

实现推理的合心，如何量化一个符号，然后把推理过程，可以定义一个计算过程，如果这个过程是是可微分的话会更好。



NYU联合Google Brain提出结合工作记忆的视觉推理架构和数据集 COG
=============================================================

https://mp.weixin.qq.com/s/mPN3MXueeZImY2z0KBD6CA


它实现了一个可配置的Input Dataset. 相当于实现一个模拟器而己。 
#. 它解决了检测神经网络是不是凑答案的问题。
#. 通过控制输入的种类与数量，来研究网络的容量问题。 
#. 另外可以利用传感器与以及gameengine来建设大量的模拟器，就像NV的虚拟训练一样。
#. 通过虚拟化与数字化来解决各种物理规律的制约。
