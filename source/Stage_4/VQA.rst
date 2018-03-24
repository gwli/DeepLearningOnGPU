视觉推理
========

推理有两个方向，一个symbols方向，另一个connectionist方向。
clevr测试集 https://cs.stanford.edu/people/jcjohns/clevr/

一种是分解一个成一个个小问题的组合，通过回答这些小问题，最终回答这个大问题。
另一种是把图形与问题放在一起，然后扔到高层join训练 。


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

#. RelateModule,主要是通过扩大 receptive field by dilated convolutions.  
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
