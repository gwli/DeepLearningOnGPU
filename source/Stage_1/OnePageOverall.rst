DL的数学模型
============


.. math:: 
   f =\sum{h(x)g(x)}

   y = f(WX+b)
   
   Loss = L(y-y')

   W_{i+1}=Wi - \lambda\frac{\partial{L}}{\partial{W_i}}


所谓的Loss函数，可以是两数之间的距离的度量，也可以两个序列之间的距离的度量。在正常情况下，我们希望两者之间的差距，越小越好。可以用cross-entroy或者协方差来表示。


ML 工作流程
============

.. image:: /Stage_1/BasicMachineLearningWorkflow.png

P  范数。

信息流
======

#.Data Training, validation, Testing 

.. graphviz::
   
   digraph G {
      Data->model->score->loss->model;
   }

#. 训练的基本问题
   
   - 过拟合与欠拟合 可以通过validation loss 来观察得到
   - 梯度消失


a naive dl  code
================

.. code-block:: python

   class pototype:
       def train/fit(train_images,train_labels):
           # build a model of images ->labels.
       def predict(image):
           #evalute the model on the image
           return class_label


   class NearestNeighbor:
       def __init__(self):
           pass

       def train(self,X,y):
           """ X is N x D where each row is an example. Y is 1-dimension of size N"""
           # the nearest neighbor classifier simply remembers all the training data
           self.Xtr = X
           self.ytr = y
       def predict(self,X):
           """ X is N x　D　where each row is an example we wish to predict label for"""
           #lets make sure that the output type meatches the input type
           Ypred = np.zeros9num_test,dtype=self.ytr.dtype)

           for i in xrange(num_test):
               #find the nearest training image to the i'th test image
               # using the L1 distance (sum of absolute vlaue differences)
               distances = np.sum(np.abs(self.Xtr-X[i,:]),axis=1)
               min_index = np.argmin(distances)
               Ypred[i]=self.ytr[min_index]



.. code-block:: python
   
   class MyModel:
       def predict(self,X):
           return X.dot(self.w)
       
       def fit(self,X,Y,eta,T):
            self.w = np.random.randn(X.shape[1])
            for _ in range(T):
                 y_hat = self.predict(X)
                 self.w = self.w - eta* X.T.dot(Y_hat -Y)
       def cost(self,X,Y):
           Y_hat = self.predict(X)
           return (Y_hat -Y).dot(Y_hat -Y)
    
     X, Y = load_my_dataset()
     model = MyModel()
     model.fit(X,Y)
大部分的机器学习都是在解决fit,predict 这两个函数是如何实现的。

流程
=====

#. Define a model
#. Compute a cost based on model output and training data
#. Minimize the cost using gradient descent wrt model parameters


非监督学习
==========

#. K-Means clustering
#. Guassian mixture models
#. Hidden Markow Models
#. Factor Analysis


#. Matrix factorization 
#. Quadratic discriminator/regressor
