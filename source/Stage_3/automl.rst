AutoML
======

谷歌出了 AutoML-cloud.

这样就可以自动优化了。

但是你需要决定的 Time and memory limits. 以及算法的搜索空间与算法优先级。

http://automl.github.io/auto-sklearn/stable/manual.html#manual

.. code-block:: python
 
    import autosklearn.classification
    automl = autosklearn.classification.AutoSklearnClassifier(
         include_estimator=['random_forest'],
         exclude_estimator=None,
         include_preprocessors=['no_preprocess'],
         exclude_preprocessors=None)
    automl.fit(x_train,y_train)
    predictions = automl.predict(x_test)
