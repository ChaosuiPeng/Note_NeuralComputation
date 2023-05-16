# Advanced Optimization Algorithms

### 快速过一遍GD、Momentum和Nesterov Momentum

GD的问题是：沿一个方向走得慢、而且可能被trapped在local optima。

提出Momentum：代入一个”verlocity"的概念，把历史的gradient按照不同的权重叠加（考虑历史轨迹），再去更新W。【好处是】同一方向前进的时候能越跑越快、而且可以概率get rid of flat region跳出local optima，而且对于来回波动（正负反复切换）的情况，也能有所改善、做到平稳前进（unlike GD，会oscillate a lot）。【问题是】因为考虑过多历史轨迹，经常overshoot。

再提出Nesterov Momentum：再考虑历史轨迹的情况下，再叠加一个预测轨迹，综合二者真正更新w。通过look ahead有效避免overshoot。

### 快速过一遍GD和SGD

GD的问题是：对每一个example都去计算gradient这个cost太大了。每一个iteration的computation cost是O(n)

1951的Stochastic Gradient Descent：每次只draw一个example with equal probability from dataset去compute gradient。【缺点】前进路线非常曲折。【优点】效果不错，而且computation cost变成了O(1)。【原理】sum structure告诉我们（缺了一些数学解释！！！！），if we consider all possible realization of it, we recover the true gradient.

![1684234305627(1)](https://github.com/ChaosuiPeng/Artificial-Intelligence-and-Machine-Learning/assets/39878006/c7766ded-df6f-4566-8d5e-57bd739436fe)

### 一些结合：SGD + Momentum, SGD + Nesterov Momentum
These can be extended to Mini-batch variant!

## Lab
In this exercise, we'll develop implementations of advanced optimization algorithms. As in Exercise 2, we will use the Boston Housing dataset and run some advanced optimization algorithms to solved the linear regression problems.

In this exercise, you will learn the following
* implement the `momentum` method
* implement the `Nesterov momentum` method
* implement the `minibatch gradient descent` method
* implement the `adaptive (stochastic) gradient descent` method


```python
import matplotlib
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt 
from sklearn import preprocessing   # for normalization
```

## Boston Housing Data

The Boston Housing data is one of the  datasets available in sklearn.
We can import the dataset and preprocess it as follows. Note we add a feature of $1$ to `x_input` to get a n x (d+1) matrix `x_in`

```python
from sklearn.datasets import load_boston
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    boston_data = load_boston()
data = boston_data['data']
x_input = data  # a data matrix
y_target = boston_data['target'] # a vector for all outputs
# add a feature 1 to the dataset, then we do not need to consider the bias and weight separately
x_in = np.concatenate([np.ones([np.shape(x_input)[0], 1]), x_input], axis=1)
# we normalize the data so that each has regularity
x_in = preprocessing.normalize(x_in)
```
