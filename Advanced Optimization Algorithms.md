# Recap: 从Gradient Descent出发

### GD、Momentum和Nesterov Momentum（优化update的step size和方向）

GD：【优点】用的是true gradient，所以下降的方向是最理想的。【缺点】计算开销大，且沿一个方向走得慢，还可能被trapped在local optima。 

提出Momentum：代入一个”verlocity"的概念，把历史的gradient按照不同的权重叠加（考虑历史轨迹），再去更新W。【好处是】同一方向前进的时候能越跑越快、而且可以概率get rid of flat region跳出local optima，而且对于来回波动（正负反复切换）的情况，也能有所改善、做到平稳前进（unlike GD，会oscillate a lot）。【问题是】因为考虑过多历史轨迹，经常overshoot。

再提出Nesterov Momentum：再考虑历史轨迹的情况下，再叠加一个预测轨迹，综合二者真正更新w。通过looking ahead point有效避免overshooting behaviour。

### SD和SGD（优化update的计算效率）

GD的问题是：对每一个example都去计算gradient这个cost太大了。每一个iteration的computation cost是O(n)

1951的Stochastic Gradient Descent：每次只draw一个example with equal probability from dataset去compute gradient。【缺点】前进路线非常曲折，converge得很慢。【优点】最后的效果不错，而且computation cost变成了O(1)。【每次迭代只用single example的原理】sum structure告诉我们（缺了一些数学解释！！！！），if we consider all possible realization of it, we recover the true gradient.

![1684234305627(1)](https://github.com/ChaosuiPeng/Artificial-Intelligence-and-Machine-Learning/assets/39878006/c7766ded-df6f-4566-8d5e-57bd739436fe)

⚠ 一些结合：SGD + Momentum, SGD + Nesterov Momentum. These can be extended to Mini-batch variant!

### Adaptive Gradient Algorithm (AdaGrad), Root Mean Square Propagation (RMSProp) 和 Adaptive Moment Estimation (Adam)（对于多维数据的优化）
GD和SGD对于每一个feature都给了同样的update。

SGD由每次抽一个，这个过程会给dense feature更多的attention，which is midleading if sparse features are relevant。

AdaGrad【优点】对于不同的features (dense / sparse)给到不同的learning rates，且more updates means more decay（数学上解释一下为什么阿！！！）。【缺点】迭代越多，dense feature能affect的程度越小，model的更新变得aggressive。

RMSProp【优点】slow down AdaGrad中learning rate decay的问题。

Adam = Momentum + RMSProp

### 对比AdaGrad和Momentum
两者都可以accelerate updates in horizontal direction and slow down in vertical direction，but what's the difference?

![1684240258830](https://github.com/ChaosuiPeng/Artificial-Intelligence-and-Machine-Learning/assets/39878006/80e1924e-e7f5-48b7-a098-2502be02e0bd)

# Recap: Python
**- shape(array a)**

Return the shape of an array

returns tuple of ints.
```python
>>> np.shape(np.eye(3))
(3, 3)

>>> np.shape([[1, 2]])
(1, 2)

>>> np.shape([0])
(1,)

>>> np.shape(0)
()
```


**- ones(shape)**

Return a new array of given shape and type, filled with ones.

shape : int or sequence of ints. Shape of the new array, e.g., ``(2, 3)`` or ``2``.
```python
>>> np.ones(5)
array([1., 1., 1., 1., 1.])

>>> np.ones((5,), dtype=int)
array([1, 1, 1, 1, 1])

>>> np.ones((2, 1))
array([[1.],
      [1.]])

>>> s = (2,2)
>>> np.ones(s)
array([[1.,  1.],
       [1.,  1.]])
```


**- concatenate((a1, a2, ...), axis=0)**

The arrays must have the same shape.

Join a sequence of arrays along an existing axis. 
```python
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])

>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
    
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
    
>>> np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])
```




# Advanced Optimization Algorithms
In this exercise, we'll develop implementations of advanced optimization algorithms. We will use the Boston Housing dataset and run some advanced optimization algorithms to solved the linear regression problems.

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

The Boston Housing data is one of the datasets available in sklearn.
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
