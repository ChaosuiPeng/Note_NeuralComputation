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
**- shape(a)**

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

**- normalize(X, norm='l2', \*, axis=1, copy=True, return_norm=False)**
from sklearn
Scale input vectors individually to unit norm (vector length).
Returns Normalized input X

**- plt的一些用法**
```python
# we plot the cost w.r.t. the iteration number
plt.plot(idx_gd, err_gd, color="red", linewidth=2.5, linestyle="-", label="gradient descent") #  gradient descent
plt.plot(idx_momentum, err_momentum, color="blue", linewidth=2.5, linestyle="-", label="momentum") # momentum
plt.legend(loc='upper right', prop={'size': 12})
plt.title('comparison between gradient descent and momentum')
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.grid()
plt.show()
```
![1684337354651](https://github.com/ChaosuiPeng/Artificial-Intelligence-and-Machine-Learning/assets/39878006/4b053084-577f-4a37-8818-6af4b5e99141)


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
```python
# 载入数据
from sklearn.datasets import load_boston
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    boston_data = load_boston()

data = boston_data['data']
x_input = data  # a data matrix
y_target = boston_data['target'] # a vector for all outputs
```

We can import the dataset and preprocess it as follows. Note we add a feature of $1$ to `x_input` to get a n x (d+1) matrix `x_in`
```python
# add a feature 1 to the dataset, then we do not need to consider the bias and weight separately
# 对506-by-13的x_input array，统一在第一列前插一列1.
# 生成506-by-1的全1 array。concatenate到x_input左边
# x_in = np.concatenate([np.ones([np.shape(x_input)[0], 1]), x_input], axis=1)
num_row = np.shape(x_input)[0]
x_1 = np.ones( [num_row, 1] )
x_in = np.concatenate([x_1, x_input], axis=1)

# we normalize the data so that each has regularity
# 正则化
x_in = preprocessing.normalize(x_in)
```

## Linear Model

A linear regression model in one variable has the following form 
$$
f(x)=\mathbf{w}^\top \mathbf{x}.
$$
The following function computes the output of the linear model on a data matrix of size n x (d+1).

```python
def linearmat_2(w, X):
    '''
    a vectorization of linearmat_1 in lab 2.
    Input: w is a weight parameter (including the bias), and X is a data matrix (n x (d+1)) (including the feature)
    Output: a vector containing the predictions of linear models
    '''
    return np.dot(X, w)
```

## Cost Function

We defined the following `mean square error` function for a linear regression problem using the square loss:

$$
C(\mathbf{y}, \mathbf{t}) = \frac{1}{2n}(\mathbf{y}-\mathbf{t})^\top (\mathbf{y}-\mathbf{t}).
$$

The python implementation is as follows

```python
def cost(w, X, y):
    '''
    Evaluate the cost function in a vectorized manner for 
    inputs `X` and outputs `y`, at weights `w`.
    '''
    residual = y - linearmat_2(w, X)  # get the residual
    err = np.dot(residual, residual) / (2 * len(y)) # compute the error
    
    return err
```

## Gradient Computation

Our methods require to use the gradient of the `cost` function. As discussed in the previous lecture, the gradient can be computed by

$$\nabla C(\mathbf{w}) =\frac{1}{n}X^\top\big(X\mathbf{w}-\mathbf{y}\big)$$

In the following, we present the python implementation on the gradient computation

```python
# Vectorized gradient function
def gradfn(weights, X, y):
    '''
    Given `weights` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,d+1) of input features including the feature $1$
          `y` - target y values
    Return gradient of each weight evaluated at the current value
    '''

    y_pred = np.dot(X, weights)
    error = y_pred - y
    return np.dot(X.T, error) / len(y)
```

## Gradient Descent

Gradient Descent iteratively updates the model by moving along the negative direction

$$\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)} - \eta\nabla C(\mathbf{w}^{(t)}),$$ 

where $\eta$ is a learning rate and $\nabla C(w^{(t)})$ is the gradient evaluated at current parameter value $\mathbf{w}^{(t)}$. In the following, we give the python implementation of the gradient descent on the linear regression problem. Here, we use `idx_res` to store the indices of iterations where we have computed the cost, and use `err_res` to store the cost of models at these iterations. These will be used to plot how the `cost` will behave `versus iteration` number.

```python
def solve_via_gradient_descent(X, y, print_every=100,
                               niter=2000, eta=1):
    '''
    Solves for linear regression weights with gradient descent.
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
          `print_every` - we report performance every 'print_every' iterations
          `niter` - the number of iterates allowed
          `eta` - learning rate
    
    Return 
        `w` - weights after `niter` iterations
        `idx_res` - the indices of iterations where we compute the cost
        `err_res` - the cost at iterations indicated by idx_res
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    idx_res = []
    err_res = []
    for k in range(niter):
        # compute the gradient
        dw = gradfn(w, X, y)
        # gradient descent
        w = w - eta * dw
        # we report the progress every print_every iterations
        # note the operator % calculates the remainder of dividing two values
        if k % print_every == print_every - 1:
            t_cost = cost(w, X, y)
            print('error after %d iteration: %s' % (k+1, t_cost))
            idx_res.append(k)
            err_res.append(t_cost)
    return w, idx_res, err_res
```

Now we apply **gradient descent** to solve the **Boston House Price** prediction problem, and get the weight `w_gd`, the indices `idx_gd` and the errors 'err_gd' on these indices 

```python
w_gd, idx_gd, err_gd = solve_via_gradient_descent( X=x_in, y=y_target)
```

## Momentum

Momentum introduces a variable `velocity` to store the historical information of the gradients. At each iteration, it updates `velocity` as a factor of the current `velocity` minus the `learning rate` times the `current gradient`

$$\mathbf{v}^{(t+1)} = \alpha\mathbf{v}^{(t)}-\eta\nabla C(\mathbf{w}^{(t)}),$$ 

where $\eta$ is a learning rate, $\alpha\in(0,1)$ is a parameter and $\nabla C(w^{(t)}$ is the gradient evaluated at current parameter value $\mathbf{w}^{(t)}$.
Then, we update the next iterate as 

$$\mathbf{w}^{(t+1)}=\mathbf{w}^{(t)}+\mathbf{v}^{(t+1)}.$$

In the following, we request you to finish the following implementation of the `momentum` on the linear regression problem.

```python
def solve_via_momentum(X, y, print_every=100,
                               niter=2000, eta=1, alpha=0.8):
    '''
    Solves for linear regression weights with momentum.
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
          `print_every` - we report performance every 'print_every' iterations
          `niter` - the number of iterates allowed
          `eta` - learning rate
          `alpha` - determines the influence of past gradients on the current update

    Return 
        `w` - weights after `niter` iterations
        `idx_res` - the indices of iterations where we compute the cost
        `err_res` - the cost at iterations
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    v = np.zeros([D])
    idx_res = []
    err_res = []
    for k in range(niter):
        # TODO: Insert your code to update w by momentum
        v = alpha * v - eta * gradfn(w, X, y)
        w = w + v
    
        if k % print_every == print_every - 1:
            t_cost = cost(w, X, y)
            print('error after %d iteration: %s' % (k+1, t_cost))
            idx_res.append(k)
            err_res.append(t_cost)
    return w, idx_res, err_res
```

Now we apply **momentum** to solve the **Boston House Price** prediction problem.
```python
w_momentum, idx_momentum, err_momentum = solve_via_momentum( X=x_in, y=y_target)
```

### Comparison between Gradient Descent and Gradient Descent with Momentum

We can now compare the behavie of Gradient Descent and Gradient Descent with Momentum. In particular, we will show how the `cost` of models found by the algorithm at different iterations would behave with respect to the iteration number.

```python
# we plot the cost w.r.t. the iteration number
plt.plot(idx_gd, err_gd, color="red", linewidth=2.5, linestyle="-", label="gradient descent") #  gradient descent
plt.plot(idx_momentum, err_momentum, color="blue", linewidth=2.5, linestyle="-", label="momentum") # momentum
plt.legend(loc='upper right', prop={'size': 12})
plt.title('comparison between gradient descent and momentum')
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.grid()
plt.show()
```

![1685337076455](https://github.com/ChaosuiPeng/Artificial-Intelligence-and-Machine-Learning/assets/39878006/5ed0727b-03d5-425b-93e1-c0bb415a932e)

As we expected, **gradient descent with momentum** is much faster than the **gradient descent**. This shows the benefit of using velocity to store historical gradient information for accelerating the algorithm.


## Nesterov Momentum

Anoter algorithm which can acclerate the training speed of gradient descent is the **Nesterov Momentum**. Analogous to **Momentum**, Nesterov Momentum also introduces a variable `velocity` to store the historical information of the gradients. The difference is that it first uses the current velocity to build a `looking ahead` point. Then the gradient computation is performed at the  `looking ahead` point. The `looking ahead` point may contain more information than the current point. Therefore, the gradient at `looking ahead` point may be more precise than the `current gradient`.
The update equation is as follows

$$\mathbf{w}^{\text{(ahead)}}=\mathbf{w}^{(t)}+\alpha\mathbf{v}^{(t)}$$
$$\mathbf{v}^{(t+1)} = \alpha\mathbf{v}^{(t)}-\eta\nabla C(\mathbf{w}^{(\text{ahead})}),$$ 

where $\eta$ is a learning rate and $\alpha\in(0,1)$ is a parameter.
Then, we update the next iterate as 

$$\mathbf{w}^{(t+1)}=\mathbf{w}^{(t)}+\mathbf{v}^{(t+1)}.$$

In the following, we request you to finish the following implementation of the `Nesterov Momentum` on the linear regression problem.

```python
def solve_via_nag(X, y, print_every=100,
                               niter=2000, eta=0.5, alpha=0.8):
    '''
    Solves for linear regression weights with nesterov momentum.
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
          `print_every` - we report performance every 'print_every' iterations
          `niter` - the number of iterates allowed
          `eta` - learning rate
          `alpha` - determines the influence of past gradients on the current update

    Return 
        `w` - weights after `niter` iterations
        `idx_res` - the indices of iterations where we compute the cost
        `err_res` - the cost at iterations
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    v = np.zeros([D])
    idx_res = []
    err_res = []
    for k in range(niter):
        # TODO: Insert your code to update w by nesterov momentum
        w_ahead = w + alpha * v
        v = alpha * v - eta * gradfn(w_ahead, X, y)
        w = w + v              
        
        if k % print_every == print_every - 1:
            t_cost = cost(w, X, y)
            print('error after %d iteration: %s' % (k+1, t_cost))
            idx_res.append(k)
            err_res.append(t_cost)
    return w, idx_res, err_res
```

Now we apply nesterov momentum to solve the Boston House Price prediction problem.

```python
w_nag, idx_nag, err_nag = solve_via_nag( X=x_in, y=y_target)
```


### Comparison between Gradient Descent and Nesterov Momentum

We can now compare the behavie of Gradient Descent and Nesterov Momentum. In particular, we will show how the `cost` of models found by the algorithm at different iterations would behave with respect to the iteration number.

```python
plt.plot(idx_gd, err_gd, color="red", linewidth=2.5, linestyle="-", label="gradient descent")
plt.plot(idx_nag, err_nag, color="blue", linewidth=2.5, linestyle="-", label="nesterov momentum")
plt.legend(loc='upper right', prop={'size': 12})
plt.title('comparison between gradient descent and momentum')
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.grid()
plt.show()
```

![1685337234373](https://github.com/ChaosuiPeng/Artificial-Intelligence-and-Machine-Learning/assets/39878006/055b55a5-393a-47c2-9fed-aba26c62e27f)



## Minibatch Gradient Descent

The optimization problem in ML often has a **sum** structure in the sense
$$
C(\mathbf{w})=\frac{1}{n}\sum_{i=1}^nC_i(\mathbf{w}),
$$
where $C_i(\mathbf{w})$ is the loss of the model $\mathbf{w}$ on the $i$-th example. In our Boston House Price prediction problem, $C_i$ takes the form $C_i(\mathbf{w})=\frac{1}{2}(\mathbf{w}^\top\mathbf{x}^{(i)}-y^{(i)})^2$.

Gradient descent requires to go through all training examples to compute a single gradient, which may be time consuming if the sample size is large. Minibatch gradient descent improves the efficiency by using a subset of training examples to build an **approximate** gradient. At each iteration, it first randomly draws a set $I\subset\{1,2,\ldots,n\}$ of size $s$, where we often call $s$ the minibatch size. Then it builds an approximate gradient by

$$
\nabla^I(\mathbf{w}^{(t)})=\frac{1}{s}\sum_{i\in I}\nabla C_i(\mathbf{w}^{(t)})
$$

Now, it updates the model by

$$
\mathbf{w}^{(t+1)}=\mathbf{w}^{(t)}-\eta_t\nabla^I(\mathbf{w}^{(t)}).
$$ 

It is recommended to use $s\in[20,100]$. Depending on different $s$, minibatch gradient descent recovers several algorithms
\begin{align*}
  s<n \;&\Rightarrow\;\text{Minibatch gradient descent}\\
  s=1 \;&\Rightarrow\;\text{Stochastic gradient descent} \\
  s=n \;&\Rightarrow\;\text{Batch gradient descent}
\end{align*}
In the following, we request you to finish the following implementation of the `minibatch gradient descent` on the linear regression problem. To search a subset of $\{1,2,\ldots,n\}$, we recommend you to use the function `random.sample`. The synatx is `random.sample(sequence, k)`, which returns $k$ length new list of elements chosen from the `sequence`. More details can be found  [here](https://www.geeksforgeeks.org/python-random-sample-function/)

```python
def solve_via_minibatch(X, y, print_every=100,
                               niter=5000, eta=0.2, batch_size=50):
    '''
    Solves for linear regression weights with nesterov momentum.
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
          `print_every` - we report performance every 'print_every' iterations
          `niter` - the number of iterates allowed
          `eta` - learning rate
          `batch_size` - the size of minibatch
    Return 
        `w` - weights after `niter` iterations
        `idx_res` - the indices of iterations where we compute the cost
        `err_res` - the cost at iterations
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    idx_res = []
    err_res = []
    tset = list(range(N))
    for k in range(niter):
        # TODO: Insert your code to update w by minibatch gradient descent
        idx = random.sample(tset, batch_size)
        #sample batch of data
        sample_X = X[idx, :]
        sample_y = y[idx]
        w = w - eta * gradfn(w, sample_X, sample_y)
        
        
        if k % print_every == print_every - 1:
            t_cost = cost(w, X, y)
            print('error after %d iteration: %s' % (k+1, t_cost))
            idx_res.append(k)
            err_res.append(t_cost)
    return w, idx_res, err_res
```
Now we apply minibatch gradient descent to solve the Boston House Price prediction problem.

```python
w_batch, idx_batch, err_batch = solve_via_minibatch( X=x_in, y=y_target)
```

## Adaptive Gradient Descent

Stochastic gradient descent uses the same learning rates for all the features. This can be problematic if there are some sparse and predictive features. The underlying reason is that there are few examples with non-zero values for the sparse features, and it is often the case that SGD will choose an example with a zero for the sparse feature. Then it would not update the corresponding coordinate in this case. This motivates the need to slow down the update of some coordinates if there is already a frequent update on that coordinate, and accelerate the update if there are few updates on that coordinate.

The key idea of `Adagrad` is to introduce a vector $\mathbf{r}$ to store the accumulated gradient norm square. We initialize $\mathbf{r}^{(0)}=0$ and update

$$ 
\mathbf{r}^{(t+1)}=\mathbf{r}^{(t)}+\hat{\mathbf{g}}^{(t)}\odot\hat{\mathbf{g}}^{(t)},
$$

where $\hat{\mathbf{g}}^{(t)}$ can be a stochastic gradient built based on a selected example or a minibatch of examples. Note $\mathbf{r}^{(t+1)}$ records the accumulated magnitude square of gradients in each coordinate up to the $t$-th iteration. In this way, the entries of $\mathbf{r}^{(t+1)}$ would be different. If ${r}_j^{(t+1)}$ is large, then this means that there are a lot of updates on the $j$-th coordinate. If ${r}_j^{(t+1)}$ is small, then this means that there are few updates on the $j$-th coordinate. The idea is to slow down the update on a coordinate if there are already many updates on that coordinate in the history, and speed up the update on a coordinate if there are few updates in the history. This can be achieved by dividing the parameter $\eta$ with $\sqrt{\mathbf{r}^{(t+1)}}$. That is

$$
\mathbf{w}^{(t+1)}\gets\mathbf{w}^{(t)}-\frac{\eta}{\delta+\sqrt{\mathbf{r}^{(t+1)}}}\odot \hat{\mathbf{g}}^{(t)} 
$$

In this way, we can have different learning rates on different coordinates. In the following, we request you to finish the following implementation of the `AdaGrad` on the linear regression problem. 

```python
def solve_via_adagrad(X, y, print_every=100,
                               niter=5000, eta=1, batch_size=50):
    '''
    Solves for linear regression weights with nesterov momentum.
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
          `print_every` - we report performance every 'print_every' iterations
          `niter` - the number of iterates allowed
          `eta` - learning rate
          `batch_size` - the size of minibatch
    Return 
        `w` - weights after `niter` iterations
        `idx_res` - the indices of iterations where we compute the cost
        `err_res` - the cost at iterations
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    idx_res = []
    err_res = []
    tset = list(range(N))
    gradients_sum = np.zeros([D]) # r
    delta = 1e-8
    for k in range(niter):
        # TODO: Insert your code to update w by Adagrad
        idx = random.sample(tset, batch_size)
        #sample batch of data
        sample_X = X[idx, :]
        sample_y = y[idx]
        
        dw = gradfn(w, sample_X, sample_y)
        gradients_sum = gradients_sum + dw ** 2
        w = w - eta * dw / (delta + np.sqrt(gradients_sum))
        
        if k % print_every == print_every - 1:
            t_cost = cost(w, X, y)
            print('error after %d iteration: %s' % (k+1, t_cost))
            idx_res.append(k)
            err_res.append(t_cost)
    return w, idx_res, err_res
```

Now we apply adaptive gradient descent to solve the Boston House Price prediction problem.

```python
w_adagrad, idx_adagrad, err_adagrad = solve_via_adagrad( X=x_in, y=y_target)
```

## Adam

For the `gradient descent with momentum`, we introduce a `velocity` to store the information of historical gradients to accerlate the optimization speed. For the `AdaGrad`/ `RMSProp`, we introduce an `accumulated gradient norm square` to store the information of historical updates on all coordinates, which allows us to have different learning rates at different features. 
The basic idea of Adam is to combine the idea of `gradient descent with momentum` and `AdaGrad`/ `RMSProp` together. It introduces both a `velocity` and an `accumulated gradient norm square`, both of which are initialized with the zero vector. Let $\hat{\mathbf{g}}^{(t)}$ be a stochastic gradient built based on either a selected example or a minibatch of examples. It first updates the velocity $\mathbf{s}$ by

$$
\mathbf{s}^{(t+1)}=\beta_1\mathbf{s}^{(t)}+(1-\beta_1)\hat{\mathbf{g}}^{(t)}.
$$

Then it updates the `accumulated gradient norm square` by

$$
\mathbf{r}^{(t+1)}=\beta_2\mathbf{r}^{(t)}+(1-\beta_2)\hat{\mathbf{g}}^{(t)}\odot\hat{\mathbf{g}}^{(t)}.
$$

After that we need to apply a bias correct to remove the bias in the above update

$$
\hat{\mathbf{s}}^{(t+1)}=\mathbf{s}^{(t+1)}/(1-\beta_1^{t+1}),\quad
\hat{\mathbf{r}}^{(t+1)}=\mathbf{r}^{(t+1)}/(1-\beta_2^{t+1}).
$$

We can now update the model by

$$
\mathbf{w}^{(t+1)}\gets\mathbf{w}^{(t)}-\frac{\eta}{\delta+\sqrt{\hat{\mathbf{r}}^{(t+1)}}}\odot \hat{\mathbf{s}}^{(t+1)}.
$$

As you can see, there are four parameters in Adam: $\eta, \delta, \beta_1, \beta_2$. Some recommended choices are

$$
\eta=0.001,\quad \beta_1=0.9,\quad\beta_2=0.999,\quad \delta=10^{-8}.
$$

In the following, we request you to finish the following implementation of the `Adam` on the linear regression problem.

```python
def solve_via_adam(X, y, print_every=100,
                               niter=5000, eta=0.1, beta1 = 0.9, beta2 = 0.999, batch_size=50):
    '''
    Solves for linear regression weights with nesterov momentum.
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
          `print_every` - we report performance every 'print_every' iterations
          `niter` - the number of iterates allowed
          `eta` - learning rate
          `beta1` - the parameter on updating velocity
          `beta2` - the parameter on updating the accumulated gradient norm square
          `batch_size` - the size of minibatch
    Return 
        `w` - weights after `niter` iterations
        `idx_res` - the indices of iterations where we compute the cost
        `err_res` - the cost at iterations
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    v = np.zeros([D])    
    idx_res = []
    err_res = []
    tset = list(range(N))
    gsquare = np.zeros([D])
    delta = 1e-8
    for k in range(niter):
        # TODO: Insert your code to update w by Adam
        idx = random.sample(tset, batch_size)
        #sample batch of data
        sample_X = X[idx, :]
        sample_y = y[idx]
        
        dw = gradfn(w, sample_X, sample_y)
        v = beta1 * v + (1 - beta1) * dw
        gsquare= beta2 * gsquare + (1 - beta2) * (dw ** 2)
        v_hat = v / (1 - beta1 ** (k + 1))
        gsquare_hat = gsquare / (1 - beta2 ** (k + 1))
        w = w - eta * v_hat / (np.sqrt(gsquare_hat) + delta)
        
        if k % print_every == print_every - 1:
            t_cost = cost(w, X, y)
            print('error after %d iteration: %s' % (k+1, t_cost))
            idx_res.append(k)
            err_res.append(t_cost)
    return w, idx_res, err_res
```

Now we apply Adam to solve the Boston House Price prediction problem.

```python
w_adam, idx_adam, err_adam = solve_via_adam( X=x_in, y=y_target)
```

### Comparison between Minibatch Gradient Descent, Adaptive Gradient Descent and Adam

We can now compare the behavie of Minibatch Gradient Descent, Adaptive Gradient Descent and Adam. In particular, we will show how the `cost` of models found by the algorithm at different iterations would behave with respect to the iteration number.

```python
plt.plot(idx_batch, err_batch, color="red", linewidth=2.5, linestyle="-", label="minibatch")
plt.plot(idx_adagrad, err_adagrad, color="blue", linewidth=2.5, linestyle="-", label="adagrad")
plt.plot(idx_adam, err_adam, color="green", linewidth=2.5, linestyle="-", label="adam")
plt.legend(loc='upper right', prop={'size': 12})
plt.title('comparison between minibatch gradient descent and adaptive gradient descent')
plt.xlabel("number of iterations")
plt.ylabel("cost")
plt.grid()
plt.show()      
```

![1685337264489](https://github.com/ChaosuiPeng/Artificial-Intelligence-and-Machine-Learning/assets/39878006/7a229c5c-1597-48cc-8fe7-f753fa79c45d)


As we see, Adam achieves the best performance. This demonstrates the effectiveness of combining the idea of momentum and Adagrad / RMSProp.
