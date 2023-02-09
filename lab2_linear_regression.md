# Linear Regression and Gradient Descent

In this exercise, we'll go through another example of linear regression from an implementation
perspective. We will use the `Boston Housing dataset`, and predict the median cost of a home
in an area of Boston. In this exercise, you will learn the following
* set up the linear regression problem using numpy
* show that vectorized code is faster
* produce scatter and line plots using Matplotlib
* solve the linear regression problem using the closed form solution
* solve the linear regression problem using gradient descent

We will use the two Python packages [NumPy](http://www.numpy.org/) and [Matplotlib](https://matplotlib.org/). NumPy is an open-source module that provides fast, precompiled numerical routines. To learn more about NumPy, you can [read this short tutorial](https://numpy.org/doc/stable/user/quickstart.html). Matplotlib is a 2D plotting library which can be used to produce [a wide range of plots](https://matplotlib.org/2.0.2/gallery.html), including histograms, power spectra, bar charts, errorcharts, and scatterplots. To learn more about Maptplotlib, you can [read this short tutorial](https://matplotlib.org/2.0.2/users/pyplot_tutorial.html). 

You should import these packages using the `import` statement. To call a function `X` from the NumPy module, you would normally have to write `NumPy.X()`. However, if you invoke NumPy functions many places in your code, this quickly becomes tedious. By adding `as np` after your `import` statement as shown below, you can instead write  `np.X()`, which is less verbose. 


```python
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing   # for normalization
```

## Boston Housing Data

The Boston Housing data is one of the "toy datasets" available in sklearn.
We can import and display the dataset description like this:

### Load the Data


```python
from sklearn.datasets import load_boston
boston_data = load_boston()
print(boston_data['DESCR'])
```

    .. _boston_dataset:
    
    Boston house prices dataset
    ---------------------------
    
    **Data Set Characteristics:**  
    
        :Number of Instances: 506 
    
        :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    
        :Attribute Information (in order):
            - CRIM     per capita crime rate by town
            - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
            - INDUS    proportion of non-retail business acres per town
            - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
            - NOX      nitric oxides concentration (parts per 10 million)
            - RM       average number of rooms per dwelling
            - AGE      proportion of owner-occupied units built prior to 1940
            - DIS      weighted distances to five Boston employment centres
            - RAD      index of accessibility to radial highways
            - TAX      full-value property-tax rate per $10,000
            - PTRATIO  pupil-teacher ratio by town
            - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
            - LSTAT    % lower status of the population
            - MEDV     Median value of owner-occupied homes in $1000's
    
        :Missing Attribute Values: None
    
        :Creator: Harrison, D. and Rubinfeld, D.L.
    
    This is a copy of UCI ML housing dataset.
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    
    
    This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.
    
    The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
    prices and the demand for clean air', J. Environ. Economics & Management,
    vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
    ...', Wiley, 1980.   N.B. Various transformations are used in the table on
    pages 244-261 of the latter.
    
    The Boston house-price data has been used in many machine learning papers that address regression
    problems.   
         
    .. topic:: References
    
       - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
       - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
    
    

To keep the example simple, we will only work with **two** features: `INDUS`
and `RM`. The explanations of these features are in the description above. 


```python
# take the boston data
data = boston_data['data']
# we will only work with two of the features: INDUS and RM
x_input = data[:, [2,5]]
y_target = boston_data['target']
```


```python
x_input
```




    array([[ 2.31 ,  6.575],
           [ 7.07 ,  6.421],
           [ 7.07 ,  7.185],
           ...,
           [11.93 ,  6.976],
           [11.93 ,  6.794],
           [11.93 ,  6.03 ]])




```python
# we normalize the data so that each has regularity
x_input = preprocessing.normalize(x_input)
```


```python
x_input
```




    array([[0.33146873, 0.9434662 ],
           [0.74026689, 0.67231312],
           [0.70137948, 0.71278806],
           ...,
           [0.86324829, 0.50477955],
           [0.86896836, 0.49486765],
           [0.89247369, 0.45109944]])



In the final step, we scale input vectors individually to unit norm (vector length). The goal of normalization is to change the values in the dataset to a common scale, which is key to get more robust results. There are several different noramlization strategies. More details can be found [here](https://scikit-learn.org/stable/modules/preprocessing.html)


### Visualization

Just to give us an intuition of how these two features INDUS and RM
affect housing prices, lets visualize the feature interactions.
As expected, the more "industrial" a neighbourhood is, the lower the
housing prices. The more rooms houses in a neighbourhood have, the
higher the median housing price.

We will now visualise the dataset using a scatter plot using the [`scatter`](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.scatter.html) function in the matplotlib.pyplot module. This function can be called as follows:

    plt.scatter( x , y )
    
The two arguments `x` and `y` are the input data. We can label the `x`- and `y`-axes as follows: 

    plt.xlabel("x_label_here")
    plt.ylabel("y_label_here") 
    
The function scatter has many additional arguments, as [described in the reference manual]([`scatter`](https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.scatter.html). 

Now, you should make a scatter plot of the `price` versus `Industrialness`. You should label the x- and y-axes by "Industrialness" and "Med House Price". Hint: Remember to include `plt.show()` at the end; otherwise, the scatter plot is not shown.
In a similar way, you can make a scatter plot of the `price` versus `Avg Num Rooms`. 


```python
# Individual plots for the two features:
plt.title('Industrialness vs Med House Price')
plt.scatter(x_input[:, 0], y_target)
plt.xlabel('Industrialness')
plt.ylabel('Med House Price')
plt.show()

plt.title('Avg Num Rooms vs Med House Price')
plt.scatter(x_input[:, 1], y_target)
plt.xlabel('Avg Num Rooms')
plt.ylabel('Med House Price')
plt.show()
```


   
![output_11_0](https://user-images.githubusercontent.com/39878006/217124526-173d57bf-155b-4807-9aa6-d3e50030405d.png)



    
![output_11_1](https://user-images.githubusercontent.com/39878006/217124573-a321d1b5-2f3b-4182-925e-1cc9de623a2e.png)
  


## Defining a Linear Regression Model

A linear regression model in our problem has the following form 
$f(x)=\mathbf{w}^\top \mathbf{x}+b=w_{1}x_{1}+w_{2}x_{2}+b,$
where $\mathbf{x}$ is the input, $\mathbf{w}$ is called the weight and $b$ is known as the bias.
The purpose of generating such a model is to predict an output (price) given an input (Industrialness, Avg Num Rooms). Given the model parameter $\mathbf{w},b$ and the new input $x$, the output predicted by our simple model is $\mathbf{w}^\top \mathbf{x}+b$. We will define a function named `linearmodel(x,w)` which represents this model. The function takes three arguments, the weight parameter $\mathbf{w}$, the bias parameter $b$ and the input $\mathbf{x}$, and it returns the predicted output $\mathbf{w}^\top \mathbf{x}+b$. 

A function is a block of organized, reusable code that is used to perform a single, related action. Like Java or C, you can declare your own function in Python. Function blocks usually begin with the keyword def followed by the function name and parentheses. 
Any input parameters or arguments should be placed within these parentheses. You can also define parameters inside these parentheses. A return statement with no arguments (i.e. return;) is the same as return None. 

    def function_name( parameters ):
       return value

Notice that Python programs get structured through indentation, i.e. code blocks are defined by their indentation. This principle makes it easier to read and understand other people's Python code, but sometimes it could cause confusion to some people, especially those who are used to using { } to specify a code blocks, like in Java or C. Note also that Python does not require a semi-colon ; at the end of each statement. 

Now, you should define the function `linearmodel` as described above. Note we use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. 
    
    np.dot(w, v) for vector dot produt
    np.dot(W, V) for matrix dot product
    
We require you to complete the following code to compute the predicted output of linear models.


```python
# To do: Insert code to define the linearmodel function here.
def linearmodel(w, b, x):
    '''
    Input: w is a weight parameter, b is a bias parameter, and x is d-dimensional vector (representing a example)
    Output: the predicted output
    '''
    y = np.dot(w, x) + b
    return y
```

The function `linearmodel` gives a prediction on a single example $\mathbf{x}$. It is often the case that we need to provide predictions on several examples $$(\mathbf{x}^{(1)},y^{(1)}),\ldots,(\mathbf{x}^{(n)},y^{(n)})$$ simultaneously. We therefore collect $n$ training examples $(\mathbf{x}^{(1)},y^{(1)}),\ldots,(\mathbf{x}^{(n)},y^{(n)})$ into an input matrix $X\in\mathbb{R}^{n\times d}$ ( $d$ is the number of features) and a vector $\mathbf{y}\in\mathbb{R}^n$. 
That is
![1675907716537](https://user-images.githubusercontent.com/39878006/217697232-bd08253f-7228-4ece-951e-cc0895c47837.png)
Given the data matrix $X\in\mathbb{R}^{n\times d}$, write a function to compute the output $\mathbf{t}=(t^{(1)},\ldots,t^{(n)})^\top$, where $t^{(i)}$ is the output of the linear model $(\mathbf{w},b)$ on $\mathbf{x}$, 
i.e., $t^{(i)}=\mathbf{w}^\top \mathbf{x}^{(i)}+b$. A direct idea is to use the `for` loop to traverse all training examples. We request you to finish the following code.


```python
def linearmat_1(w, b, X):
    '''
    Input: w is a weight parameter, b is a bias parameter, and X is a data matrix (n x d)
    Output: a vector containing the predictions of linear models
    '''
    # n is the number of training examples
    n = X.shape[0]
    t = np.zeros(n)
    for i in range(n):
        # To do: Insert your code to compute the predicted output for the i-th example, and assign it to t[i]
        t[i] = np.dot(w, X[i]) + b # linearmodel(w, b, X[i, :])
    return t
```

### Vectorization

In the function `linearmat_1`, we do prediction by traversing all training examples one by one. This implementation is very slow. Python provides much more efficient implementation in terms of vectorization. By vectorization we mean that we write the prediction in terms of matrix. In Python, vectorized code written in numpy tend to be faster than code that uses a `for` loop. We now show how to achieve this.

As discussed in the lecture, we can absorb the bias into the weight vector by adding a feature of `1`. The benefit is that we do not need to consider separately the bias parameter and the weight parameter. That is,
![1675907954337](https://user-images.githubusercontent.com/39878006/217697542-7a1da1a8-867e-44d2-8230-747eec0675f6.png)

In this case, the predictions $\mathbf{t}$ can be written in terms of a matrix multiplication
![1675908054065](https://user-images.githubusercontent.com/39878006/217697900-6ac6e0d7-edb6-4801-8c68-9c2cf0204f39.png)

where we use a new notation $\mathbf{w}=$

$$
\begin{pmatrix}
  b\\
  w_1\\
  \vdots\\
  w_d
  \end{pmatrix}.
$$

Note here we include the bias in the weight vector $\mathbf{w}$.


```python
def linearmat_2(w, X):
    '''
    a vectorization of linearmat_1.
    Input: w is a weight parameter (including the bias), and X is a data matrix (n x (d+1)) (including the feature 1)
    Output: a vector containing the predictions of linear models
    '''
    
    # To do: Insert you code to get a vectorization of the predicted output computation for a linear model
    return np.dot(X, w) # be careful of the order
```

## Comparing speed of the vectorized vs unvectorized code

We'll see below that the vectorized code already
runs much faster than the non-vectorized code! 

Hopefully this will convince you to always vectorized your code whenever possible. We first import `time` module to include various time-related functions. The time() function returns the current system time in ticks since 00:00:00 hrs January 1, 1970(epoch).

Time for non-vectorized code:


```python
import time
w = np.array([1,1])
b = 1
t0 = time.time()
p1 = linearmat_1(w, b, x_input)
t1 = time.time()
print('the time for non-vectorized code is %s' % (t1 - t0))
```

    the time for non-vectorized code is 0.0009686946868896484
    

Time for vectorized code:


```python
# we add the bias to the weight vector (wb means weights with bias)
wb = np.array([b, w[0], w[1]]) 
# add an extra feature (column in the input) that are just all ones
x_in = np.concatenate([np.ones([np.shape(x_input)[0], 1]), x_input], axis=1)
t0 = time.time()
p2 = linearmat_2(wb, x_in)
t1 = time.time()
print('the time for vectorized code is %s' % (t1 - t0))
print('diff in two implementations is %s' % np.dot(p2 - p1, p2 - p1))
```

    the time for vectorized code is 0.0
    diff in two implementations is 1.4988357199199224e-29
    

Note that **vectorization** is much faster than **non-vectorization**. Also, these two approaches yield almost the same results: the difference is less than $10^{-28}$.|

## Defining the Cost Function

In lecture, we defined the cost function for a linear regression problem using the square loss:

$$C(\mathbf{y}, \mathbf{t}) = \frac{1}{2n} \sum_{i=1}^n (y^{(i)}-t^{(i)})^2,$$
where $y^{(i)}$ is the $i$-th true output and $t^{(i)}$ is the $i$-th predicted output.

As we discussed in the lecture, this can be written as
$$
C(\mathbf{y}, \mathbf{t}) = \frac{1}{2n}(\mathbf{y}-\mathbf{t})^\top (\mathbf{y}-\mathbf{t}).
$$
Use this equation to define the cost function for the linear regression problem. Note that $\mathbf{v}^\top\mathbf{v}$ should be implemented by the function `np.dot`. The underlying reason is that NumPy's transpose() effectively reverses the shape of an array. If the array is one-dimensional, this means it has no effect. Therefore, if `v` is a one-dimensional array in python, `v.T` is the same as $v$.


```python
def cost(w, X, y):
    '''
    Evaluate the cost function in a vectorized manner for 
    inputs `X` and outputs `y`, at weights `w`.
    '''
    # TO DO: Insert your code to compute the cost
    t = linearmat_2(w, X)
    # residual = y - t
    # return np.dot(residual, residual) / (2*len(y))
    return np.dot( (y-t).T, (y-t)) / (2*X.shape[0])
```

For example, the cost for this hypothesis...


```python
cost(wb, x_in, y_target)
```




    246.40613962236236



## Plotting cost in weight space

We'll plot the cost for two of our weights, assuming that bias = 31.11402451. We'll see where that number comes from later.
Notice the shape of the contours are ovals.

We assign some values to $w$, using a `np.arange(start, stop, step)` function call as follows:

    w1s = np.arange(-22, -10, 0.01)
    w2s = np.arange(0, 12, 0.1)

Then we use `np.meshgrid(w1s, w2s)` to build a coordinate system, which is a matrix and each element gives a $(w_1, w_2)$ pair. For each $(w_1, w_2)$, we then apply the `cost` function to compute the cost at this weight vector and therefore get a cost matrix. This is achieved by a double `for` loop. More details about `meshgrid` can be found at [here](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html)

Finally we apply the `plt.contour(W1, W2, z_cost, 25)` to plot the contour, where points in each curve achieves the same `cost`. We also label the x-, y-axis and give a title.


        


```python
w1s = np.arange(-22, -10, 0.01)
w2s = np.arange(0, 12, 0.1)
b = 31.11402451    
W1, W2 = np.meshgrid(w1s, w2s)
z_cost = np.zeros([len(w2s), len(w1s)])  
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w = np.array([b, W1[i, j], W2[i, j]])
        z_cost[i, j] = cost(w, x_in, y_target)
CS = plt.contour(W1, W2, z_cost, 25) # plt.contour(X, Y, Z, [levels], **kwargs)， levels在这里是整数 - 代表等高线条数
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Costs for various values of w1 and w2 for b=31.11402451')
plt.xlabel("w1")
plt.ylabel("w2")
plt.plot([-16.44307658], [6.79809451], 'o') # this will be the minima that we'll find later
plt.show()
```


    
![output_29_0](https://user-images.githubusercontent.com/39878006/217124623-9f1d9afa-aa64-4fcf-9d6c-006446b0d775.png)
 


## Exact Solution

In the lecture, we show that the liner regression problem has a closed-form solution
$$
\mathbf{w}^*=(X^\top X)^{-1}X^\top y.
$$
We now implement the **exact solution** in python. To this aim, we need to compute the `inverse` of a matrix. Python has provided a function to this aim. 

`np.linalg.inv(A)` computes the inverse of the matrix $A$.  We require you to complete the following code to compute the exact solution.


```python
def solve_exactly(X, y):
    '''
    Solve linear regression exactly. (fully vectorized)
    
    Given `X` - n x (d+1) matrix of inputs
          `y` - target outputs
    Returns the optimal weights as a (d+1)-dimensional vector
    '''
    # TODO: Insert your code to return the exact solution
    temp = np.dot(X.T, X)
    temp = np.linalg.inv(temp)
    temp = np.dot(temp, X.T)
    temp = np.dot(temp, y)
    # A = np.dot(X.T, X)
    # c = np.dot(X.T, y)
    # return np.dot(np.linalg.inv(A), c)
    return temp
```


```python
w_exact = solve_exactly(x_in, y_target)
print(w_exact)
```

    [ 31.11402451 -16.44307658   6.79809451]
    


```python
w_exact.shape[0]
```




    3



Now it is clear why we choose bias = 31.11402451 in the visualization of the function.

## Gradient Function and Gradient Descent

In this final section, we are going to use the gradient descend method to find  $w^{*} = \text{arg min}_w C(w)$, that is, the value of parameter $w$ such that $C(w)$ reaches the (locally) minimum value. In gradient descent, we usually work with an *iterative update scheme* for the weight $w$: 
$$\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)} - \eta\nabla C(\mathbf{w}^{(t)}),$$ 
where $\eta$ is a learning rate and $\nabla C(w^{(t)})$ is the gradient evaluated at current parameter value $\mathbf{w}^{(t)}$. 

In order to implement gradient descent, we need to be able to compute the *gradient*
of the cost function with respect to a weight $\mathbf{w}$. In the lecture, we have derived the following closed-form solution of the gradient:

$$\nabla C(\mathbf{w}) = \frac{1}{n}\big(X^\top X\mathbf{w}-X^\top\mathbf{y}\big)=\frac{1}{n}X^\top\big(X\mathbf{w}-\mathbf{y}\big)$$ We require you to complete the following code for the gradient computation.


```python
# Vectorized gradient function
def gradfn(weights, X, y):
    '''
    Given `weights` - a current "Guess" of what our weights should be
          `X` - matrix of shape (N,d+1) of input features including the feature $1$
          `y` - target y values
    Return gradient of each weight evaluated at the current value
    '''
    # TODO: Insert your code to return the gradient
    n = X.shape[0]
    temp = np.dot(X,weights) - y
    return np.dot(X.T, temp) / n
```

With this function, we can solve the optimization problem by repeatedly
applying gradient descent on $w$. We requie you to complete the following code for gradient descent.


```python
def solve_via_gradient_descent(X, y, print_every=500,
                               niter=10000, eta=1):
    '''
    Given `X` - matrix of shape (N,D) of input features
          `y` - target y values
          `print_every` - we report performance every 'print_every' iterations
          `niter` - the number of iterates allowed
          `eta` - learning rate
    Solves for linear regression weights.
    Return weights after `niter` iterations.
    '''
    N, D = np.shape(X)
    # initialize all the weights to zeros
    w = np.zeros([D])
    idx_res = []
    err_res = []
    for k in range(niter):
        # TODO: Insert your code to update w by gradient descent
        w = w - eta * gradfn(w, X, y)

        if k % print_every == print_every - 1:
            print('Weight after %d iteration: %s' % (k, str(w)))
            idx_res.append(k)
            err_res.append(cost(w, X, y))
    plt.plot(idx_res, err_res, color="red", linewidth=2.5, linestyle="-")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()
    return w
```


```python
solve_via_gradient_descent( X=x_in, y=y_target)
```

    Weight after 499 iteration: [21.73748977 -9.06606134 13.51359176]
    Weight after 999 iteration: [ 26.33023653 -12.67941785  10.22425523]
    Weight after 1499 iteration: [ 28.67339713 -14.52290608   8.54607785]
    Weight after 1999 iteration: [ 29.86884767 -15.46343017   7.68989323]
    Weight after 2499 iteration: [ 30.47875123 -15.94327352   7.2530788 ]
    Weight after 2999 iteration: [ 30.78991622 -16.18808346   7.03022167]
    Weight after 3499 iteration: [ 30.94866861 -16.31298235   6.91652281]
    Weight after 3999 iteration: [ 31.02966205 -16.37670417   6.85851511]
    Weight after 4499 iteration: [ 31.07098386 -16.40921423   6.82892033]
    Weight after 4999 iteration: [ 31.09206572 -16.42580044   6.81382144]
    Weight after 5499 iteration: [ 31.10282142 -16.43426251   6.80611819]
    Weight after 5999 iteration: [ 31.10830884 -16.43857976   6.80218809]
    Weight after 6499 iteration: [ 31.11110845 -16.44078236   6.800183  ]
    Weight after 6999 iteration: [ 31.11253677 -16.4419061    6.79916003]
    Weight after 7499 iteration: [ 31.11326549 -16.44247941   6.79863812]
    Weight after 7999 iteration: [ 31.11363727 -16.44277191   6.79837185]
    Weight after 8499 iteration: [ 31.11382695 -16.44292114   6.798236  ]
    Weight after 8999 iteration: [ 31.11392372 -16.44299728   6.7981667 ]
    Weight after 9499 iteration: [ 31.11397309 -16.44303612   6.79813134]
    Weight after 9999 iteration: [ 31.11399828 -16.44305594   6.7981133 ]
    

![output_39_1](https://user-images.githubusercontent.com/39878006/217124655-2203424f-3854-42b7-b00e-2cecaf196ae1.png)
    

    array([ 31.11399828, -16.44305594,   6.7981133 ])


As you can see, gradient descent find solutions very similar to the exact solution. This shows that gradient descent is an effective method to find the best linear model in our case.
