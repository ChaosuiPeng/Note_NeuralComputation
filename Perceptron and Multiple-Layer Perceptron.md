### Perceptron and Multiple-Layer Perceptron
In this exercise, you will learn the following
* data generation with the random library
* define a linear classification model
* python class
* train and test a perceptron

We will first need to import some necessary libraries
* numpy provides a high-performance multidimensional array object, and tools for working with these arrays. 
* random implements pseudo-random number generators
* matplotlib is a plotting library 
* sklearn.metrics provides method to compute the performance measure of models

```python
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
```

## Data Generation

We will generate a dataset for binary classification with the output label being either $+1$ or $-1$. This is achieved by a function `generate_data`.

**Input**: `no_points` is the number of examples in the dataset

**Output**: the dataset for binary classification. `X` is a nx2 matrix and `Y` is a nx1 vector, where n is the number of points.

```python
def generate_data(no_points):
    X = np.zeros(shape=(no_points, 2))
    Y = np.zeros(shape=no_points)
    for ii in range(no_points):
        X[ii, 0] = random.randint(0,20)
        X[ii, 1] = random.randint(0,20)
        if X[ii, 0]+X[ii, 1] > 20:
            Y[ii] = 1 
        else:
            Y[ii] = -1
    return X, Y
```

`X = np.zeros(shape=(no_points, 2))` is used to generate a n-by-2 **zero** matrix X, and `Y = np.zeros(shape=no_points)` is to generate a n-by-1 **zero** vector Y. 
Then we use a `for` loop to set the value of X and Y.

* `X[ii, 0] = random.randint(0,20)`: the **first** feature of the `ii`-th example is randomly generated from the set {0,1,2,...,19}.
* `X[ii, 1] = random.randint(0,20)`: the **second** feature of the `ii`-th example is randomly generated from the set {0,1,2,...,19}.

We say $x^{(ii)}$ is a positive example if $x^{(ii)}_1+x^{(ii)}_2>20$, and a negative example otherwise. Then, we generate a **linearly separable** dataset.

## Class 

Classes provide a means of bundling data and functionality together. Creating a new class creates a new type of object, allowing new instances of that type to be made. The class definitions begin with a `class` keyword, followed by the class name and a colon. 

All classes have a function called __init__(), which is always executed when the class is being initiated. 

`Example`: **Create a class** named Person, use the __init__() function to assign values for name and age:

```python
class Person():
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

**Create new object instances** (instantiation) of that class.

```python
p1 = Person("John", 36)

print(p1.name)
print(p1.age)
```

    John
    36

In this example, we create an object p1, and assign the name attribute with "John", the age attribute with 36.

We can **create some methods** for the class. 
Here let us create a method in the Person class (Insert a function that prints a greeting, and execute it on the p1 object):

```python
class Person():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def myfunc(self):
        print("Hello my name is " + self.name)

p1 = Person("John", 36)
p1.myfunc()
```
