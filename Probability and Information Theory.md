## Random Variables

random variable x and vector-valued variables **x**

discrete and continuous

## Probability Distributions

概率质量 Probability Mass Functions (PMF, discrete variables) P

概率密度 Probability Density Functions (PDF, contuous variables) p

边缘概率 Marginal probability (multivariate distribution): integration p(x) and summation P(x)

条件概率 Conditional Probability P(y = y | x = x), chain rule / product rule


Independence and Conditional Independence

期望值 Expectation: the expected value of some function _f(x)_ w.r.t. a probability distribution P(x), is the average or mean value, that _f_ takes on when _x_ is drawn from P.

方差 Variance: gives a measure of how mucht ehvalues of a function of a random variable x vary as we sample different values of x from its probability distribution.

标准差 standard deviation: the square root of the variacne

协方差 Covariance: gives some sense of how much two values are linearly related to each other, as well as the scale of these variables.

相关 correlation:normalize the contribution of each variable, in order to measure only how much the variables are related. 

independence is a stronger requirement than zero covariance:

independent ➡ zero covariance. 

nonzero covariance ➡ dependent. 

zero covariance ➡ no linear dependence.

zero covariance ≠> independent.

covariance matrix.


## Probability Distributions

伯努利分布 Bernoulli Distribution: 一件事情，只有两种可能的结果。伯努利分布中，其中一种结果的概率为a，另一种结果的概率为1-a。

范畴分布 Multinoulli Distribution: 在具有k个不同状态的单个离散随机变量上的分布，其中k为一个有限值。

⚠ Bernoulli 和 Multinoulli的共同点： simple domain, model discrete variables, feasible to enumerate all the states.

高斯分布 Gaussian Distribution: normal distribution. 两参数：mean (central peak) and standard deviation (square -> variance).

standard normal distribution: μ = 0, σ = 1.

multivariate normal distribution

指数分布 Exponential Distributions

拉普拉斯分布 Laplace Distributions

狄拉克分布 Dirac Distribution

经验分布 Empirical Distribution

混合分布 Mixture Distribution

