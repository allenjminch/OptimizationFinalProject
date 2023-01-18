'''
Created on Nov 29, 2022

@author: Allen Minch
'''
import numpy as np
import matplotlib.pyplot as plt
import time

# setting up the least squares problem (code taken from HW 9 submission)
X = np.reshape(np.random.normal(loc=0.0, scale=1.0, size=1000), (100, 10))
n = 100
epsilon = np.random.normal(loc=0.0, scale=1.0, size=100)
beta = np.ones((10,))
y = np.matmul(X, beta) + epsilon
outer = np.matmul(X.transpose(), X)
dot = np.matmul(X.transpose(), y)

# true solution
theta_hat = np.matmul(np.linalg.inv(outer), dot)

# the full gradient of F(theta)
def grad(theta):
    # this is what one finds the gradient to be if you expand out F(theta) as a quadratic function
    outer = np.matmul(X.transpose(), X)
    dot = np.matmul(X.transpose(), y)
    return 2 * (np.matmul(outer, theta) - dot) / n

# implementation of full gradient descent on F(theta) with an initial guess of theta0 for the optimal point and constant step size t
def gradDescent(theta0, t):
    iterates = np.zeros((10001, 10))
    iterates[0] = theta0
    iterations = [0]
    theta = theta0
    k = 0
    while k < 10000:
        theta = theta - t * grad(theta)
        k += 1
        iterates[k] = theta
        iterations.append(k)
    return iterations, iterates

# implementation of stochastic gradient descent with an initial guess theta0 for the optimal point, a minibatch size B, 
# and an initial step size t
def SGDMB(theta0, t, B, p):
    iterates = np.zeros((10001, 10))
    iterates[0] = theta0
    iterations = [0]
    theta = theta0
    k = 0
    while k < 10000:
        randSample = np.random.choice(n, B)
        trans = X[randSample, :].transpose()
        outer = np.matmul(trans, X[randSample, :])
        dot = np.matmul(trans, y[randSample])
        grad = 2 * (np.matmul(outer, theta) - dot) / B
        step = t / (k + 1)**p
        theta = theta - step * grad
        k += 1
        iterates[k] = theta
        iterations.append(k)
    return iterations, iterates

# adam algorithm for the final project
# alpha is the initial step size, theta0 the initial guess, and B the minibatch size
def adam(theta0, alpha, B, p):
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10**(-8)
    iterates = np.zeros((10001, 10))
    iterates[0] = theta0
    iterations = [0]
    theta = theta0
    k = 0
    m = np.zeros((10,))
    v = np.zeros((10,))
    while k < 10000:
        k += 1
        randSample = np.random.choice(n, B)
        trans = X[randSample, :].transpose()
        outer = np.matmul(trans, X[randSample, :])
        dot = np.matmul(trans, y[randSample])
        g = 2 * (np.matmul(outer, theta) - dot) / B
        g2 = np.power(g, 2)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g2
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        step = alpha / k**p
        theta = theta - step * (m_hat / (np.sqrt(v_hat) + epsilon * np.ones((10,))))
        iterates[k] = theta
        iterations.append(k)
    return iterations, iterates

theta0 = np.ones((10,))
# fullGradDescent = gradDescent(theta0, 1)
# stocGradDescent = SGDMB(theta0, 0.16, 10)
# adamResults = adam(theta0, 0.16, 10)

# fullGradError = [np.linalg.norm(fullGradDescent[1][i] - theta_hat) for i in range(10001)]
# stocGradError = [np.linalg.norm(stocGradDescent[1][i] - theta_hat) for i in range(10001)]


# step size experiment (varying the initial step size) (an experiment that I tried but didn't end up using in the report)
plt.figure()
plt.title("Least squares adam with sigma = 1.0 and B = 10")
for i in range(6):
    alpha = 0.1 * 2**i
    adamResults = adam(theta0, 0.1 * 2**i, 10)
    adamError = [np.linalg.norm(adamResults[1][i] - theta_hat) for i in range(10001)]
    plt.loglog(adamResults[0][100:], adamError[100:], label = "alpha = {0}".format(alpha))
plt.xlabel("Iteration")
plt.ylabel("Error in iterate")
plt.legend(loc = 'best')
plt.show()



# batch size experiment
fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
plt.figure()
plt.title("Least squares adam and SGD with sigma = 1.0, init step size = 0.4, p = 1")
for i in range(3):
    B = 2 * 4**i
    tAdamStart = time.time()
    adamResults = adam(theta0, 0.4, 2 * 4**i, 1)
    tAdamFinish = time.time()
    print("With B = {0}, Adam took t = {1}".format(B, tAdamFinish - tAdamStart))
    tSGDStart = time.time()
    SGDResults = SGDMB(theta0, 0.4, 2 * 4**i, 1)
    tSGDFinish = time.time()
    print("With B = {0}, SGD took t = {1}".format(B, tSGDFinish - tSGDStart))
    adamError = [np.linalg.norm(adamResults[1][i] - theta_hat) for i in range(10001)]
    stocGradError = [np.linalg.norm(SGDResults[1][i] - theta_hat) for i in range(10001)]
    ax1.loglog(adamResults[0][1:], adamError[1:], label = "Adam, B = {0}".format(B))
    ax1.loglog(SGDResults[0][1:], stocGradError[1:], label = "SGD, B = {0}".format(B))
    plt.loglog(adamResults[0][1:], adamError[1:], label = "Adam, B = {0}".format(B))
    plt.loglog(SGDResults[0][1:], stocGradError[1:], label = "SGD, B = {0}".format(B))
plt.xlabel("Iteration")
plt.ylabel("Error in iterate")
plt.legend(loc = 'best')
plt.figure()
tGDStart = time.time()
GDResults = gradDescent(theta0, 0.65)
tGDFinish = time.time()
print("GD took t = {0}".format(tGDFinish - tGDStart))
GDError = [np.linalg.norm(GDResults[1][i] - theta_hat) for i in range(10001)]
plt.title("Least squares GD with constant step size 0.65")
plt.loglog(GDResults[0][1:], GDError[1:], label = "GD")
plt.xlabel("Iteration")
plt.ylabel("Error in iterate")
plt.show()


# experiment involving sigma
plt.figure()
tAdamStart = time.time()
adamResults = adam(theta0, 0.4, 10, 1)
tAdamFinish = time.time()
print("With sigma = 2.0, Adam took t = {0}".format(tAdamFinish - tAdamStart))
tSGDStart = time.time()
SGDResults = SGDMB(theta0, 0.4, 10, 1)
tSGDFinish = time.time()
print("With sigma = 2.0, SGD took t = {0}".format(tSGDFinish - tSGDStart))
adamError = [np.linalg.norm(adamResults[1][i] - theta_hat) for i in range(10001)]
SGDError = [np.linalg.norm(SGDResults[1][i] - theta_hat) for i in range(10001)]
plt.title("Least squares adam and SGD with alpha = 0.4, B = 10, p = 1, sigma = 2.0")
plt.xlabel("Iteration")
plt.ylabel("Value of iterate")
plt.loglog(adamResults[0][1:], adamError[1:], label = "Adam, sigma = 2.0")
plt.loglog(SGDResults[0][1:], SGDError[1:], label = "SGD, sigma = 2.0")
plt.legend(loc = "best")
plt.figure()
tGDStart = time.time()
GDResults = gradDescent(theta0, 0.15)
tGDFinish = time.time()
print("With sigma = 2.0, GD took t = {0}".format(tGDFinish - tGDStart))
GDError = [np.linalg.norm(GDResults[1][i] - theta_hat) for i in range(10001)]
plt.title("Least squares GD with constant step size 0.15, sigma = 2.0")
plt.loglog(GDResults[0][1:], GDError[1:], label = "GD")
plt.xlabel("Iteration")
plt.ylabel("Error in iterate")
plt.show()

# experiment with step sizes declining according to different power laws
plt.figure()
plt.title("Least squares adam and SGD with sigma = 1.0, alpha = 0.4, B = 10")
for i in range(3):
    p = 0.25 * 2**i
    tAdamStart = time.time()
    adamResults = adam(theta0, 0.4, 10, p)
    tAdamFinish = time.time()
    print("With p = {0}, Adam took t = {1}".format(p, tAdamFinish - tAdamStart))
    tSGDStart = time.time()
    SGDResults = SGDMB(theta0, 0.4, 10, p)
    tSGDFinish = time.time()
    print("With p = {0}, SGD took t = {1}".format(p, tSGDFinish - tSGDStart))
    adamError = [np.linalg.norm(adamResults[1][i] - theta_hat) for i in range(10001)]
    stocGradError = [np.linalg.norm(SGDResults[1][i] - theta_hat) for i in range(10001)]
    plt.loglog(adamResults[0][1:], adamError[1:], label = "Adam, p = {0}".format(p))
    plt.loglog(SGDResults[0][1:], stocGradError[1:], label = "SGD, p = {0}".format(p))
plt.xlabel("Iteration")
plt.ylabel("Error in iterate")
plt.legend(loc = 'best')
plt.show()