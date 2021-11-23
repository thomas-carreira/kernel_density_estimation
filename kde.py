import numpy as np


def normalKernel1D(X):
    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (X ** 2))


def normalKernelMultiD(X, d):
    return (1 / ((2 * np.pi) ** (d / 2))) * np.exp(-0.5 * np.dot(X.T, X))


def deltaProd(X, c, h, d):
    """ Fits a Normal Kernel to its one of the attributes of the training examples
        h : the size of the kernel
        d : the dimensionality of the space
        c : the center of the kernel """
    result = 1
    for i in range(d):
        result = result * normalKernel1D((X[i] - c[i]) / h)
    return result


def deltaMultiD(X, c, h, d):
    """ Fits mulrivariate form of Normal Kernel
        h : the size of the kernel
        d : the dimensionality of the space
        c : the center of the kernel """

    return normalKernelMultiD((X - c) / h, d)


def densityEstimation(X, c, h, d):
    """ Fits mulrivariate form of Normal Kernel
        h : the size of the kernel
        d : the dimensionality of the space
        c : the center of the kernel """
    DE = np.empty([0,0])

    # Si c n'a qu'une seule instance, on ne peut pas boucler sur c,
    # car s'il est en deux dimension on ne va pas obtenir le résultat attendu
    if c.size / 2 == 1:
        coef = 1 / (1 * h)
        for x in X:
            DE = np.append(DE, coef * deltaMultiD(x, c, h, d))
    else:
        n = len(c)
        coef = 1 / (n * h)
        for x in X:
            res = 0
            for ci in c:
                res += coef * deltaMultiD(x, ci, h, d)

            DE = np.append(DE, res)

    return DE