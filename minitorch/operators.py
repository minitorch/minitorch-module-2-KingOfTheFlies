"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.

def mul(x: float, y: float) -> float:
    """
    f(x, y) = x * y
    """
    return x * y


def id(x: float) -> float:
    """
    f(x) = x
    """
    return x


def add(x: float, y: float) -> float:
    """
    f(x, y) = x + y
    """
    return x + y


def neg(x: float) -> float:
    """
    f(x) = -x
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    returns 1.0 if x is less than y else 0.0
    """
    return float(x < y)


def eq(x: float, y: float) -> float:
    """
    returns 1.0 if x is equal to y else 0.0
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    returns x if x is greater than y else y
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    returns 1.0 if |x - y| < 1e-2 else 0.0
    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """
    f(x) = 1.0 / (1.0 + e^{-x}) - sigmoid function

    for stability if x < 0:
    f(x) = e^{x} / (1.0 + e^{x})
    """
    if lt(x, 0):
        return exp(x) / (1.0 + exp(x))
    return 1.0 / (1.0 + exp(-x))


def relu(x: float) -> float:
    """
    f(x) = x if x is greater than 0, else 0
    """
    return max(0, x)


EPS = 1e-6


def log(x: float) -> float:
    """
    f(x) = log(x)
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """
    f(x) = e^{x}
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    f(x) = log(x) => returns d * f'(x) = d / x
    """
    return d / x


def inv(x: float) -> float:
    """
    f(x) = 1 / x
    """
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """
    f(x) = 1 / x => returns d * f'(x) = -d / (x ** 2)$
    """
    return -d / (x ** 2)


def relu_back(x: float, d: float) -> float:
    """
    f(x) = relu(x) => returns d * f'(x) = d if x >= 0 else 0
    """
    if lt(x, 0):
        return 0
    return d


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies input fn to all elements, returns a new list
    """
    def map_fn(lst: Iterable[float]) -> Iterable[float]:
        return [fn(l) for l in lst]
    return map_fn


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith.

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    def zipWith_fn(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(el1, el2) for el1, el2 in zip(ls1, ls2)]

    return zipWith_fn


def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value x_0

    Returns:
         Function that takes a list lst of elements
         x_1 ... x_n and computes the reduction fn(x_3, fn(x_2,fn(x_1, x_0)))`
    """
    def reduce_fn(ls: Iterable[float]) -> float:
        result = start
        for item in ls:
            result = fn(item, result)
        return result
    return reduce_fn


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of ls1 and ls2 using zipWith and add"
    return zipWith(add)(ls1, ls2)


def negList(lst: Iterable[float]) -> Iterable[float]:
    "Use map and neg to negate each element in lst"
    return map(neg)(lst)


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using reduce and add"
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    "Product of a list using reduce and mul"
    return reduce(mul, 1)(ls)
