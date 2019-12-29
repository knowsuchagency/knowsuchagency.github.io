+++
draft = false
date = 2017-10-30
title = "Collatz Conjecture"
description = "Collatz Conjecture in Python"
slug = "collatz-conjecture-in-python"
tags = [ "python", "math", "programming",]
categories = []
externalLink = ""
series = []
+++
<!--more-->
According to [Wikipedia](https://en.wikipedia.org/wiki/Collatz_conjecture)

>The Collatz conjecture is a conjecture in mathematics named after Lothar Collatz. It concerns a sequence defined as follows: start with any positive integer n. Then each term is obtained from the previous term as follows: if the previous term is even, the next term is one half the previous term. Otherwise, the next term is 3 times the previous term plus 1. The conjecture is that no matter what value of n, the sequence will always reach 1.

## Recursive Implementation

Mathematically, this problem is naturally recursive and this is one way you could implement it as such in Python.


```python
def collatz(n, previous=None):
    if previous is None:
        previous = []
    if n == 1:
        return previous + [n]
    
    next_number = n//2 if n % 2 == 0 else n*3 + 1
    return collatz(next_number, previous+[n])


for n in range(1, 10):
    print(collatz(n))
```

    [1]
    [2, 1]
    [3, 10, 5, 16, 8, 4, 2, 1]
    [4, 2, 1]
    [5, 16, 8, 4, 2, 1]
    [6, 3, 10, 5, 16, 8, 4, 2, 1]
    [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
    [8, 4, 2, 1]
    [9, 28, 14, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]


## Curried Recursive Implementation

This shows an example of [currying](https://en.wikipedia.org/wiki/Currying), where you take a function that takes multiple arguments, and break it up into two or more functions where each function acts on only one argument in isolation. In functional languages like F#, variadic functions (those that take multiple arguments) are automatically curried, but this is how we would go about doing it in Python. 

What I like about this is that we no longer have to assign a default value to the `previous` parameter and check its value on each call to our recursive function (`if previous is None`) like in your previous recursive example.


```python
def collatz(n):
    def get(previous):
        nonlocal n
        if n == 1:
            return previous + [n]
        else:
            previous += [n]
            n = n//2 if n % 2 == 0 else n*3+1
            return get(previous)
    return get([])


for n in range(1, 10):
    print(collatz(n))
```

    [1]
    [2, 1]
    [3, 10, 5, 16, 8, 4, 2, 1]
    [4, 2, 1]
    [5, 16, 8, 4, 2, 1]
    [6, 3, 10, 5, 16, 8, 4, 2, 1]
    [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
    [8, 4, 2, 1]
    [9, 28, 14, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]


## Imperative Generator Implementation

Another way we can solve this problem is using Python generator functions. This example uses a while loop to evaluate new values.


```python
from functools import partial

# this decorator will turn a generator function to one that returns a list
listify = partial(lambda generator: lambda arg: list(generator(arg)))
```

```python
@listify
def collatz(n):
    while n != 1:
        yield n
        n = n//2 if n % 2 == 0 else n*3 + 1
        
    yield n
    
    if n == 1:
        return 


for n in range(1, 10):
    print(collatz(n))
```

    [1]
    [2, 1]
    [3, 10, 5, 16, 8, 4, 2, 1]
    [4, 2, 1]
    [5, 16, 8, 4, 2, 1]
    [6, 3, 10, 5, 16, 8, 4, 2, 1]
    [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
    [8, 4, 2, 1]
    [9, 28, 14, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]


## Recursive Generator Implementation

Finally, this solution (my favorite) uses lazy evaluation and recursion to `yield from` itself until the base condition is met, that `n == 1`.


```python
@listify
def collatz(n):
    yield n
    
    if n == 1:
        return
    
    next_number = n//2 if n % 2 == 0 else n*3 + 1
    
    yield from collatz(next_number) 
        

for n in range(1, 10):
    print(collatz(n))
```

    [1]
    [2, 1]
    [3, 10, 5, 16, 8, 4, 2, 1]
    [4, 2, 1]
    [5, 16, 8, 4, 2, 1]
    [6, 3, 10, 5, 16, 8, 4, 2, 1]
    [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
    [8, 4, 2, 1]
    [9, 28, 14, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]

