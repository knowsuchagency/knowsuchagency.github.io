+++
draft = true
date = 2019-12-28
title = "Monads for Mortals"
description = "Functional Programming for the Rest of Us"
slug = "monads-for-mortals-in-python"
tags = [ "python", "functional programming", "monads", "functors", "applicatives", "math",]
categories = []
externalLink = ""
series = []
+++
<!--more-->
# Monads for Mortals in Python

a.k.a *YAMT* yet another monad tutorial

## What's functional programming and why should I care?

1. Understanding functional programming will make you a better programmer
2. Dropping FP  jargon in conversations will make you seem smarter than you really are
3. FP isn't trivial, and learning it will expand your mind

I'm not going to waste a lot of time explaining why you should choose a functional style over imperative
programming. I assume that if you're reading this, that you have decided for yourself already that there
is value in these techniques.

My goal is not to write a guide that is comprehensive or perfect. 
I'm not a mathematician; I'm a working stiff without a formal math background.

Personally, I find myself attracted to these concepts because ~~I'm a masochist~~ I believe using them can help us write more correct, robust, testable code.

## Who is this intended for and why?

I've been writing software full-time for the better part of 10 years and almost everything I've done
has been in Python. I've always found myself drawn to functional languages, patterns, and their underlying mathematical
principles for a couple of reasons.

One of those reasons is purely that I find the topic interesting and I find the idea of describing
our programs purely in terms of data and transformations on that data to be elegant and compelling.

The second reason is that writing software is really hard and we inevitably as programmers spend a lot of
time fixing bugs. Functional programming can help us to completely avoid a huge range of potential
bugs. Imagine that one day we can write programs such that if they compile, we're guaranteed that they'll
work as we expect them to. We're not there yet, but modern functional programming languages and techniques can help us get close.

As mentioned before, for my entire career as an engineer, my bread-and-butter has been Python. Most of the
functional programming literature out there is in/for languages like Haskell or Scala, for good reason.

My goal here is to try and take some of those concepts and show how they apply to the Python language, since
there is a lot less written about FP in Python at the moment.

To follow along, you should have an intermediate level of python. 
You should be comfortable with decorators, and have a basic understanding of OOP.
In terms of math, so long as you understand sets and functions, you should be fine.

# Functional Programming

## Semigroups

[A semigroup is a nonempty set G with an associative binary operation.][1]

What?

Let's look at some examples of semigroups.

[1]: https://faculty.etsu.edu/gardnerr/5410/notes/I-1.pdf


```python
import functools
import operator

# A semigroup is a nonempty set G...
g = [2, 3, 4]

# ...with an associative binary operation
op = operator.mul

# so what?

# ... it means we can compose those elements together

total = functools.reduce(op, g)
assert total == 24

g = ['h', 'e', 'l', 'l', 'o']
greeting = functools.reduce(operator.add, g)
assert greeting == 'hello'

# more often, we use the built-in sum function to reduce
# sets under addition

g = [1, 2, 3]
assert functools.reduce(operator.add, g) == sum(g)
```

## Monoids

Definition I.1.1. For a multiplicative binary operation on G × G, we define the
following properties:
(i) Multiplication is associative if a(bc) = (ab)c for all a, b, c, ∈ G.
(ii) Element e ∈ G is a two-sided identity if ae = ea = a for all a ∈ G.

[A monoid is a semigroup with an identity.][1]

[1]: https://faculty.etsu.edu/gardnerr/5410/notes/I-1.pdf

Let's say we were part of an e-commerce site and we had a csv
that contained per-customer order totals for a given month.

We want to add up all the money each customer spent
to figure out the total spent that month.


```python
import csv
import io

january_order_totals = """
customer,order_total
sam,54.71
john,
andrea,72.11
""".strip()

reader = csv.DictReader(io.StringIO(january_order_totals))

cash_spent_per_customer = [row["order_total"] for row in reader]

print(cash_spent_per_customer)
```

    ['54.71', '', '72.11']


We have a minor problem in that we have an empty value -- john didn't spend
any money in January.

We solve this by replacing the empty value with an identity.


```python
cash_spent_per_customer = [
    float(s) if s else 0 for s in cash_spent_per_customer
]

total = sum(cash_spent_per_customer)

print(f"The total spent in january was: {total}")
```

    The total spent in january was: 126.82


We solved our data validation error by creating a monoid.

Cool.
