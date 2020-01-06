+++
draft = false
date = 2019-12-28
title = "A Moment of Monoids"
description = "You already know monoids"
slug = "a-moment-of-monoids"
tags = [ "python", "functional programming", "math", "semigroups", "monoids",]
categories = []
externalLink = ""
series = [ "functional programming",]
+++
<!--more-->
## What's functional programming and why should I care?

1. Understanding functional programming will make you a better programmer
2. Dropping FP  jargon in conversations will make you seem smarter than you really are
3. FP isn't trivial, and learning it will expand your mind

There's been a lot written about functional programming, most of it in languages like Haskell or Scala.

So let's do some Python.


## Semigroups

[A semigroup is a nonempty set G with an associative binary operation.][1]

Here's a secret. You probably already know what a semigroup is.

You use them every time you `sum(a_collection_of_numbers)`.

Let's look at some examples.

[1]: https://faculty.etsu.edu/gardnerr/5410/notes/I-1.pdf


```python
import functools
import operator

# A semigroup is a nonempty set G...

stuff = [2, 3, 4]

# ...with an associative binary operation

multiply = operator.mul

# meaning we can compose those elements together
# almost as if we fold them one on top of another
# until we're left with a single thing

# that's what we're doing when we call `reduce` in this example

total = functools.reduce(multiply, stuff)

assert total == 24

letters = ['h', 'e', 'l', 'l', 'o']

greeting = functools.reduce(operator.add, letters)

assert greeting == 'hello'

# more often, we use the built-in sum function to reduce
# sets under addition

numbers = [1, 2, 3]

assert sum(numbers) == functools.reduce(operator.add, numbers)
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

Now you know what semigroups and monoids are. 

Drop those terms at the next masquerade ball you attend and you will be the LIFE of the PARTY ;)

Just make you're you're actually wearing your mask to hide your identity when you do.
