+++
draft = false
date = 2017-09-20
title = "Domain Modeling for Great Good"
description = "Making illegal states unrepresentable"
slug = "avoiding-illegal-state-through-domain-modeling"
tags = [ "python", "architecture", "domain modeling", "programming",]
categories = []
externalLink = ""
series = []
+++
<!--more-->
## In the beginning, there was the Imperative

It all started when a friend of mine asked for help on their introductory computer science class. 

They had been assigned the task of writing a program that would ask a user to **input a number of seconds**, and then print how much time that was **in terms of weeks, days, hours, and seconds**.

For example, 

```

f(0) -> "0 Seconds"
f(100) -> "1 Minute 40 Seconds"
f(86400) -> "1 Day"
f(172801) -> "2 Days 1 Second"
```

My friend had to turn in the assignment that night, so I quickly banged out a naive solution with the promise that I'd go over it with them later.


```python
def seconds_to_string(seconds):
    if seconds < 0:
        raise ValueError("seconds must be greater than zero")
    elif seconds == 0:
        return '0 seconds'
    
    string = ''
    
    weeks = seconds / 60 / 60 / 24 // 7
    if weeks:
        string += f'{weeks} weeks '
    seconds -= weeks * 60 * 60 * 24 * 7
    
    days = seconds / 60 / 60 // 24
    if days:
        string += f'{days} days '
    seconds -= days * 60 * 60 * 24
    
    hours = seconds / 60 // 60
    if hours:
        string += f'{hours} hours '
    seconds -= hours * 60 * 60
    
    minutes = seconds // 60
    if minutes:
        string += f'{minutes} minutes '
    seconds -= minutes * 60
        
    if seconds:
        string += f'{seconds} seconds'
        
    return string
    
    
seconds_to_string(987987)
```

    '1.0 weeks 4.0 days 10.0 hours 26.0 minutes 27.0 seconds'



## Where do we go from here?

This function works, but one thing we might want to do is to create a type of object that more clearly illustrates the semantics of a time measurement.

For starters, it would be great if we could [make illegal states unrepresentable](https://fsharpforfunandprofit.com/posts/designing-with-types-making-illegal-states-unrepresentable/) as much as possible. For example, let's say that we only want units of time like `Weeks, Days, Hours, Minutes, and Seconds`, to be represented as natural numbers. That means no negative numbers or floating point representations of those types.

For the sake of argument, let's also say that we want equivalent units of time to be equal to one another. We also want to be able to perform arithmetic with our units in ways that make sense.

Lastly, units of time should know how to represent themselves in string form.

i.e.

```

Days(7) == Weeks(1)

Minutes(1) + Seconds(20) -> Minutes(1) # whole numbers only, left unit takes precedent

Seconds(20) + Minutes(1) -> Seconds(80)

str(Seconds(0)) -> '0 Seconds'
str(Seconds(1)) -> '1 Second'
str(Seconds(101)) -> '101 Seconds'
```

---


```python
from fractions import Fraction

class TimeUnit(int):
    """A class that defines the semantics of a unit of time i.e. seconds, minutes, hours etc."""

    def __new__(cls, x):
        """Ensure no negative units are created."""
        if x < 0:
            raise ValueError(f'{cls.__name__} must be greater than zero. x={x}')
        return super().__new__(cls, x)

    def __eq__(self, other):
        if isinstance(other, TimeUnit):
            return int(self.to_seconds()) == other.to_seconds()
        return super().__eq__(other)
    
    def __add__(self, other):
        if isinstance(other, TimeUnit):
            return self.from_seconds(int(self.to_seconds())+other.to_seconds())
        return super().__add__(other)
    
    def __radd__(self, other):
        if isinstance(other, TimeUnit):
            return self.from_seconds(int(self.to_seconds())+other.to_seconds())
        return super().__radd__(other)
    
    def __sub__(self, other):
        if isinstance(other, TimeUnit):
            return self.from_seconds(int(self.to_seconds())-other.to_seconds())
        return super().__sub__(other)
    
    def __mul__(self, other):
        if isinstance(other, TimeUnit):
            return self.from_seconds(int(self.to_seconds())*other.to_seconds())
        return super().__mul__(other)
    
    def __div__(self, other):
        if isinstance(other, TimeUnit):
            return self.from_seconds(int(self.to_seconds())/other.to_seconds())
        return super().__div__(other)
    
    def __repr__(self):
        singular = self == 1
        units = self.__class__.__name__[:-1] if singular else self.__class__.__name__
        return f'{int(self)} {units}'
    
    def __str__(self):
        return repr(self)

    @classmethod
    def from_seconds(cls, seconds):
        raise NotImplementedError

    def to_seconds(self):
        raise NotImplementedError
        
# Create our Seconds, Hours, Days, and Weeks classes
# from out TimeUnit base class

class Seconds(TimeUnit):
    @classmethod
    def from_seconds(cls, seconds):
        return cls(seconds)

    def to_seconds(self):
        return self


class Minutes(TimeUnit):
    @classmethod
    def from_seconds(cls, seconds):
        return cls(Fraction(seconds, 60))

    def to_seconds(self):
        return Seconds(self * 60)


class Hours(TimeUnit):
    @classmethod
    def from_seconds(cls, seconds):
        return cls(Fraction(seconds, 60 * 60))

    def to_seconds(self):
        return Seconds(self * 60 * 60)


class Days(TimeUnit):
    @classmethod
    def from_seconds(cls, seconds):
        return cls(Fraction(seconds, 60 * 60 * 24))

    def to_seconds(self):
        return Seconds(self * 60 * 60 * 24)


class Weeks(TimeUnit):
    @classmethod
    def from_seconds(cls, seconds):
        return cls(Fraction(seconds, 60 * 60 * 24 * 7))

    def to_seconds(self):
        return Seconds(self * 60 * 60 * 24 * 7)
```

```python
# poor man's tests
# in the real world, we should test all the operations
# ideally using something like hypothesis, but this hopefully
# serves well enough to demonstrate that our units of measure
# now work well with one-another

def test_equality():
    assert Seconds(60) == Minutes(1)
    assert Minutes(60) == Hours(1)
    assert Hours(24) == Days(1)
    assert Days(7) == Weeks(1)
    print('equality tests passed')
    
def test_conversions():
    assert Seconds(1) + Minutes(1) == 61
    assert Minutes(1) + Seconds(1) == 1
    assert Minutes(1) + Hours(1) == 61
    assert Hours(1) + Minutes(1) == 1
    assert Hours(1) + Days(1) == 25
    assert Days(1) + Hours(1) == 1
    assert Days(1) + Weeks(1) == 8
    assert Weeks(1) + Days(1) == 1
    print('conversions passed')
    
test_equality()
test_conversions()
```

    equality tests passed
    conversions passed


## Why all the new code?

For starters, we know that any instances of our units of time will be positive whole numbers. Additionally, we can also compare and convert our units of measurement to one another pretty easily now that we've defined the respective *dunder* methods

So that's pretty cool. We can even continue to use our old function with our new types so long as we make sure the argument is in *seconds*.


```python
seconds_to_string(
    Weeks(1).to_seconds() + \
    Hours(3).to_seconds() + \
    Seconds(78)
)
```

    '1.0 weeks 3.0 hours 1.0 minutes 18.0 seconds'



### [Comprehensions](http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html)

Before we move on, we're going to see a few examples of comprehensions.

For the uninitiated, comprehensions follow the following format,

[**statement** for **variable** in **iterable** if **optional predicate**]

in addition, the `.join` method of a string takes an iterable of strings as an argument and returns a string that joins the elements of the iterable together.


```python
# for a comma-separated list of even numbers 0-through-10
print(', '.join(str(n) for n in range(10) if n % 2 == 0), end='\n\n')

random_things = ['pizza', None, "beer", 0, 42, {}, 'netflix']
things_string = '++'.join(str(thing) for thing in random_things)

print('all the things:', things_string , end='\n\n')

# notice the predicate can be a variable itself, it will be
# evaluated on its "truthiness"
some_things_string = '++'.join(str(thing) for thing in random_things if thing)

print('only that which is True:', some_things_string)
```

    0, 2, 4, 6, 8
    
    all the things: pizza++None++beer++0++42++{}++netflix
    
    only that which is True: pizza++beer++42++netflix


Anyway, we can still do better.

Let's start by re-writing our old function to take any instance **TimeUnit**. Since **TimeUnit** instances can't be negative, we don't have to test for that in our function. In addition, every **TimeUnit** has a `from_seconds` method and knows how to perform arithmetic correctly with other members of the same type -- saving us some code.


```python
def time_unit_to_string(unit):
    
    seconds = unit.to_seconds()
    
    if seconds == 0:
        return str(seconds)
    
    # a stack where we'll store all our units
    # greater than one
    
    units = []
    
    weeks = Weeks.from_seconds(seconds)
    if weeks:
        units.append(str(weeks))
    seconds -= weeks
    
    days = Days.from_seconds(seconds)
    if days:
        units.append(str(days))
    seconds -= days
    
    hours = Hours.from_seconds(seconds)
    if hours:
        units.append(str(hours))
    seconds -= hours
    
    minutes = Minutes.from_seconds(seconds)
    if minutes:
        units.append(str(minutes))
    seconds -= minutes
        
    if seconds:
        units.append(str(seconds))
        
    return ' '.join(units)
```

## You promised recursion

"But, Stephan", you interject. 

"Can't we do better than to have a stack in our `time_unit_to_string` function where we append the values we want to return? Also, constantly changing the value of variables like `seconds` in the function makes me sad. I like it when my program is correct, but I LOVE it when it's pure, idiomatic, functional, well-tested, and well-documented."

I wipe a tear from my eye, knowing the harsh vicissitudes of life may unfairly rob you of much of your innocent idealism, but recursion isn't going to explain itself itself itself itself...

### Recursion


A **recursive** function, is simply a function that calls itself. A recursive function must have at least one **base case**, which is the point at which the function ceases to return from itself and actually spits something out.


```python
# This contrived example of a recursive function that doesn't do much

def diminish(n):
    if n <= 0:
        print('this is the base case')
        return n
    print(f'called diminish({n})')
    return diminish(n-1)

x = diminish(5)
assert x == 0
```

    called diminish(5)
    called diminish(4)
    called diminish(3)
    called diminish(2)
    called diminish(1)
    this is the base case


### Closures

A **closure** is simply a fancy name for a function defined within another function. The inner function will have access to the higher function's [namespace](https://stackoverflow.com/questions/3913217/what-are-python-namespaces-all-about).

Let's create a function that allows us to "count" by certain numbers as an example.


```python
# we define from_ as a nonlocal variable in our closure 
# since we want to change its value from that scope,
# not just "read" it

def counts(from_=0, by=1):
    def closure():
        nonlocal from_
        result = from_+by
        from_ += by
        return result
    return closure

counter = counts()
print(counter(), counter(), counter())

count_from_10_by_5 = counts(from_=10, by=5)
print(' '.join(str(count_from_10_by_5()) for _ in range(3)))
```

    1 2 3
    15 20 25


### Caveat

This following example is just one way you *could* use recursion to avoid loops and changing variables. I am **definitely** not trying to argue this is the "best" way to solve this particular problem, as we'll see. Also, I just want to make it clear that as a mere mortal that doesn't intuitively perceive the universe in its natural resplendent recursive beauty, this is probably (almost certainly) not the best example of a recursive function that solves our problem. 

With that out of the way...


```python
def seconds_to_string(seconds):
    # coerce integers into our Seconds class
    seconds = Seconds(seconds)

    def inner(seconds, string, unit_class):
        
        # These are the base cases of the recursive function
        # where our function will eventually terminate and return
        if seconds == 0 and string:
            # in this case, the input was larger than sixty, so some unit of time
            # besides seconds was computed, but there are no seconds left over
            
            # since we append a space to each string we return in a recursive call
            # where some unit greater than 1 was computed, we need to strip the output
            return string.strip()
        elif seconds < 60:
            # in this case, we may or may not have computed units of time other than
            # seconds, but since we append the seconds string at the end -- after any
            # spaces -- we don't need to strip the output of whitespace
            return string + str(seconds)
        
        time_unit = unit_class.from_seconds(seconds)
        
        # if the unit of time is not zero i.e. `Weeks.from_seconds(800) == 0`
        # then we append the string for that unit of time to the last string
        # that was input to the function and add it as a parameter to the 
        # next function call
        s = str(time_unit) + ' ' if time_unit else ''
        
        if time_unit.__class__ is Weeks:
            return inner(seconds-time_unit, string+s, Days)
        elif time_unit.__class__ is Days:
            return inner(seconds-time_unit, string+s, Hours)
        elif time_unit.__class__ is Hours:
            return inner(seconds-time_unit, string+s, Minutes)
        elif time_unit.__class__ is Minutes:
            return inner(seconds-time_unit, string+s, Seconds)
            
    return inner(seconds, '', Weeks)



def test_string_func(func):
    input_ = 8989
    output = func(input_)
    assert output == '2 Hours 29 Minutes 49 Seconds', f'{func.__name__}({input_}) -> {output}'
    
    input_ = 0
    output = func(input_)
    assert output == '0 Seconds', f'{func.__name__}({input_}) -> {output}'
    
    input_ = 60
    output = func(input_)
    assert output == '1 Minute', f'{func.__name__}({input_}) -> {output}'
    
    input_ = 1
    output = func(input_)
    assert output == '1 Second', f'{func.__name__}({input_}) -> {output}'
    
    input_ = Seconds(61) + Hours(1) + Weeks(1)
    output = func(input_)
    assert output == '1 Week 1 Hour 1 Minute 1 Second', f'{func.__name__}({input_}) -> {output}'
    
    print(f'stringification tests passed for {func.__name__}')
    
    
test_string_func(seconds_to_string)
```

    stringification tests passed for seconds_to_string


### How can we make this less horrible?

Well, for one, we could use iteration to avoid dispatching on the `time_class` argument in the recursive function


```python
def seconds_to_string_with_for_loop(seconds):
    
    seconds = Seconds(seconds)

    def inner(seconds, string):
        
        if seconds == 0 and string:
            return string.strip()
        
        elif seconds < 60:
            return string + str(seconds)
        
        for unit in (Weeks, Days, Hours, Minutes, Seconds):
            time_unit = unit.from_seconds(seconds)
            if time_unit:
                return inner(seconds-time_unit, string + str(time_unit) + ' ')

    return inner(seconds, '')

seconds_to_string_with_for_loop(72783)
```

    '20 Hours 13 Minutes 3 Seconds'



### Closures, again

Let's see another example with a factory function that uses a closure. Again, I don't think this is actually a good solution to this problem at all, but I think it's useful as an example in how one might think about using closures.


```python
def unit_factory(seconds):
    
    seconds = Seconds(seconds)

    def get_time_unit_instance(unit):
        nonlocal seconds
        time_unit = unit.from_seconds(seconds)
        seconds -= time_unit
        return time_unit

    return get_time_unit_instance



def seconds_to_string_with_closure_factory(seconds):
    if seconds < 60:
        return str(Seconds(seconds))
    
    factory = unit_factory(seconds)
    time_units = (factory(u) for u in (Weeks, Days, Hours, Minutes, Seconds))
    
    return ' '.join(str(unit) for unit in time_units if unit)


seconds_to_string_with_closure_factory(27364283)
```

    '45 Weeks 1 Day 17 Hours 11 Minutes 23 Seconds'



### Sweet, Sweet [Generators](https://realpython.com/blog/python/introduction-to-python-generators/)

Lastly, a generator is a callable object in Python (normally just a function) that abides by the iterator protocol (you can loop over it) that *yields* values as opposed to only *returning* them.

In this example, you can clearly understand what's happening by reading the code, which I would argue is what Python is all about. 

Rather than **imperatively** decrementing seconds each time we calculate a unit of time, or using a **recursive** function or **closure** to do the same thing, we rely on the semantics of the **generator** to do it for us much more clearly by decrementing the `sec` variable within the generator function's *own* scope **after** we yield the value we want! How cool is that?! 

This works because the Python interpreter basically suspends execution and starts back up immediately after the yield statement of a generator function, so we know that `secs` will be decremented before the next iteration of the loop, every time.

I think this really makes more intuitive sense than any of the previous examples, but at least we may have a couple more tools in our mental toolbox now that we've seen a couple different solutions.


```python
def seconds_to_string_with_generator_func(seconds):
    
    seconds = Seconds(seconds)
    
    if seconds < 60:
        return str(seconds)
    
    def gen_unit_strings(secs):
        units = (Weeks, Days, Hours, Minutes, Seconds)
        for unit in (u.from_seconds(secs) for u in units):
            if unit: yield str(unit)
            secs -= unit
    
    return ' '.join(gen_unit_strings(seconds))


seconds_to_string_with_generator_func(2342455)
```

    '3 Weeks 6 Days 2 Hours 40 Minutes 55 Seconds'


