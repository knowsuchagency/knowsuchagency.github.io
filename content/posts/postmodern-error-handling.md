+++
draft = false
date = 2017-02-24
title = "Postmodern Error Handling in Python 3"
description = "An exploration of error handling in Python"
slug = "postmodern-error-handling"
tags = [ "python", "error-handling", "programming",]
categories = []
externalLink = ""
series = []
+++
<!--more-->
## Postmodern error handling in Python 3.6

I'll be the first person to admit I have no idea what postmodernism actually means, but it sounds cool for an article on error-handling, and I would argue that the facilities (post?)modern Python provides for handling errors is pretty damn cool. derp

Recently, an acquaintance of mine posed this question on a message board we both participate in:

> okay nerds
what do you call a union-type with three states?

> so I was writing a Maybe implementation (in python) to deal with a processing pipeline, but then it turned into an Either, and then it turned into something with three states

> basically there's either 1) the final value, 2) no value but a known error occurred, 3) no value but an unknown error occurred

> the final states could be: 1) hey it succeeded, here's the object's id 2) no value but your json blob was missing a required attribute (or some other known error), or 3) we encountered an error that the programmer didn't forsee

To which another strapping young gentleman replied:

>  if it were Rust you could enum your errors ಠ‿ಠ

Well, truth be told, I feel like I probably know about as little about [**fp**](https://www.smashingmagazine.com/2014/07/dont-be-scared-of-functional-programming/) as I do about postmodernism, but I thought this was an interesting question and I do at least know that Python's standard library does provide us Enums ┬─┬﻿ ノ( ゜-゜ノ) so I figured I would take a crack at it.

---

To start, let's quickly talk about what an Enum (enumeration) is for those of us who have yet to encounter them.

At a very basic level, I would define an Enum as a particular set of choices (or states).

Let's say, for example, that you [lost a bet](https://www.youtube.com/watch?v=zBYz_RInbrQ), and now have a set of options to choose from as punishment.

Those options are

* **A**: Get slapped 10 times right now
* **B**: Get slapped 5 times at any random time from now to eternity

Those are the only 2 possibilities; you don't get any other choices. We could model this idea using an Enum.


```python
from enum import Enum

class SlapBet(Enum):
    TEN_SLAPS = 1 # get slapped ten times right now
    FIVE_SLAPS = 2 # fear slaps possibly for eternity
    
smart_choice = SlapBet.TEN_SLAPS

stupid_choice = SlapBet.FIVE_SLAPS

print(
    smart_choice, # notice the pretty repr
    smart_choice == SlapBet.TEN_SLAPS,
    smart_choice == stupid_choice,
    sep='\n'
)
```

    SlapBet.TEN_SLAPS
    True
    False


So, why would we do this? Why go through the trouble of using an Enum when we could functionally do the same thing with something like integers? Well, the first reason, and the most important one, in my opinion, is readability. 

The second reason is that Enum instances can hold values we may want to use. 

Let's say you have a function that has an optional parameter.


```python
def connection(person_1, person_2, relationship='knows'):
    """Return a string that represents the relationship between two people."""
    
    # this sexy f"" string format is part of the new Python 3.6
    # hotness. An article for another day.
    
    return f"{person_1} {relationship} {person_2}"

connection("Ron", "Harry", relationship="is best friends with")
```

    'Ron is best friends with Harry'



Now let's say we want to restrain the possible relationships that people can have.

Maybe we want to do this to prevent errors or to simply prevent weirdness like the following.


```python
connection("Ron", "Harry", relationship="is dating Hermoine but secretly wants")
```

    'Ron is dating Hermoine but secretly wants Harry'



### Enums to the rescue!


```python
from enum import Enum

class Relationship(Enum):
    knows = 'knows'
    likes = 'likes'
    loves = 'loves'
    detests = 'detests'
    bff = "is best friends with"

def connection(person_1, person_2, relationship=Relationship.likes):
    """Return a string that represents the relationship between two people."""
    
    # notice we use the .value of the Relationship instance to get the string
    
    return f"{person_1} {relationship.value} {person_2}"

connection("Ron", "Harry", relationship=Relationship.bff)
```

    'Ron is best friends with Harry'



## Interlude

Now, before tackle the original question that started us on this journey, let's quickly talk about another new and incredibly useful and important feature since Python 3.5 that has been vastly improved in Python 3.6 - type annotations.

In our previous example, there would be nothing to prevent someone from doing the following:


```python
evil = lambda: 'power'
evil.value = 'corrupts'

connection("Ron", "Harry", relationship=evil)
```

    'Ron corrupts Harry'



That's obviously not how we intended our function to be used, but part of what makes Python so powerful, its dynamic nature, is what allows such nefarious behavior. If only there were a way to save ourselves and people who use our code from making such mistakes...

### Behold!


```python
from enum import Enum

class Relationship(Enum):
    knows: str = 'knows'
    likes: str = 'likes'
    loves: str = 'loves'
    detests: str = 'detests'
    bff: str = "is best friends with"
    
def connection(person_1: str,
               person_2: str,
               relationship: Relationship=Relationship.knows) -> str:
    """Return a string that represents the relationship between two people."""
    
    # notice we use the .value of the Relationship instance to get the string
    
    return f"{person_1} {relationship.value} {person_2}"

connection("Ron", "Harry", relationship=Relationship.bff)
```

    'Ron is best friends with Harry'



"Huh?", you say. "It looks to me like things just got more verbosified. I like making up words." 

Cool word, and you would be right, things did get more wordy. Moreover, adding all those annotations won't prevent someone from doing what we mentioned before,

```python
evil = lambda: 'power'
evil.value = 'corrupts'

connection("Ron", "Harry", relationship=evil) # this still works, though we don't want it to
```

... **unless**, that is, we run that code using - [**mypy**](http://mypy-lang.org/)


Mypy allows you to add type annotations and enforce them prior to running your program, so the only way the above function would run is if the relationship parameter was of type Relationship when called. Amazing!

Now, anyone reading our code would know exactly the types of things that could be passed as parameters to functions, and mypy will help us to enforce the type annotations we set. 

That makes our code much more legible and provides us and people using our code some nice sanity checks - knowing certain kinds of human errors will be caught prior to our code being run.

---

## Now, back to our original question about modeling 3 possible conditions.

1. Task ran without error. Data returned.
2. A known error occurred during task execution. No data returned.
3. A catastrophic runtime error occurred. No data returned

So, before we look at the code, it's cool to note that as of Python 3.6 we now have typed NamedTuples that we can declare using a new syntax in 3.6. We had this typed namedtuples in Python 3.5 but I think the new syntax is much cleaner.

```python

# old way
Employee = NamedTuple('Employee', [('name', str), ('id', int)])

# new sauce
class Employee(NamedTuple):
    name : str
    id   : int
```

Also, an Optional describes something that can be of a certain type, or None.

```python
possible_integer: Optional[int] = None # we could make this an integer later on
```

You'll see the use of optional values a lot in functional programming. In fact, the following pattern of combining tuples and optional values is the primary way we handle errors in Go. Optionals are also ubiquitous in Swift - not only in the context of error-handling. 

Optionals are a really handy concept, and now Python has them as well as a lot of the other type safety goodness of other languages through mypy.


```python
from typing import NamedTuple, Optional
import requests
import logging
import json
import enum


class ApiInteraction(enum.Enum):
    """The 3 possible states we can expect when interacting with the API."""
    SUCCESS = 1
    ERROR = 2
    FAILURE = 3


class ApiResponse(NamedTuple):
    """
    This is sort of a really dumbed-down version of an HTTP response,
    if you think of it in terms of status codes and response bodies.
    """
    status: ApiInteraction
    payload: Optional[dict]


        
def hit_endpoint(url: str) -> ApiResponse:
    """
    1. Send an http request to a url
    2. Parse the json response as a dictionary
    3. Return an ApiResponse object
    """
    
    
    try:
        
        response = requests.get(url) # step 1
        payload = response.json() # step 2
        
    except json.decoder.JSONDecodeError as e:
        
        # something went wrong in step 2; we knew this might happen
        
        # log a simple error message
        
        logging.error(f'could not decode json from {url}')
        
        # log the full traceback at a lower level
        
        logging.info(e, exc_info=True)
        
        # since we anticipated this error, make the
        # ApiResponse.status an ERROR as opposed to a failure
        
        return ApiResponse(ApiInteraction.ERROR, None)
    
    except Exception as e:
        
        # something went wrong in step 1 or 2 that
        # we couldn't anticipate
        
        # log the exception with the traceback
        
        logging.error(f"Something bad happened trying to reach {url}")
        logging.info(e, exc_info=True)
        
        # Since something catastrophic happened that
        # we didn't anticipate i.e. (DivideByBananaError)
        # we set the ApiResponse.status to FAILURE
        
        return ApiResponse(ApiInteraction.FAILURE, None)
    
    else:
        
        # Everything worked as planned! No errors!
        
        return ApiResponse(ApiInteraction.SUCCESS, payload)


# Python is awesome. We can either use the function by itself
# or use it as a constructor for our ApiResponse class 
# by doing thefollowing:


ApiResponse.from_url = hit_endpoint
```

```python
def test_endpoint_response():

    url = 'http://httpbin.org/headers'
    response = ApiResponse.from_url(url)
    assert response.status == ApiInteraction.SUCCESS
    assert response.status == hit_endpoint(url).status # our function and constructor work the same!
    assert response.payload is not None


    url = 'http://twitter.com'
    response = ApiResponse.from_url(url)
    assert response.status == ApiInteraction.ERROR
    assert response.status == hit_endpoint(url).status
    assert response.payload is None


    url = 'foo'
    response = ApiResponse.from_url(url)
    assert response.status == ApiInteraction.FAILURE
    assert response.status == hit_endpoint(url).status
    assert response.payload is None
        
    
test_endpoint_response()
```

    ERROR:root:could not decode json from http://twitter.com
    ERROR:root:could not decode json from http://twitter.com
    ERROR:root:Something bad happened trying to reach foo
    ERROR:root:Something bad happened trying to reach foo


## In collusion

I'm really excited about the way Python is evolving as a language, ecosystem, and community. I think that recent developments in typing add a lot to the expressiveness to the language and provide us with some really useful guarantees when leveraged through mypy.

Using **Enums** and **NamedTuples** this way not only provides us with some welcome sanity checks, I think it makes the code much more readable and **testable**. I also really like the idea of having relatively simple data structures that define a finite set of possible states that we can create arbitrary constructors for outside of their class definitions, similar to other languages like Go. This isn't something new to Python, but recent developments in Python's syntax and its ecosystem make patterns like these much more useful and intuitive, in my opinion.

Guido Van Rossum, our benevolent dictator, himself, has said innovations in Python's type system is what has him most excited about the language moving forward, according to his latest interview with on [Michael Kennedy's](https://twitter.com/mkennedy) [TalkPython.fm](https://talkpython.fm/episodes/show/100/python-past-present-and-future-with-guido-van-rossum). The more I read about these features and use them, the more I understand why.
