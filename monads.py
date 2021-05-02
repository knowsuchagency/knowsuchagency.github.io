"""
A basic implementation of the identity monad in python.

We use the hypothesis library to test that our monad obeys
the monad, applicative, and functor laws.

Credit to the following publications for definitions, knowledge, and inspiration:
* https://mmhaskell.com/monads/laws
* https://fsharpforfunandprofit.com/posts/elevated-world/
* https://homepages.inf.ed.ac.uk/wadler/topics/monads.html
"""
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial, lru_cache
from typing import *

import hypothesis.strategies as st
from hypothesis import given, infer

Scalar = Union[AnyStr, int, bool, float]

Function = Callable[[Scalar], Scalar]

ScalarOrFunction = Union[Scalar, Function]

UnaryFunction = Callable[[ScalarOrFunction], ScalarOrFunction]

BinaryFunction = Callable[
    [ScalarOrFunction, ScalarOrFunction], ScalarOrFunction
]

UnaryOrBinaryFunction = Union[UnaryFunction, BinaryFunction]


class Functor(ABC):
    """
    A functor is a thing with a map method that obeys a set of rules
    as to how its map method will behave.
    """

    @abstractmethod
    def map(self, function: UnaryFunction) -> "Functor":
        raise NotImplementedError


class Applicative(Functor):
    """
    An applicative is a functor with an apply method
    that follows a set of rules.

    Applicatives also a unit method which will take a normal function or
    value which it will "lift" the value/function into a context/effect.

    We can think sort of think of unit as just a glorified __init__
    """

    @classmethod
    @abstractmethod
    def unit(cls, value: Any) -> "Applicative":
        raise NotImplementedError

    @abstractmethod
    def apply(self, lifted: "Applicative") -> "Applicative":
        raise NotImplementedError


class Monad(Applicative):
    """
    A monad can be thought of as a container that obeys a set of laws.
    """

    @abstractmethod
    def bind(self, function: Callable[[Scalar], "Monad"]) -> "Monad":
        raise NotImplementedError


@dataclass
class Identity(Monad):
    """
    The identity monad.

    The simplest of monads. It does nothing but wrap a value.
    """

    value: Any

    @classmethod
    def unit(cls, value: Any) -> "Identity":
        return unit(value, cls)

    def map(self, function: UnaryFunction) -> "Identity":
        return map(self, function)

    def apply(self, lifted: "Identity") -> "Identity":
        return apply(self, lifted)

    def bind(self, function: Callable[[Scalar], "Identity"]) -> "Identity":
        return bind(self, function)

    def __eq__(self, other: Any):
        if not isinstance(other, Identity):
            return False
        elif self.value is other.value:
            return True
        elif self.value == other.value:
            return True
        elif callable(self.value) and callable(other.value):
            # in the event we need to compare functions,
            # we assume both functions accept 0 for simplicity's sake
            return self.value(0) == other.value(0)
        else:
            return False


# since monads are applicatives which are in-turn functors, from here on I will be defining functions
# solely in terms of monads because my goal is to make this easy to read and understand


def unit(
    value: Union[Scalar, UnaryFunction], M: Type[Monad] = Identity
) -> Monad:
    """
    AKA: return, pure, yield, point

    The purpose of `unit` is to take a value, and "lift" it into a context.
    """
    return M(value) if not isinstance(value, M) else value


def map(monad: Monad, function: UnaryFunction) -> Monad:
    """
    AKA: map, fmap, lift, Select

    Given a monad and a function, return a new monad where the function is applied to the monad's value.

    Note, it's normally bad practice to define control flow using exceptions, but due to Python's dynamic
    nature, we aren't guaranteed to have the necessary type information up-front in order to know whether
    we need to simply apply the function to the lifted value, partially apply two functions, or compose them.
    """

    try:
        return monad.unit(function(monad.value))
    except TypeError:
        return monad.unit(partially_apply_or_compose(function, monad.value))


def apply(lifted_function: Monad, monad: Monad) -> Monad:
    """
    AKA: ap, <*>

    `apply` takes a monad and a function lifted in a monad
    and applies the lifted function to the value in the
    other monad
    """

    return map(monad, lifted_function.value)


def bind(monad: Monad, function: Callable[[Scalar], Monad]) -> Monad:
    """
    AKA: flatMap, andThen, collect, SelectMany, >>=, =<<

    `bind` takes a monad and a function and applies that function to the monad.
    The function expects a normal value and returns a monad
    """
    return map(monad, function)


# ---- tests ---- #


@st.composite
def monads(draw):
    """Build us some monads, would you?"""

    scalars = st.integers()

    unary_functions = st.functions(like=lambda x: x, returns=scalars)

    value = draw(st.one_of(scalars, unary_functions))

    value = value if not callable(value) else memoize(value)

    return Identity(value)


@given(integer=st.integers(), f=infer, g=infer)
def test_functor_laws(
    integer: int, f: UnaryOrBinaryFunction, g: UnaryOrBinaryFunction
):
    """
    fmap id  ==  id
    fmap (f . g)  ==  fmap f . fmap g
    """

    # we will continue to see this pattern where we use our memoization function
    # to make the functions hypothesis generates behave predictably

    f, g = memoize(f), memoize(g)

    monad = unit(integer)

    # I'll put the regular function invocation form prior to the method
    # invocation form of each law from now on.
    # I find that sometimes one is more readable than the other

    """
    identity
    
        fmap id = id
    """

    assert map(monad, identity) == monad

    assert monad.map(identity) == monad

    """
    composition
    
        fmap (g . f) = fmap g . fmap f
    """

    assert map(unit(integer), compose(f, g)) == map(map(unit(integer), g), f)

    assert unit(integer).map(compose(f, g)) == unit(integer).map(g).map(f)


@given(monad=monads(), value=infer, f=infer, g=infer)
def test_monad_laws(
    monad: Monad,
    value: Scalar,
    f: UnaryOrBinaryFunction,
    g: UnaryOrBinaryFunction,
):
    r"""
    return a >>= f = f
    m >>= return = m
    (m >>= f) >>= g = m >>= (\x -> f x >>= g)
    """
    f, g = _memoize_and_monadify(f), _memoize_and_monadify(g)

    """
    left identity
    
        return a >>= f = f
    """

    assert bind(unit(value), f) == f(value)

    assert unit(value).bind(f) == f(value)

    """
    right identity
    
        m >>= return = m
    """

    assert bind(monad, unit) == monad

    assert monad.bind(unit) == monad

    r"""
    associativity
    
        (m >>= f) >>= g = m >>= (\x -> f x >>= g)
    """

    assert bind(bind(monad, f), g) == bind(monad, lambda x: bind(f(x), g))

    assert monad.bind(f).bind(g) == monad.bind(lambda x: bind(f(x), g))


@given(
    monad=monads(),
    integer=st.integers(),
    f=infer,
    g=infer,
    u=infer,
    v=infer,
    w=infer,
)
def test_applicative_laws(
    monad: Monad,
    integer: int,
    f: UnaryOrBinaryFunction,
    g: UnaryOrBinaryFunction,
    u: UnaryOrBinaryFunction,
    v: UnaryOrBinaryFunction,
    w: UnaryOrBinaryFunction,
):
    """
    pure id <*> v = v
    pure f <*> pure x = pure (f x)
    u <*> pure y = pure ($ y) <*> u
    pure (.) <*> u <*> v <*> w = u <*> (v <*> w)

    oof
    """

    f, g, u, v, w = (memoize(func) for func in (f, g, u, v, w))

    """
    identity
    
        pure id <*> v = v
    """

    assert apply(unit(identity), monad) == monad

    assert unit(identity).apply(monad) == monad

    """
    homomorphism

        pure f <*> pure x = pure (f x)
    """
    x = integer

    assert apply(unit(f), unit(x)) == unit(f(x))

    assert unit(f).apply(unit(x)) == unit(f(x))

    """
    interchange

        u <*> pure y = pure ($ y) <*> u
    """

    y = integer

    assert apply(unit(f), unit(y)) == apply(unit(lambda g: g(y)), unit(f))

    assert unit(f).apply(unit(y)) == unit(lambda g: g(y)).apply(unit(f))

    """
    composition

        pure (.) <*> u <*> v <*> w = u <*> (v <*> w)
    """

    u, v, w = unit(u), unit(v), unit(w)

    assert apply(apply(apply(unit(compose), u), v), w) == apply(u, apply(v, w))

    assert unit(compose).apply(u).apply(v).apply(w) == u.apply(v.apply(w))


def _memoize_and_monadify(function: UnaryFunction):
    """Memoize function and wrap its return value in a monad."""

    @memoize
    def f(x):

        return unit(function(x))

    return f


def memoize(func):
    """
    Since functions generated by hypothesis aren't deterministic, this decorator will allow us to make
    them so.
    """
    return lru_cache(maxsize=None)(func)


def identity(x: Any) -> Any:
    return x


def partially_apply_or_compose(
    f: UnaryOrBinaryFunction, g: UnaryFunction
) -> Callable:
    """If the arity of f is greater than one, return a function with g partially applied to f else compose f and g."""
    if len(inspect.signature(f).parameters) > 1:
        # f is probably the composition function
        return partial(f, g)
    else:
        return compose(f, g)


def compose(f: Callable, g: Callable) -> Callable:
    def f_after_g(x):
        return f(g(x))

    return f_after_g


def test():
    test_functor_laws()
    test_monad_laws()
    test_applicative_laws()


if __name__ == "__main__":
    test()
