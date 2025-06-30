import itertools
import typing as ty
from collections import abc


def powerset(iterable: abc.Iterable[ty.Any]) -> ty.Iterator[ty.Tuple]:
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(xs, n) for n in range(len(xs) + 1)
    )
