"""
This file provides useful methods that can be used to increase the safe box volume during multiphase training.
"""


def inflate_multiplier(multiplier: float):
    """Returns a function that inflates the last volume by multiplying it with a given multiplier.

    Args:
        multiplier (float): The factor by whitch we multiply the last volume of the safe box.
    
    Returns:
        inflate_func_mul : A function that takes the last volume and returns the inflated volume.
    """
    def inflate_func_mul(last_volume: float) -> float:
        return multiplier * last_volume

    return inflate_func_mul


def narrower_halver(half: float):
    """Returns a function that narrows the interval between two volumes by moving a fraction (half) of the way from interval_a to interval_b.

    Args:
        half (float): The fraction of the interval to move towards interval_b (for example 0.5 if we want the half).

    Returns:
        narrower_func_halver: A function that takes two interval boundaries and returns the narrowed value.
    """
    def narrower_func_halver(interval_a: float, interval_b: float) -> float:
        return (interval_b - interval_a) * half + interval_a

    return narrower_func_halver
