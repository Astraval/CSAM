def inflate_multiplier(multiplier: float):
    def inflate_func_mul(last_volume: float) -> float:
        return multiplier * last_volume

    return inflate_func_mul


def narrower_halver(half: float) -> float:
    def narrower_func_halver(interval_a: float, interval_b: float) -> float:
        return (interval_b - interval_a) * half + interval_a

    return narrower_func_halver
