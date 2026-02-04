import numpy as np


class LongstaffSchwartzPricer:

    def __init__(
        self,
        model,
        payoff,
        risk_free_rate
    ):
        self.model = model
        self.payoff = payoff
        self.r = risk_free_rate

    def price(
        self,
        time,
        steps,
        paths,
        seed=None
    ):
        raise NotImplementedError