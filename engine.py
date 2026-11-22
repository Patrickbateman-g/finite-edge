"""Auditable strategy-capacity equations, no third-party dependencies.
Dollar ADV is per instrument, turnover is gross traded notional / NAV / day.
Returns are annual arithmetic excess returns, not CAGR or investment forecasts.
"""
from dataclasses import dataclass, asdict
from math import sqrt, isfinite

@dataclass(frozen=True)
class Scenario:
    alpha: float = .12
    daily_vol: float = .02
    adv: float = 20_000_000
    names: int = 20
    turnover: float = .10
    spread_bps: float = 5  # FULL quoted bid-ask spread
    fee_bps: float = 1
    impact: float = .5
    participation: float = .05
    days: float = 1
    half_life: float = 5

    def validate(self):
        for k,v in asdict(self).items():
            if not isfinite(v): raise ValueError(k+' must be finite')
        if not 0 < self.turnover <= 2: raise ValueError('turnover must be in (0,2]')
        if self.adv <= 0 or self.daily_vol <= 0 or self.impact <= 0: raise ValueError('positive liquidity, volatility and impact required')
        if self.names < 1 or int(self.names) != self.names: raise ValueError('names must be a positive integer')
        if not 0 < self.participation <= 1 or self.days < 1 or self.half_life <= 0: raise ValueError('invalid execution assumptions')
        if self.spread_bps < 0 or self.fee_bps < 0: raise ValueError('costs cannot be negative')


def coefficients(s):
    s.validate()
    # Uniform execution: average delay is (T-1)/2 trading days.
    gross = s.alpha * 2 ** (-(s.days-1)/(2*s.half_life))
    fixed = 252*s.turnover*(s.spread_bps/2+s.fee_bps)/10000
    # Turnover is a daily FLOW. With continuously overlapping daily orders,
    # stretching execution does NOT reduce steady-state daily participation.
    k = 252*s.turnover*s.impact*s.daily_vol*sqrt(s.turnover/(s.names*s.adv))
    return gross, fixed, k


def evaluate(s, capital):
    if not isfinite(capital) or capital < 0: raise ValueError('capital must be finite and nonnegative')
    gross, fixed, k = coefficients(s)
    impact = k*sqrt(capital)
    rate = gross-fixed-impact
    participation = capital*s.turnover/(s.names*s.adv)
    return dict(capital=capital,gross=gross,fixed=fixed,impact=impact,net=rate,
                profit=capital*rate,participation=participation,
                executable=participation<=s.participation)


def capacity(s):
    gross,fixed,k=coefficients(s)
    remaining=gross-fixed
    zero=(remaining/k)**2 if remaining>0 else 0
    operational=s.participation*s.names*s.adv/s.turnover
    optimum=min(4*zero/9,operational)
    return dict(break_even=zero,operational=operational,usable=min(zero,operational),
                optimum=optimum,profit=evaluate(s,optimum)['profit'],
                positive_edge=remaining>0)
