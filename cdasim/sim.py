#!/usr/bin/env python3
import argparse
import heapq
import itertools
import json
import math
import multiprocessing
import random
import sys


# Smallest float above 0. Should only effect numbers really close to 0. Allows
# division by 0 to be 0 without a conditional.
TINY = 2.2250738585072014e-308

class SummStats(object):
    def __init__(self, *initial_values):
        self.n = 0
        self.mean = 0
        self.squared_error = 0

        for val in initial_values:
            self.add(val)

    def add(self, val):
        self.n += 1
        delta = val - self.mean
        self.mean += delta / self.n
        self.squared_error += delta * (val - self.mean)
        return self

    @property
    def variance(self):
        if self.n > 1:
            return self.squared_error / (self.n - 1)
        else:
            return float('nan')

    @property
    def stddev(self):
        return math.sqrt(self.variance)


class Agent(object):
    def __init__(self, role, strategy, max_value, buyer, shading, markup, **_):
        self.role = role
        self.strategy = strategy
        self.buyer = buyer
        self.shading = shading
        self.value = random.random() * max_value
        self.submitted = False
        self.active = True
        self.surplus = 0

        # Set limit price
        if markup == 'standard':
            sign = 1 if buyer else -1
            self.price = self.value * (1 - sign * self.shading)
        elif markup == 'exponential':
            sign = -1 if buyer else 1
            self.price = self.value * math.exp(sign * self.shading)
        elif markup == 'shift':
            sign = 1 if buyer else -1
            self.price = self.value - sign * self.shading * max_value / 2
        elif markup == 'ideal':  # Contribution
            other_value = 0 if buyer else max_value
            self.price = (self.value * (1 - self.shading / 2) +
                          other_value * self.shading / 2)
        else:
            raise ValueError("Unknown markup " + markup)

    def execute(self, market):
        market.submit(self, self.buyer, self.price)
        self.submitted = True

    def transaction(self, price):
        self.active = False
        if self.buyer:
            self.surplus = self.value - price
        else:
            self.surplus = price - self.value

    def __repr__(self):
        return 'Agent({!r})'.format(self.strategy)


class Market(object):
    def __init__(self):
        self.bids = []
        self.asks = []
        self.time = 0

        self.num_trans = 0
        self.price_stats = SummStats()

    def _handle_matched(self, buy, sell, price):
        buy[2].transaction(price)
        sell[2].transaction(price)
        self.num_trans += 1
        self.price_stats.add(price)


class Continuous(Market):
    def submit(self, agent, buy, amount):
        if agent.submitted:
            collection = self.bids if buy else self.asks
            index = next(i for i, (_, __, a) in enumerate(collection)
                         if a == agent)
            if buy:
                collection[index] = (-amount, self.time, agent)
            else:
                collection[index] = (amount, self.time, agent)
            # This heapify is less efficient than necessary, we really just
            # need to fix the location, but it's all O(n) so...
            heapq.heapify(collection)

        else:
            if buy:
                own = self.bids
                other = self.asks
                sign = 1
            else:
                own = self.asks
                other = self.bids
                sign = -1

            order = (-sign * amount, self.time, agent)
            if other and other[0][0] <= sign * amount:
                matched = heapq.heappop(other)
                price = sign * matched[0]
                if buy:
                    self._handle_matched(order, matched, price)
                else:
                    self._handle_matched(matched, order, price)

            else:
                heapq.heappush(own, order)

        self.time += 1

    def clear(self):
        pass


class Discrete(Market):
    def submit(self, agent, buy, amount):
        order = (-amount if buy else amount, self.time, agent)
        if agent.submitted:
            collection = self.bids if buy else self.asks
            index = next(i for i, (_, __, a) in enumerate(collection)
                         if a == agent)
            collection[index] = order

        else:
            (self.bids if buy else self.asks).append(order)

        self.time += 1

    def clear(self):
        self.bids.sort()
        self.asks.sort()

        price = float('nan')
        for (bid, _, _), (ask, _, _) in zip(self.bids, self.asks):
            if -bid < ask:
                break
            price = (ask - bid) / 2

        for buy, sell in zip(self.bids, self.asks):
            if -buy[0] < sell[0]:
                break
            self._handle_matched(buy, sell, price)


def order_agents(agents, arrivals):
    buyers = []
    sellers = []

    for agent in agents:
        (buyers if agent.buyer else sellers).append(agent)
    random.shuffle(buyers)
    random.shuffle(sellers)

    if arrivals == 'simple':  # Contribution
        while buyers and sellers:
            yield (buyers if random.random() < 0.5 else sellers).pop()
        for agent in buyers:
            yield agent
        for agent in sellers:
            yield agent

    elif arrivals == 'actual':
        unsub_buyers = len(buyers)
        unsub_sellers = len(sellers)
        while buyers and sellers and (unsub_buyers or unsub_sellers):
            if random.random() < 0.5:
                collection = buyers
                other = sellers
            else:
                collection = sellers
                other = buyers
            agent = collection[-1]
            if agent.buyer:
                unsub_buyers -= not agent.submitted
            else:
                unsub_sellers -= not agent.submitted
            yield agent

            if not agent.active:  # Transacted
                collection.pop()
                other[:] = [o for o in other if o.active]

            elif len(collection) > 1:
                # No transaction, put back in random order
                swap = random.randrange(0, len(collection) - 1)
                collection[swap], collection[-1] = \
                    collection[-1], collection[swap]

    else:
        raise ValueError("Unknown arrival type: " + arrivals)


def observation(market, agents, spec):
    # Calculate total surplus of buyers and sellers
    players = {}
    buyers_values = []
    sellers_values = []
    ce_surplus = 0
    ce_price = 0
    ce_trans = 0
    surplus = 0
    buyer_surplus = 0
    seller_surplus = 0
    for agent in agents:
        players.setdefault((agent.role, agent.strategy),
                           []).append(agent.surplus)
        surplus += agent.surplus
        if agent.buyer:
            buyers_values.append(agent.value)
            buyer_surplus += agent.surplus
        else:
            sellers_values.append(agent.value)
            seller_surplus += agent.surplus

    # From values compute social optimum (ce) values
    buyers_values.sort(reverse=True)
    sellers_values.sort()

    for buy_value, sell_value in zip(buyers_values, sellers_values):
        if sell_value > buy_value:
            break
        ce_trans += 0.5 if buy_value == sell_value else 1
        ce_surplus += buy_value - sell_value
        ce_price = (buy_value + sell_value) / 2

    # Calculate types of in efficiency, can use value relative to ce_price to
    # determine if agent is intra- or extra-marginal.
    v_ineff = 0
    em_ineff = 0
    for agent in agents:
        if agent.buyer:
            if agent.active and agent.value > ce_price:
                v_ineff += agent.value - ce_price
            elif not agent.active and agent.value < ce_price:
                em_ineff += ce_price - agent.value
        else:
            if agent.active and agent.value < ce_price:
                v_ineff += ce_price - agent.value
            elif not agent.active and agent.value > ce_price:
                em_ineff += agent.value - ce_price

    spec['players'] = players
    spec['features'] = dict(
        surplus=surplus,
        ce_surplus=ce_surplus,
        efficiency=surplus / (ce_surplus + TINY),
        trans_ratio=market.num_trans / (ce_trans + TINY),
        price_ratio=market.price_stats.mean / (ce_price + TINY),
        buyer_surplus=buyer_surplus,
        buyer_efficiency=buyer_surplus / (ce_surplus + TINY),
        seller_surplus=seller_surplus,
        seller_efficiency=seller_surplus / (ce_surplus + TINY),
        v_surplus=v_ineff,
        v_ineff=v_ineff / (ce_surplus + TINY),
        em_surplus=em_ineff,
        em_ineff=em_ineff / (ce_surplus + TINY),
        price_dispersion=market.price_stats.stddev / (ce_price + TINY),
    )
    return spec


def execute(spec):
    config = spec['configuration']

    agents = []
    for role, strategies in spec['assignment'].items():
        for strategy, count in strategies.items():
            player_strat = dict(zip(['type', 'shading', 'markup'],
                                    strategy.lower().split('_')))
            strats = config.copy()
            strats.update(player_strat)
            strats['max_value'] = float(strats['max_value'])
            strats['shading'] = float(strats['shading'])
            strats['buyer'] = strats['type'].lower() == 'b'

            for _ in range(count):
                agents.append(Agent(role, strategy, **strats))

    market = (Continuous() if config['market'].lower() == 'cda'
              else Discrete())

    for agent in order_agents(agents, config['arrivals']):
        agent.execute(market)
    market.clear()

    return observation(market, agents, spec)


def iter_repeat(iterable, count):
    return itertools.chain.from_iterable(itertools.repeat(elem, count)
                                         for elem in iterable)


def output_function(sims_per_obs):
    num_processed = [0]
    summ_players = {}
    summ_features = {}
    base = {}

    def output(result):
        if num_processed[0] == 0:
            summ_players.update({rs: [SummStats(p) for p in payoffs]
                                 for rs, payoffs in result['players'].items()})
            summ_features.update({feat: SummStats(val) for feat, val
                                  in result['features'].items()})
            base.update(result)
        else:
            for rs, payoffs in result['players'].items():
                for agg, payoff in zip(summ_players[rs], payoffs):
                    agg.add(payoff)
            for feat, val in result['features'].items():
                summ_features[feat].add(val)

        num_processed[0] += 1
        if num_processed[0] == sims_per_obs:
            base['players'] = list(
                itertools.chain.from_iterable(
                    ({'role': rs[0], 'strategy': rs[1], 'payoff': p.mean}
                     for p in payoffs)
                    for rs, payoffs in summ_players.items()))
            base['features'] = {feat: val.mean for feat, val
                                in summ_features.items()}
            json.dump(base, sys.stdout)
            sys.stdout.write('\n')
            sys.stdout.flush()

            num_processed[0] = 0
            summ_players.clear()
            summ_features.clear()
            base.clear()

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'num_obs', metavar='num-obs', type=int, help="""Number of observations
        to gather.""")
    parser.add_argument(
        '--single', action='store_true', help="""Run the simulation in a single
        process.""")
    parser.add_argument(
        '--sims-per-obs', metavar='<sims>', type=int, default=1, help="""Number
        of simulations to run per observation""")
    args = parser.parse_args()

    inp = iter_repeat(map(json.loads, sys.stdin), args.num_obs * args.sims_per_obs)
    outp = output_function(args.sims_per_obs)
    if args.single:
        for spec in inp:
            outp(execute(spec))
    else:
        with multiprocessing.Pool() as pool:
            for result in pool.imap(execute, inp):
                outp(result)


if __name__ == '__main__':
    main()
