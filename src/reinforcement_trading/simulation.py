import math
import BSE

# set up parameters for the session

start_time = 0.0
end_time = 600.0
duration = end_time - start_time


# schedule_offsetfn returns time-dependent offset on schedule prices
def schedule_offsetfn(t):
    pi2 = math.pi * 2
    c = math.pi * 3000
    wavelength = t / c
    gradient = 100 * t / (c / pi2)
    amplitude = 100 * t / (c / pi2)
    offset = gradient + amplitude * math.sin(wavelength * t)
    return int(round(offset, 0))



# #        range1 = (10, 190, schedule_offsetfn)
# #        range2 = (200,300, schedule_offsetfn)

# #        supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
# #                            {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
# #                            {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
# #                          ]



range1 = (95, 95, schedule_offsetfn)
supply_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
                    ]

range1 = (105, 105, schedule_offsetfn)
demand_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
                    ]

order_sched = {'sup':supply_schedule, 'dem':demand_schedule,
               'interval':30, 'timemode':'drip-poisson'}


# Traders other than us
def add_other_traders(traders):
    buyers_spec = [('GVWY', 4), ('SHVR', 4),
                   ('ZIC', 4), ('ZIP', 4)]
    sellers_spec = buyers_spec
    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
    trader_stats = BSE.populate_market(traders_spec, traders, True, True)
    return trader_stats


class LoggerTrader(BSE.Trader):
    def __init__(self, tid):
        super(LoggerTrader, self).__init__('LoggerTrader', tid, 0)
        self.log = []

    def respond(self, time, lob, trade, verbose):
        self.log.append((time, lob.copy(), trade))

    def getorder(self, time, time_left, lob):
        return None


# Simulation
def market_data():
    logger = LoggerTrader('L01')
    traders = {'L01': logger}
    trader_stats = add_other_traders(traders)
    BSE.market_session('trial_id', start_time, end_time, traders, trader_stats,
                       order_sched, False)
    return logger.log
