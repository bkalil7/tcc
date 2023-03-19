import math
import numpy as np
import pandas as pd
from tqdm import tqdm

periods = pd.read_csv("distance_data/Periods.csv", header=None)
Pt = pd.read_csv("distance_data/Pt_formatted.csv", header=0)
Rt = pd.read_csv("distance_data/Rt_formatted.csv", header=0)
Rm = pd.read_csv("distance_data/Rm.csv", header=None)
Rf = pd.read_csv("distance_data/Rf.csv", header=None)
Vt = pd.read_csv("distance_data/Vt.csv", header=None)
ticker2 = pd.read_csv("distance_data/ticker2.csv", header=None)
ticker_b = pd.read_csv("distance_data/ticker_b.csv", header=None)

days, num_assets = np.shape(Rt)  # days = T in matlab code | num_assets = N

print(f"Days: {days} | Assets: {num_assets}")

daylag = 0
wi_update = 1
years = 2015 - 1990 + 0.5

no_pairs = 20
trading_costs = 0
# buy/sell (percentage cost for opening and closing pairs: 0.001, 0.002, for example)
percentage_costs = 0
# set whether (0) or not (2) positive trading volume is required for opening/closing a pair
trade_req = 0
# Choose how much loss we are willing to accePt on a given pair, compared to 1, i.e, 0.93 = 7% stop loss
Stop_loss = 0.95

if Stop_loss != float('-inf'):
    stop_dir = 100 - (Stop_loss * 100)
# Choose how much gain we are willing to accePt on a given pair, compared to 1, i.e 1.10 = 10% stop gain
Stop_gain = float('inf')
s1218 = 1  # listing req. (look ahead): 12+6 months (=1)

opening_threshold = 1.5
closing_threshold = 0.75

duration_limit = float('inf')

avg_price_dev = np.zeros(
    (days-sum(periods.iloc[0:2, 0].to_list()), no_pairs*2))
# 12 months are w/o price deviations. The first 12 months are formation period

# Operations array
operations = []

# Keeps track of the return of each pair
first_traininig = int(periods.iloc[0, 3])
Rpair = np.zeros((days-first_traininig, no_pairs))
Rp_ew_cc = pd.DataFrame(np.zeros((days-first_traininig, 2)),
                        columns=['Return', 'Semester'])
Rp_vw_fi = pd.DataFrame(np.zeros((days-first_traininig, 2)),
                        columns=['Return', 'Semester'])
ret_acum_df = pd.DataFrame(
    np.zeros((days-first_traininig, 4)), columns=['CC', 'FI', 'RMRF', 'SEMESTER'])
RmRf = np.zeros((days-first_traininig, 1))
risk_free = pd.DataFrame(np.zeros((days-first_traininig, 2)),
                        columns=['Return', 'Semester'])

# TODO
MDDpairs = np.zeros((np.max(np.shape(periods))-1, no_pairs))
# Pre allocate Maximum Drawdown (MDD) matrix for each pair out of sample;
MDDw = np.zeros((np.max(np.shape(periods))-1, 5))
# Preallocate MDD matrix out of sample for 4 weighting schemes + Rm + (Rm-Rf).
Sortino_pairs = np.zeros((np.max(np.shape(periods))-2, no_pairs))
Sortino_w = np.zeros((np.max(np.shape(periods))-2, 5))

periods_with_open_pair = 0  # number of periods with pairs opened
periods_without_open_pair = 0  # number of periods without pairs opened
pairs_number = 0
pair_open = 0
days_open = np.zeros((no_pairs*10000, 1))
# measures number of days each pair open; bad programming, but we do not know how many pairs we get
# measures number of times pairs opened in each pair per 6 month period
no_pairs_opened = np.zeros((int(years*2-2), no_pairs))

counter = 0  # Keeps track of the days in the main loop

# ----------------------------------------------------
# Start of Main Loop - Creating Price Index
# ----------------------------------------------------
# Main part of the program starts here
# ----------------------------------------------------

big_loop = 0
i = 0

while big_loop < (years * 2 - 2):
    twelve_months = int(periods.iloc[big_loop, 3])
    six_months = int(periods.iloc[big_loop + 2, 0])

    print(
        f"big_loop: {big_loop}, i: {i}, twelve_months: {twelve_months}, six_months: {six_months}")

    # ----------------------------------------------------
    # Create price index IPt by setting first Pt>0 to 1
    # ----------------------------------------------------

    # Preallocate a zeros matrix with the size of the Formation + Trading period
    # IPt = Indexed Price at time t
    IPt = np.zeros((int(twelve_months + six_months), num_assets))

    print("Generating Assets Price Index")

    # print(f"Num assets: {num_assets}")
    for j in tqdm(range(0, num_assets)):
        m = 0
        for i2 in range(0, int(twelve_months+six_months)):  # same here
            if not math.isnan(Pt.iloc[i+i2, j]) and m == 0:
                IPt[i2, j] = 1
                m = 1
            elif not math.isnan(Pt.iloc[i+i2, j]) and m == 1:
                IPt[i2, j] = IPt[i2-1, j] * (1 + Rt.iloc[i+i2, j])
                # if j == 408:
                #    print(f"Return value at day {i2}: {Rt.iloc[i+i2-1, j]}")
                #    print(f"IPT value at day {i2}: {IPt[i2, j]}")

    pd.DataFrame(IPt).to_csv("IPt.csv", header=None,
                             index=False, na_rep='NULL')

    listed1 = IPt[0, :] > 0  # Listed at the beginning (1xN vector of booleans)
    # listed at the end: 12/18 months from now (1xN vector of booleans)
    listed2 = IPt[int(twelve_months+six_months*(s1218 == 1))-1, :] > 0
    listed = np.multiply(listed1, listed2)

    listed_num = np.sum(listed)
    listed_indexes = np.where(listed > 0)[0]
    # print(f"listed_indexes: {listed_indexes}")
    listed_stocks = Pt.columns[listed_indexes]
    # print(f"Listed stockes: {listed_stocks}")

    [D, ia, ib] = np.intersect1d(
        ticker2.iloc[:, big_loop], ticker2.iloc[:, big_loop+1], return_indices=True)

    # como a variável id no matlab não é utilizada, deixei ela de fora do script em python
    ic = np.isin(D, ticker2.iloc[:, big_loop+2])
    # a função np.isin não retorna o valor, então teria que buscar outra alternativa
    # caso necessário

    Dic_unique_sorted, B_idx = np.unique(D[ic], return_index=True)

    listed_union = np.intersect1d(listed_stocks, Dic_unique_sorted)
    # print(f"Listed union len {np.shape(listed_union)} value {listed_union}")

    index_listed2 = [Pt.columns.get_loc(i) for i in listed_union if i in Pt]
    index_listed2.sort()
    # print(f"index_listed2: {index_listed2}")

    no_listed2 = len(index_listed2)
    print(f"no_listed2: {no_listed2}")

    # ----------------------------------------------------
    # Add filters (if needed)
    # ----------------------------------------------------
    # e.g. remove if liquidity below value X, the second listed stock series etc.
    # ----------------------------------------------------
    # Desc stat of the price series
    # ----------------------------------------------------
    no_comp = np.transpose(sum(np.transpose(IPt > 0)))

    print(f'Period {big_loop}')
    # print(f'Time series mean no of stock series {np.mean(no_comp)}')
    # print(f'Max number of stock series {max(no_comp)}')
    # print(f'Min number of stock series {min(no_comp)}')

    # ----------------------------------------------------
    # Calc SSEs
    # ----------------------------------------------------

    sse = np.zeros((no_listed2, no_listed2))
    for j in tqdm(range(0, no_listed2 - 1)):
        for k in range(j+1, no_listed2):
            sse[j, k] = sum(np.power(IPt[0:int(
                twelve_months), index_listed2[j]]-IPt[0:int(twelve_months), index_listed2[k]], 2))

    print(f"SSE shape: {np.shape(sse)}")
    pd.DataFrame(sse).to_csv("SSE.csv", header=[
        str(i) for i in index_listed2], index=False)

    # ----------------------------------------------------
    # Find min SSEs
    # ----------------------------------------------------

    max_SSE = np.nanmax(sse) + 1
    min_SSE = np.zeros((no_pairs, 1))
    pairs = []
    min_SSE_ro = np.zeros((1, no_pairs))
    min_SSE_co = np.zeros((1, no_pairs))

    # print(f"Initial Max SSE: {max_SSE}")

    for ii in range(0, no_pairs):
        t_SSE = max_SSE
        for k in range(0, no_listed2-1):
            for l in range(k+1, no_listed2):
                if sse[k, l] > 0 and sse[k, l] < t_SSE:
                    # print(f"New minimum found at ({k},{l})")
                    t_SSE = sse[k, l]  # new minimum found

        # print(f"Minimum SSE = {t_SSE}")

        if t_SSE == max_SSE:
            print("Error")

        ro, co = np.where(sse == t_SSE)
        ro = ro[0]
        co = co[0]
        # print(f"Indexes: ({ro},{co}) Pair: {Pt.columns[index_listed2[ro]]}-{Pt.columns[index_listed2[co]]}")
        min_SSE[ii, 0] = sse[ro, co]
        # print(f"Col: {ro} | index_listed: {index_listed2[ro]} | ticker: {Pt.columns[index_listed2[ro]]} | other ticker: {Pt.columns[ro]}")
        pairs.append({"s1_col": int(ro), "s2_col": int(co),
                     "s1_ticker": Pt.columns[index_listed2[ro]], "s2_ticker": Pt.columns[index_listed2[co]]})
        sse[ro, co] = max_SSE  # prevent re-selection

    # pd.DataFrame(min_SSE_ro).to_csv("min_SSE_ro.csv", header=None, index=False)
    # pd.DataFrame(min_SSE_co).to_csv("min_SSE_co.csv", header=None, index=False)

    # print(f"min_SSE_ro variable: {min_SSE_ro}")
    # print(f"min_SSE_co variable: {min_SSE_co}")
    # print(f"min_SSE len: {np.shape(min_SSE)} | variable: {min_SSE}")

    # ----------------------------------------------------
    # Calculate returns during the 6 month period
    # ----------------------------------------------------

    count_temp = counter

    print(f"counter value: {counter} | counter_temp = {count_temp}")

    print(
        f"Portfolio period from days {i+twelve_months} to {i+twelve_months+six_months-1}")
    for p in range(0, no_pairs):
        first_col = index_listed2[pairs[p]['s1_col']]
        second_col = index_listed2[pairs[p]['s2_col']]
        counter = count_temp
        pairs_opened = 0
        new_pairs_opened = 0
        lag = 0
        can_trade = True
        last_operation = 0


        std_limit = np.std(IPt[0:twelve_months, first_col] -
                           IPt[0:twelve_months, second_col])  # standard deviation

        print(f"Std limit: {std_limit}")
        spread = IPt[0:twelve_months, first_col] - IPt[0:twelve_months, second_col]
        #print(f"Spread series: {spread}")

        # Fixed volatility estimated in the 12 months period. Doing the calculation one pair at a time
        # Presets all variables for each pair
        Rcum = np.zeros((twelve_months, 1))
        counter_ret = 1
        Rcum_ret = [1]

        wi = []

        for j in range(i+twelve_months, i+twelve_months+six_months-1):  # portfolio period
            # Defining the period as from the first day of the twe_month to the last day of the twe_month

            """ if Rcum_ret[-1] < 0.85:
                print(f"pair_no {p}")
                print(f"pairs_opened: {pairs_opened}")
                print(f"pair indexes: ({first_col},{second_col})")
                break """

            if daylag == 0:  # w/o one day delay
                if not can_trade:
                    if (last_operation == 1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) <= closing_threshold*std_limit) or (last_operation == -1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) >= -closing_threshold*std_limit):
                        can_trade = True

                if pairs_opened == -1:  # pairs opened: long 1st, short 2nd stock
                    # print("Pair is long on 1st")
                    # If a sign to open has been given, then calcule the returns
                    """ print(f"Weights: w1 {wi[0]} w2 {wi[1]}")
                    print(f"Return indexes {j},{second_col} | Returns r1 {Rt.iloc[j, first_col]} | r2 {Rt.iloc[j, second_col]}")
                    print(f"Calculated return {np.multiply(Rt.iloc[j, first_col], wi[0]) - np.multiply(Rt.iloc[j, second_col], wi[1])}")
                     """
                    Rpair[counter, p] = np.multiply(
                        Rt.iloc[j, first_col], wi[0]) - np.multiply(Rt.iloc[j, second_col], wi[1])
                    # Rpair is the return of each pair.
                    Rcum[counter_ret, 0] = Rpair[counter, p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    lag = lag + 1  # used for paying tc

                    if wi_update == 1:  # The weight of each asset in the pair is updated.
                        wi[0] = wi[0]*(1+Rt.iloc[j, first_col])
                        wi[1] = wi[1]*(1+Rt.iloc[j, second_col])

                        # print(f"Updated weights w1 {wi[0]} - w2 {wi[1]}")

                elif pairs_opened == 1:  # pairs opened: short 1st, long 2nd stock
                    # print("Pair is short on 1st")
                    """ print(f"Weights: w1 {wi[0]} w2 {wi[1]}")
                    print(
                        f"Return indexes {j},{second_col} | Returns r1 {Rt.iloc[j, first_col]} - weighted {np.multiply(Rt.iloc[j, first_col], wi[0])} r2 {Rt.iloc[j, second_col]} - weighted {np.multiply(Rt.iloc[j, second_col], wi[1])}")
                    print(
                        f"Calculated return {np.multiply(-Rt.iloc[j, first_col], wi[0]) + np.multiply(Rt.iloc[j, second_col], wi[1])}")
                     """
                    Rpair[counter, p] = np.multiply(
                        -Rt.iloc[j, first_col], wi[0]) + np.multiply(Rt.iloc[j, second_col], wi[1])
                    # print(f"Rpair value {Rpair[counter, p]}")
                    Rcum[counter_ret, 0] = Rpair[counter, p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    lag = lag + 1

                    if wi_update == 1:
                        wi[0] = wi[0]*(1+Rt.iloc[j, first_col])
                        wi[1] = wi[1]*(1+Rt.iloc[j, second_col])
                        # print(f"Updated weights w1 {wi[0]} - w2 {wi[1]}")

                else:
                    Rpair[counter, p] = 0  # closed (this code not necessary)

                if ((pairs_opened == 1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) <= closing_threshold*std_limit) or (counter_ret > duration_limit) or (Rcum_ret[-1] < Stop_loss) or (Rcum_ret[-1] >= Stop_gain)
                        or (pairs_opened == -1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) >= -closing_threshold*std_limit)) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    #print(f"Closing position {pairs_opened}. Price diff: {IPt[j-i, first_col]-IPt[j-i, second_col]}")
                    
                    converged = True

                    if Rcum_ret[-1] < Stop_loss:
                        Rcum_ret[-1] = Stop_loss
                        converged = False
                        can_trade = False
                        last_operation = pairs_opened

                    if counter_ret > duration_limit:
                        converged = False
                        can_trade = False
                        last_operation = pairs_opened

                    pairs_opened = 0  # close pairs: prices cross
                    # when pair is closed reset lag (it serves for paying tc)
                    lag = 0
                    # add a marker for closing used to calc length of the "open-period"
                    avg_price_dev[counter, no_pairs+p] = 1
                    # Includes trading cost in the last day of trading, due to closing position
                    Rpair[counter, p] = Rpair[counter, p] - percentage_costs

                    print(
                        f"Pair {pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']} closed. Days: {counter_ret} | Return: {Rcum_ret[-1]}")
                    operations.append({
                        "Semester": big_loop,
                        "Days": counter_ret,
                        "S1": pairs[p]['s1_ticker'],
                        "S2": pairs[p]['s2_ticker'],
                        "Pair": f"{pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']}",
                        "Return": Rcum_ret[-1],
                        "Converged": converged
                    })

                    counter_ret = 1

                elif can_trade and (pairs_opened == 0) and (+IPt[j-i, first_col]-IPt[j-i, second_col] > opening_threshold*std_limit) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    #print(f"Opening short position. Price diff: {IPt[j-i, first_col]-IPt[j-i, second_col]} | Threshold: {1.5*std_limit}")
                    if pairs_opened == 0:  # record dev (and time) at open
                        Rcum = np.zeros((six_months, 1))
                        counter_ret = 1
                        avg_price_dev[counter, p] = 2*(+IPt[j-i, first_col] - IPt[j-i, second_col])/(
                            IPt[j-i, first_col] + IPt[j-i, second_col])

                    # print(
                    #    f"Trading short pair {pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']} on day {j}")
                    pairs_opened = 1  # open pairs
                    lag = lag + 1  # - Lag was 0. On the next loop C will be paid
                    wi = [1, 1]

                elif can_trade and (pairs_opened == 0) and (IPt[j-i, first_col]-IPt[j-i, second_col] < -opening_threshold*std_limit) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    #print(f"Opening long position. Price diff: {IPt[j-i, first_col]-IPt[j-i, second_col]} | Threshold: {-1.5*std_limit}")
                    if pairs_opened == 0:  # record dev (and time) at open
                        Rcum = np.zeros((six_months, 1))
                        counter_ret = 1
                        avg_price_dev[counter, p] = 2*(-IPt[j-i, first_col] + IPt[j-i, second_col])/(
                            IPt[j-i, first_col] + IPt[j-i, second_col])
                    # print(
                    #    f"Trading long pair {pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']} on day {j}")
                    pairs_opened = -1  # open pairs
                    lag = lag + 1
                    wi = [1, 1]

                counter += 1

            elif daylag == 1:
                if pairs_opened == -1:  # pairs opened: long 1st, short 2nd stock
                    Rpair[counter, p] = (+Rt.iloc[j, first_col] * wi[0] -
                                         Rt.iloc[j, second_col] * wi[1]) - (lag == 2)*trading_costs
                    Rcum[counter_ret, 0] = Rpair[counter, p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    if wi_update == 1:
                        wi[0] = wi[0]*(1+Rt.iloc[j, first_col])
                        wi[1] = wi[1]*(1+Rt.iloc[j, second_col])

                elif pairs_opened == 1:  # pairs opened: short 1st, long 2nd stock
                    Rpair[counter, p] = (-Rt.iloc[j, first_col] * wi[0] +
                                         Rt.iloc[j, second_col] * wi[1]) - (lag == 2)*trading_costs
                    Rcum[counter_ret, 0] = Rpair[counter, p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    if wi_update == 1:
                        wi[0] = wi[0]*(1+Rt.iloc[j, first_col])
                        wi[1] = wi[1]*(1+Rt.iloc[j, second_col])

                else:
                    Rpair[counter, p] = 0  # closed (this code not necessary)

                pairs_opened = new_pairs_opened

                if (pairs_opened == +1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) <= 0
                    or Rcum_ret[-1] <= Stop_loss
                    or (Rcum_ret[-1] >= Stop_gain)
                        or pairs_opened == -1 and (IPt[j-i, first_col]-IPt[j-i, second_col]) >= 0) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):

                    new_pairs_opened = 0  # close prices: prices cross
                    # If the pairs are open and the spread is smaller than the
                    # threshold, close the position
                    lag = 0
                    # see above, marker
                    avg_price_dev[counter+1, no_pairs+p] = 1

                    if wi_update == 1:
                        Rpair[counter, p] = Rpair[counter, p] - \
                            trading_costs - percentage_costs

                elif (+IPt[j-i, first_col]-IPt[j-i, second_col] > 2.0*std_limit) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    new_pairs_opened = 1  # open pairs
                    # If the difference between the prices are larger than
                    # the limit, and there is volume, open the position (short 1st, long 2nd)
                    lag = lag + 1
                    if pairs_opened == 0:
                        Rcum = np.zeros((six_months, 1))
                        counter_ret = 1

                elif (-IPt[j-i, first_col]+IPt[j-i, second_col] > 2.0*std_limit) and ((trade_req + (Vt.iloc[j, first_col] > 0) + (Vt.iloc[j, second_col] > 0)) > 1):
                    new_pairs_opened = -1  # open pairs
                    # If the difference between the prices are larger than
                    # the limit, and there is volume, open the position (short 2nd, long 1st)
                    lag = lag + 1
                    if pairs_opened == 0:  # - If the pair was closed, reset accumulated return matrix and counter
                        Rcum = np.zeros((six_months, 1))
                        counter_ret = 1

                if new_pairs_opened == +1 and lag == 1:
                    avg_price_dev[counter, p] = 2*(+IPt[j-i, first_col] - IPt[j-i, second_col])/(
                        IPt[j-i, first_col] + IPt[j-i, second_col])
                    lag = lag + 1
                    wi = [1, 1]

                elif new_pairs_opened == -1 and lag == 1:

                    avg_price_dev[counter, p] = 2*(-IPt(j-i, first_col) + IPt(
                        j-i, second_col))/(IPt(j-i, first_col) + IPt(j-i, second_col))
                    lag = lag + 1
                    wi = [1, 1]

                counter += 1

        if pairs_opened != 0:
            print(
                f"Pair {pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']} did not converged, Days: {counter_ret} | Return: {Rcum_ret[-1]}")
            # Includes trading cost in the last day of trading, due to closing position
            Rpair[counter-1, p] = Rpair[counter-1, p] - \
                trading_costs - percentage_costs
            avg_price_dev[counter-1, no_pairs+p] = 1
            operations.append(
                {
                    "Semester": big_loop,
                    "Days": counter_ret,
                    "S1": pairs[p]['s1_ticker'],
                    "S2": pairs[p]['s2_ticker'],
                    "Pair": f"{pairs[p]['s1_ticker']}-{pairs[p]['s2_ticker']}",
                    "Return": Rcum_ret[-1],
                    "Converged": False
                }
            )

    # print(f"Rpair len: {len(Rpair)} | value: {Rpair}")
    pd.DataFrame(Rpair).to_csv("Rpair.csv", header=None, index=False)

    # Using 2 Weighting Schemes - Fully Invested and Committed Capital
    # ------------------------------------------------------------
    # Calculate portfolio returns (ew, vw) out of percentage Rpair
    # ------------------------------------------------------------

    print(f"Calculating returns from day {counter-six_months+1} to {counter}")

    Rpair_temp = Rpair[counter-six_months+1:counter+1]
    # Ret_acum = np.zeros((six_months, 4))
    #
    # eq-weighted average on committed cap.; weights reset to "one" (or any equal weight) after each day
    #
    wi = np.ones((1, no_pairs))
    np.append(wi, np.cumprod(1+Rpair_temp))
    # print(f"Wi shape: {np.shape(wi)}")

    #
    # ew-weighted, committed cap.; weights "restart" every 6 month period;
    # (each portfolio gets 1 dollar at the beginning)
    # print(f"Rpair_temp len: {np.shape(Rpair_temp)} value: {Rpair_temp}")

    pd.DataFrame(Rpair_temp[:six_months]).to_csv(
        "rpair_limited.csv", header=None, index=False, na_rep='NULL')
    pd.DataFrame(np.sum(np.multiply(wi, Rpair_temp[:six_months]), axis=1)).to_csv(
        "row_sum.csv", header=None, index=False)
    Rp_ew_cc['Return'][counter-six_months+1:counter+1] = np.nansum(
        np.multiply(wi, Rpair_temp[:six_months]), axis=1) / np.sum(wi)

    # print(f"Rp_ew_cc len: {np.shape(Rp_ew_cc)} | value: {Rp_ew_cc}")

    ret_acum_df['CC'][counter-six_months+1:counter+1] = np.cumprod(
        1 + (Rp_ew_cc['Return'][counter-six_months+1:counter+1]))[:six_months]
    # [MaxDD,~] = maxdrawdown(ret2tick(Rp_ew_cc(counter-Six_mo:counter-1,:)));
    # MDDw(big_loop,2) = MaxDD;
    # Sortino = sortinoratio(Rp_ew_cc(counter-Six_mo:counter-1,:),0);
    # Sortino_w(big_loop,2) = Sortino;

    #
    # vw-weighted, fully invested; weights "restart" from 1 every time a new pair is opened;
    # Capital divided between open portfolios.

    pd.DataFrame(avg_price_dev).to_csv(
        "avg_price_dev.csv", header=None, index=False)

    # indicator for days when pairs open
    pa_open = np.zeros((six_months, no_pairs))
    for i2 in range(0, no_pairs):
        pa_opened_temp = 0
        temp_lag = 0

        for i1 in range(0, six_months):
            if pa_opened_temp == 1 and daylag == 0:  # opening period not included, closing included
                pa_open[i1, i2] = 1
                days_open[pairs_number, 0] = days_open[pairs_number, 0] + 1

            if pa_opened_temp == 1 and daylag == 1 and temp_lag == 1:
                pa_open[i1, i2] = 1
                days_open[pairs_number, 0] = days_open[pairs_number, 0] + 1

            if pa_opened_temp == 1 and daylag == 1 and temp_lag == 0:
                temp_lag = 1

            if avg_price_dev[counter-six_months+i1+1, i2] != 0:
                pa_opened_temp = 1
                pairs_number = pairs_number + 1

            if avg_price_dev[counter-six_months+i1+1, no_pairs+i2] != 0:
                pa_opened_temp = 0
                temp_lag = 0

    # pd.DataFrame(pa_open).to_csv("pa_open.csv", header=None, index=False)

    wi2 = np.multiply(wi, pa_open)

    for i2 in range(0, six_months):  # takes care in a situation where no pairs are open
        if sum(pa_open[i2, :]) == 0:
            wi2[i2, 0:no_pairs] = 0.2 * np.ones((1, no_pairs))
            pa_open[i2, 0:no_pairs] = np.ones((1, no_pairs))

    # pd.DataFrame(wi2).to_csv("wi2.csv", header=None, index=False)

    Rp_vw_fi['Return'][counter-six_months+1:counter+1] = np.divide(np.nansum(np.multiply(
        wi2, Rpair_temp[:six_months+1, :]), axis=1), np.sum(wi2, axis=1))

    ret_acum_df['FI'][counter-six_months+1:counter +
                      1] = np.cumprod(1+(Rp_vw_fi['Return'][counter-six_months+1:counter+1]))
    # [MaxDD,~] = maxdrawdown(ret2tick(Rp_vw_fi(counter-Six_mo:counter-1,:)));
    # MDDw(big_loop,4) = MaxDD;
    # Sortino = sortinoratio(Rp_vw_fi(counter-Six_mo:counter-1,:),0);
    # Sortino_w(big_loop,4) = Sortino;

    RmRf[counter-six_months+1:counter +
                   1] = (Rm.iloc[counter-six_months+1:counter + 1, :] - Rf.iloc[counter-six_months+1:counter + 1, :]).to_numpy()

    risk_free['Return'][counter-six_months+1:counter + 1] = (Rf.iloc[counter-six_months+1:counter + 1, 0])

    ret_acum_df['RMRF'][counter-six_months+1:counter+1] = np.cumprod(
        1+(Rm.iloc[:six_months, 0]-Rf.iloc[:six_months, 0]).to_numpy())

    ret_acum_df['SEMESTER'][counter-six_months+1:counter+1] = int(big_loop)
    Rp_ew_cc['Semester'][counter-six_months+1:counter+1] = int(big_loop)
    Rp_vw_fi['Semester'][counter-six_months+1:counter+1] = int(big_loop)
    risk_free['Semester'][counter-six_months+1:counter+1] = int(big_loop)

    if Stop_loss == float("-inf"):
        Rp_ew_cc.to_csv("distance_results/duration_limit/Rp_ew_cc.csv", index=False)
        Rp_vw_fi.to_csv("distance_results/duration_limit/Rp_vw_fi.csv", index=False)
        pd.DataFrame(RmRf).to_csv("distance_results/duration_limit/RmRf.csv", header=None, index=False)
        risk_free.to_csv("distance_results/duration_limit/risk_free.csv")
        ret_acum_df.to_csv("distance_results/duration_limit/ret_acum_df.csv")
    else:
         
        Rp_ew_cc.to_csv(f"distance_results/stop_{math.trunc(stop_dir)}/Rp_ew_cc.csv", index=False)
        Rp_vw_fi.to_csv(f"distance_results/stop_{math.trunc(stop_dir)}/Rp_vw_fi.csv", index=False)
        pd.DataFrame(RmRf).to_csv(f"distance_results/stop_{math.trunc(stop_dir)}/RmRf.csv", header=None, index=False)
        risk_free.to_csv(f"distance_results/stop_{math.trunc(stop_dir)}/risk_free.csv")
        ret_acum_df.to_csv(f"distance_results/stop_{math.trunc(stop_dir)}/ret_acum_df.csv")

    for i2 in range(0, no_pairs):
        if sum(avg_price_dev[counter-six_months+1:counter+1, i2] != 0) != 0:
            periods_with_open_pair = periods_with_open_pair + 1
            no_pairs_opened[big_loop, i2] = no_pairs_opened[
                big_loop, p] + sum(avg_price_dev[counter-six_months+1:counter+1, i2] != 0)
        else:
            periods_without_open_pair = periods_without_open_pair + 1

    i = i + periods.iloc[big_loop, 0]

    big_loop = big_loop + 1

operations_df = pd.DataFrame(operations)
if Stop_loss == float('-inf'):
    operations_df.to_csv(f"distance_results/duration_limit/operations.csv")
else:
    operations_df.to_csv(f"distance_results/stop_{math.trunc(stop_dir)}/operations.csv")
