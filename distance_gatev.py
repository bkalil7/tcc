import math
import numpy as np
import pandas as pd
from tqdm import tqdm

periods = pd.read_csv("data/Periods.csv", header=None)
Pt = pd.read_csv("data/Pt.csv", header=None)
Rt = pd.read_csv("data/Rt.csv", header=None)

days,num_assets = np.shape(Rt) # days = T in matlab code | num_assets = N

print(f"Days: {days} | Assets: {num_assets}")

daylag = 0
wi_update = 1
years = 2015 - 1990 + 0.5

no_pairs = 35
trading_costs = 0
percentage_costs = 0 # buy/sell (percentage cost for opening and closing pairs: 0.001, 0.002, for example)
trade_req = 0 # set whether (0) or not (2) positive trading volume is required for opening/closing a pair
Stop_loss =  float('-inf') # Choose how much loss we are willing to accePt on a given pair, compared to 1, i.e, 0.93 = 7% stop loss
Stop_gain =  float('inf') # Choose how much gain we are willing to accePt on a given pair, compared to 1, i.e 1.10 = 10% stop gain
s1218 = 1 # listing req. (look ahead): 12+6 months (=1)

avg_price_dev = np.zeros((days-sum(periods.iloc[0:2,0].to_list()), no_pairs*2)) 
# 12 months are w/o price deviations. The first 12 months are formation period

Rpair = np.zeros((int(periods.iloc[-2,3]),no_pairs)) # Keeps track of the return of each pair

#TODO
""" MDDpairs = zeros(length(Periods)-1,no_pairs); 
% Pre allocate Maximum Drawdown (MDD) matrix for each pair out of sample;
MDDw = zeros(length(Periods)-1,5); 
% Preallocate MDD matrix out of sample for 4 weighting schemes + Rm + (Rm-Rf).
Sortino_pairs = zeros(length(Periods)-2,no_pairs);
Sortino_w = zeros(length(Periods)-2,5); """

periods_with_open_pair = 0 # number of periods with pairs opened
periods_without_open_pair = 0 # number of periods without pairs opened
pairs_number = 0; pair_open = 0
days_open = np.zeros((no_pairs*10000,1))
# measures number of days each pair open; bad programming, but we do not know how many pairs we get
no_pairs_opened = np.zeros((int(years*2-2),no_pairs)) # measures number of times pairs opened in each pair per 6 month period

counter = 0 # Keeps track of the days in the main loop

# ----------------------------------------------------
# Start of Main Loop - Creating Price Index
# ----------------------------------------------------
# Main part of the program starts here
# ----------------------------------------------------

big_loop = 0
i = 1

while big_loop <= (years * 2 - 2):
    twelve_months = periods.iloc[big_loop, 3]
    six_months = periods.iloc[big_loop + 2, 0]

    # ----------------------------------------------------
    # Create price index IPt by setting first Pt>0 to 1
    # ----------------------------------------------------

    # Preallocate a zeros matrix with the size of the Formation + Trading period
    IPt = np.zeros((int(twelve_months + six_months), num_assets)) # IPt = Indexed Price at time t

    print("Generating Assets Price Index")

    for j in tqdm(range(0, num_assets)): # + 1 because range is not inclusive with stop index
        m = 0
        for i2 in range(0, int(twelve_months+six_months)): # same here
            if not math.isnan(Pt.iloc[i+i2 - 1, j]) and m == 0:
                IPt[i2,j] = 1
                m = 1
            elif not math.isnan(Pt.iloc[i+i2 - 1, j]) and m == 1:
                IPt[i2,j] = IPt[i2-1,j] * (1 + Rt.iloc[i+i2-1,j])
    
    pd.DataFrame(IPt).to_csv("IPt.csv", header=None, index=False)

    listed1 = IPt[0,:] > 0 # Listed at the beginning (1xN vector of booleans)
    listed2 = IPt[int(twelve_months+six_months*(s1218==1))-1,:] > 0 # listed at the end: 12/18 months from now (1xN vector of booleans)
    listed = np.multiply(listed1, listed2)

    listed_num = np.sum(listed)
    listed_indexes = np.where(listed > 0)[0]
    print(f"Listed stockes: {listed_num}")

    

    break
                