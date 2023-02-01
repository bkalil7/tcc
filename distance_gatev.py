import math
import numpy as np
import pandas as pd
from tqdm import tqdm

periods = pd.read_csv("data/Periods.csv", header=None)
Pt = pd.read_csv("data/Pt.csv", header=None)
Rt = pd.read_csv("data/Rt.csv", header=None)
ticker2 = pd.read_csv("data/ticker2.csv", header=None)
ticker_b = pd.read_csv("data/ticker_b.csv", header=None)

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

    [D,ia,ib] = np.intersect1d(ticker2.iloc[:,big_loop],ticker2.iloc[:,big_loop+1],return_indices=True)

    #print(f"D len: {np.shape(D)} | variable: {D}")
    #print(f"ia len: {np.shape(ia)} | variable: {ia}")
    #print(f"ib len: {np.shape(ib)} | variable: {ib}")

    ic = np.isin(D, ticker2.iloc[:,big_loop+3]) # como a variável id no matlab não é utilizada, deixei ela de fora do script em python
                                                # a função np.isin não retorna o valor, então teria que buscar outra alternativa 
                                                # caso necessário

    #print(f"ic len: {np.shape(ic)} | variable: {ic}")
    #print(f"D(ic) len: {np.shape(D[ic])} | variable: {D[ic]}")

    Dic_unique_sorted, B_idx = np.unique(D[ic], return_index=True)
    ie = np.in1d(D[ic], ticker_b)

    #print(f"Dic_unique_sorted len: {np.shape(Dic_unique_sorted)} | variable: {Dic_unique_sorted}")
    #print(f"B_idx len: {np.shape(B_idx)} | variable: {B_idx}")
    #print(f"ie len: {np.shape(ie)} | variable: {ie}")
    
    ig = B_idx[ie]
    #print(f"ig len: {np.shape(ig)} | variable: {ig}")

    index_listed2= np.transpose(ig)
    no_listed2 = sum(ie)
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
    print(f'Time series mean no of stock series {np.mean(no_comp)}')
    print(f'Max number of stock series {max(no_comp)}')
    print(f'Min number of stock series {min(no_comp)}')

    # ----------------------------------------------------
    # Calc SSEs
    # ----------------------------------------------------

    sse = np.zeros((no_listed2, no_listed2))
    for j in tqdm(range(0,no_listed2 - 1)):
        for k in range(j+1, no_listed2):
            sse[index_listed2[j],index_listed2[k]] = sum(np.power(IPt[1:int(twelve_months),index_listed2[j]]-IPt[1:int(twelve_months),index_listed2[k]],2))

    pd.DataFrame(sse).to_csv("SSE.csv", header=None, index=False)

    # ----------------------------------------------------
    # Find min SSEs
    # ----------------------------------------------------

    max_SSE = np.nanmax(sse) + 1
    min_SSE = np.zeros((no_pairs,1))
    min_SSE_ro = np.zeros((1,no_pairs))
    min_SSE_co = np.zeros((1,no_pairs))

    print(f"Initial Max SSE: {max_SSE}")

    for ii in range(0, no_pairs):
        t_SSE = max_SSE
        for k in range(0, no_listed2-1):
            for l in range(k+1, no_listed2):
                if sse[k,l] > 0 and sse[k,l] < t_SSE:
                    #print(f"New minimum found at ({k},{l})")
                    t_SSE = sse[k,l] # new minimum found
        
        #print(f"Minimum SSE = {t_SSE}")

        if t_SSE == max_SSE:
            print("Error")
            
        ro,co = np.where(sse == t_SSE)
        ro = ro[0]
        co = co[0]
        #print(f"Indexes = ({ro},{co})")
        min_SSE[ii,0] = sse[ro,co]
        min_SSE_ro[0,ii] = int(ro) # column of the 1st stock in a pair
        min_SSE_co[0,ii] = int(co) # column of the 2nd stock in a pair
        sse[ro,co] = max_SSE # prevent re-selection

    print(f"min_SSE_ro variable: {min_SSE_ro}")
    print(f"min_SSE_co variable: {min_SSE_co}")
    print(f"min_SSE len: {np.shape(min_SSE)} | variable: {min_SSE}")

    # ----------------------------------------------------
    # Calculate returns during the 6 month period
    # ----------------------------------------------------

    count_temp = counter

    for p in range(0, no_pairs):
        counter = count_temp 
        pairs_opened = 0 
        new_pairs_opened = 0 
        lag = 0
        std_limit = np.std(IPt[1:twelve_months,min_SSE_ro(p)]-IPt[1:+twelve_months,min_SSE_co(p)]) #standard deviation

        print(f"Std limit: {std_limit}")

        # Fixed volatility estimated in the 12 months period. Doing the calculation one pair at a time
        # Presets all variables for each pair
        Rcum = np.zeros((twelve_months,1)) 
        counter_ret = 1
        Rcum_ret = 1

        wi = []

        for j in range(i+twelve_months, i+twelve_months+six_months-1): # portfolio period
            # Defining the period as from the first day of the twe_month to the last day of the twe_month
            if daylag == 0: # w/o one day delay
                if pairs_opened == -1: # pairs opened: long 1st, short 2nd stock
                    # If a sign to open has been given, then calcule the returns
                    Rpair[counter,p] =+ np.multiply(Rt[j,min_SSE_ro(p)], wi[1]) - np.multiply(Rt(j,min_SSE_co(p)), wi[2])
                    # Rpair is the return of each pair.
                    Rcum[counter_ret,1] = Rpair[counter,p] 
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum);  
                    lag = lag + 1; # used for paying tc

                    if wi_update == 1: # The weight of each asset in the pair is updated. 
                        wi[1]=wi[1]*(1+Rt[j,min_SSE_ro(p)])
                        wi[2]=wi[2]*(1+Rt[j,min_SSE_co(p)])
                elif pairs_opened == 1: # pairs opened: short 1st, long 2nd stock
                    Rpair[counter,p] = - np.multiply(Rt[j,min_SSE_ro(p)], wi[1]) + np.multiply(Rt[j,min_SSE_co(p)], wi[2])
                    Rcum[counter_ret,1] = Rpair[counter,p]
                    counter_ret = counter_ret + 1
                    Rcum_ret = np.cumprod(1+Rcum)
                    lag = lag + 1
                        
                    if wi_update == 1:
                            wi[1]=wi[1]*(1+Rt[j,min_SSE_ro(p)])
                            wi[2]=wi[2]*(1+Rt[j,min_SSE_co(p)])
                else:
                    Rpair[counter,p] = 0 # closed (this code not necessary)
    break