import numpy as np
import pandas as pd
import bs4 as bs
import requests
import yfinance as yf
from datetime import datetime
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

from cointegration_code.tickers import TICKERS

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

tickers = [s.replace('\n', '') for s in tickers]
start = datetime(2000,1,1)
end = datetime(2022,12,13)
print("Starting to download data")
data = yf.download(tickers, start=start, end=end)
raw_close_prices = data['Adj Close']
print("Data downloaded")
print(raw_close_prices)

#------------------------------------------------------------------------------
# IMPORTANT FUNCTIONS
#------------------------------------------------------------------------------

# MQO para encontrar o coeficiente de cointegração e criando a serie do spread
def OLS(data_ticker1, data_ticker2):
    spread = sm.OLS(data_ticker1,data_ticker2)
    spread = spread.fit()
    return data_ticker1 + (data_ticker2 * -spread.params[0]), spread.params[0]


# ADF test
def ADF(spread):
    return ts.adfuller(spread) # H0: Raiz unitária.


# Encontra o coeficiente de cointegração e realiza o ADF test
def ADF_test(data_ticker1, data_ticker2):
    ols = OLS(data_ticker1, data_ticker2)
    spread = ols[0]
    gamma = ols[1]
    return ADF(spread),gamma


# Encontra os pares cointegrados
def find_cointegrated_pairs_mod(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))
    gammas_matrix= np.ones((n, n))
    keys = data.keys()
    for i in range(n):
        for j in range(i+1, n):
            S1 = keys[i]
            S2 = keys[j]
            result = ADF_test(data[S1], data[S2])
            gammas_matrix[i, j] =result[1] # gamma
            pvalue = result[0][1] # pvalue
            pvalue_matrix[i, j] = pvalue
    return pvalue_matrix, gammas_matrix


# Ordenando os melhores pares
def top_coint_pairs(data,pvalue_matrix,gamma, alpha, n): 
#alpha = nivel de significancia para o teste ADF
#n = top n ativos com o menor pvalue    
    
    pvalues_f = pvalues[np.where(pvalues < alpha)] # pvalores menores que alpha
    stock_a = data.columns[list(np.where(pvalues < alpha))[0]] # relacionando o pvalor com a ação A
    stock_b = data.columns[list(np.where(pvalues < alpha))[1]] # relacionando o pvalor com a ação B
    gammas_f = gammas[np.where(pvalues < alpha)] # relacionando o pvalor com o gamma
    N = len(list(np.where(pvalues < alpha)[0])) # quantidade de pares cointegrados

    d = []
    for i in range(N):
      d.append(
            {
                'Stock A': stock_a[i],
                'Stock B': stock_b[i],
                'P-Values': pvalues_f[i],
                'Gamma': gammas_f[i]
            }
        )

    return pd.DataFrame(d).sort_values(by="P-Values").iloc[:n,]


# Calcula os retornos da carteira e armazenando em um data frame
def calculate_profit(spread, threshold, par1, par2, resumo):
    
    log_ret = spread.diff() # log return eh o incremento
    dias = spread.index
    z_score = (spread-spread.mean())/spread.std()
    portfolio_return = []
    pos = 0 # 0: sem posição aberta
            # 1: Comprei o meu portfolio h = (1,-gamma)
            # -1: Vendi o meu portfolio h = -(1,-gamma)

    dias_abertura = []
    dias_fechamento = []

    count = 0
    dia_abertura = 0
    dia_fechamento = 0

    for i in range(1,len(z_score)):
                        
        if (z_score[i] > threshold) and (pos == 0):
            pos = -1

            count += 1
            dia_abertura = dias[i]
            retornos_op = []


        elif (z_score[i] < -threshold)  and (pos == 0):
            pos = 1            

            count += 1
            dia_abertura = dias[i]
            retornos_op = []

        else:

          if (pos == 1) and (z_score[i] < -0.75):
            portfolio_return.append(log_ret[i]*pos)
            retornos_op.append(log_ret[i]*pos)


          elif (pos == 1) and (z_score[i] >= -0.75):
            portfolio_return.append(log_ret[i]*pos)
            pos = 0

            dia_fechamento = dias[i]
            delta_dias = dia_fechamento - dia_abertura
            retornos_op.append(log_ret[i]*pos)
            retorno_op = pd.Series(retornos_op).sum()
            
            resumo.append([count, dia_abertura, dia_fechamento, delta_dias, retorno_op, par1, par2])

            
          elif (pos == -1) and (z_score[i] > 0.75):
            portfolio_return.append(log_ret[i]*pos)
            retornos_op.append(log_ret[i]*pos)


          elif (pos == -1) and (z_score[i] <= 0.75):
            portfolio_return.append(log_ret[i]*pos)
            pos = 0

            dia_fechamento = dias[i]
            delta_dias = dia_fechamento - dia_abertura
            retornos_op.append(log_ret[i]*pos)
            retorno_op = pd.Series(retornos_op).sum()

            resumo.append([count, dia_abertura, dia_fechamento, delta_dias, retorno_op, par1, par2])

          else: 
            if pos != 0:
              dia_fechamento = dias[i]
              delta_dias = dia_fechamento - dia_abertura
              retornos_op.append(log_ret[i]*pos)
              retorno_op = pd.Series(retornos_op).sum()

              resumo.append([count, dia_abertura, dia_fechamento, delta_dias, retorno_op, par1, par2])
            
            
            pos = 0

    total_ret = pd.Series(portfolio_return).sum()

    return total_ret, resumo


# Calcula o expoente de hurst
def get_hurst_exponent(time_series):
    
    # Definindo o intervalo de taus
    max_tau = round(len(time_series)/4)
    taus = range(2, max_tau)

    # Calculando a variável k
    k = [np.std(np.subtract(time_series[tau:], time_series[:-tau])) for tau in taus]
    
    #'To calculate the Hurst exponent, we first calculate the standard deviation of the differences between a series and its lagged version, for a range of possible lags.'

    # Calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(taus), np.log(k), 1)
    
    #'We then estimate the Hurst exponent as the slope of the log-log plot of the number of lags versus the mentioned standard deviations.'

    return reg[0]
