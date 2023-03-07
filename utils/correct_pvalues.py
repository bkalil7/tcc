import pandas as pd
import numpy as np

for l in range(0,49):
    print(f"cointegration_data/pvalues_semester_{l}.csv")
    p_values = pd.read_csv(f"cointegration_data/pvalues_semester_{l}.csv", header=None)
    for i in range(p_values.shape[0]): #iterate over rows
        for j in range(p_values.shape[1]):
            if p_values.iloc[i,j] == 0:
                print(f"Aqui ({i},{j})")
                p_values.iloc[i,j] = 1

    p_values.to_csv(f"cointegration_data/pvalues_semester_{l}_test.csv", header=False, index=False)