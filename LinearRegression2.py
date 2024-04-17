######Linear regression from existing package

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
df=pd.read_csv(r"C:\Users\li\Desktop\all\job\python\3LinearRegression\FuelConsumption.csv")
column_headers = list(df.columns.values)
X=df[column_headers[4]]
Y=df[column_headers[8]]
X=(X-X.mean())/(X.var())#Normalization
Y=(Y-Y.mean())/(Y.var()) 
X = sm.add_constant(X)# A constant variable of 1 (b0*1) is added
ols = sm.OLS(Y, X)
results = ols.fit()
print(results.summary())

