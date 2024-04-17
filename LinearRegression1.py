######Linear regression from scratch

import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\li\Desktop\all\job\python\3LinearRegression\FuelConsumption.csv")
print(df.sample(3))

column_headers = list(df.columns.values)
#get the fitted model
X=df[column_headers[4]]
Y=df[column_headers[8]]

def LinearR(X,Y):
    X=(X-X.mean())/(X.var())#Normalization
    Y=(Y-Y.mean())/(Y.var())       
    a,ai,aj=[],[],[]
    for i in np.arange(-1,1,0.01):
        for j in np.arange(-1,1,0.01):
            c=i+j*X-Y
            c=c.mul(c)
            c=c.sum()
            a.append(c)
            ai.append(i)
            aj.append(j)
    idd=a.index(min(a))
    fit_model={"intercept":ai[idd],"slope":aj[idd]}
    #get R square
    a1=fit_model["intercept"]+fit_model["slope"]*X-Y
    a1=a1.mul(a1)
    a2=Y.mul(Y)
    RR=(a2.sum()-a1.sum())/a2.sum()
    fit_model["Rsquare"]=RR
    F=(a2.sum()-a1.sum())/a1.sum()
    F=F*(len(X)-2)
    fit_model["F"]=F
    return fit_model
def Get_p_value(X):
    n=len(X)
    outr=[]
    for i in range(50):
        dfr = pd.DataFrame(np.random.randn(n, 2), columns=list('AB'))
        fit_model_r=LinearR(dfr["A"],dfr["B"])
        outr.append(fit_model_r["F"])
        print(i)
    g1=pd.DataFrame(outr)
    p_value=sum(g1>fit_model["F"])/len(g1)
    fit_model["p_value"]=p_value
fit_model=LinearR(X,Y)
Get_p_value(X)#this costs a lot of time
print(f"The slope of linear model: {fit_model['slope']}")
print(f"The intercept of linear model: {fit_model['intercept']}")
print(f"The r-square of linear model: {fit_model['Rsquare']}")
print(f"The F-statistics of linear model: {fit_model['F']}")
print(f"The p-value of linear model: {fit_model['p_value']}")