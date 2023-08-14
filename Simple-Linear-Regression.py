import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn  import linear_model
from sklearn.model_selection import train_test_split
path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
df=pd.read_csv(path)
# print(df)
# print (df.columns)
cdf=df[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
print (cdf.head())
viz=cdf[['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# viz.hist()
plt.scatter(cdf.FUELCONSUMPTION_COMB,cdf.CO2EMISSIONS,color='red')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()
reg= linear_model.LinearRegression()
x=cdf[["FUELCONSUMPTION_COMB"]].values
y=cdf[['CO2EMISSIONS']].values
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)
reg.fit(X_train,y_train)
print('cofficients:',reg.coef_)
print('Intercept',reg.intercept_)
pred=reg.predict(X_test)
# print(pred[0:5])
# print(y_test[0:5])
