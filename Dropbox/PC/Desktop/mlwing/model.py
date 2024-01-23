import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.model_selection import train_test_split
df = pd.read_csv('TvMarketing.csv')
X = df.TV
y = df.Sales
X = np.array(X)
X = np.reshape(X,(-1,1))
y = np.array(y)
y = np.reshape(y,(-1,1))
X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr = LinearRegression()
lr.fit(X_train,X_test)
y_pred_test = lr.predict(y_train)


r2 = r2_score(y_test,y_pred_test)
plt.figure(figsize=(20,8))
plt.scatter(y_train,y_test,color='blue',label='Plot')
plt.plot(y_train,y_pred_test,color='red',linewidth=2,label='BEST FIT')

plt.legend(loc=4)
plt.title('Tv Marketing')
plt.xlabel('TV')
plt.ylabel('Sales')

plt.show()
print(f'R2 Score : {r2 }')
print(f'Training score: {lr.score(X_train,X_test)}')
print(f'Test score: {lr.score(y_train,y_test)}')