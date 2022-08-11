import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Enes\\PycharmProjects\\PANDAS\\Pandas\\veriler.csv")
df.drop('ulke',axis=1,inplace=True)


x = df.iloc[:,:3]
y = df.iloc[:,3:]

from sklearn.model_selection import train_test_split


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.33 , random_state = 0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=10 , criterion="entropy") # estimators ağaç sayısıdır 
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

print(f"pred = {y_pred}")

