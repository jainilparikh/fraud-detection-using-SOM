import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import bone,pcolor,colorbar,plot,show
from minisom import MiniSom

df=pd.read_csv('/home/jainil/program/datasets/creditcard.csv')
print(df)


x=df.iloc[:,0:15].values
train_x_2=x
a=MinMaxScaler()
x=a.fit_transform(x)
train_x=x
y=df.iloc[:,15].values
print(y.shape)

b=MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
b.random_weights_init(x)
b.train_random(data=x,num_iteration=100)
print(b.distance_map())
pcolor(b.distance_map().T)
colorbar()
c=b.win_map(x)
print(c[4,2],c[9,6])
cust=np.concatenate((c[4,2],c[9,6]),axis=0)
cust=a.inverse_transform(cust)
print(cust[1][0])



print(cust.shape)
k=np.zeros((690,1))
n=0
train_x_3=[]
for i in range(0,len(train_x_2)):
    train_x_3.append(train_x_2[i][0])
while(n<27):
    for i in train_x_3:
        if(n>27):
            break
        if(cust[n][0]==i):
            n=n+1;
            k[n]=1;
            print("this is a")
d=0
for i in k:
    if(i==1):
        d=d+1;
print(d)
x_train,x_test,y_train,y_test=train_test_split(train_x,k)
c=0
print(x_train.shape)
print(y_train.shape)
classifier=Sequential()
classifier.add(Dense(units=15,input_dim=15,activation="relu"))
classifier.add(Dense(units=30,activation="relu"))
classifier.add(Dense(units=10,activation="relu"))
classifier.add(Dense(units=1,activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(x_train,y_train,epochs=10,batch_size=10)
a=classifier.predict(x_test)
for i in range(0,len(a)):
    if(a[i]>0.5):
        a[i]=1
    else:
        a[i]=0
for i in range(0,len(a)):
    if(a[i]!=y_test[i]):
        c=c+1
print(c/len(a))
acc=1-(c/len(a))
print("The accuracy is "+str(acc));
