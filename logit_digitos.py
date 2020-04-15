import sklearn.datasets as skdata
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1))
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
scores=np.zeros((10,4))
for c in range(-5,5):
    for Tol in range(-2,2):
        tol=10**Tol
        C=10**c
        clf = LogisticRegression( C=C, penalty='l1', solver='saga', tol=tol)
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        scores[c+5,Tol+2]=score
Cm=10**0
tolm=10**-2
clf = LogisticRegression( C=Cm, penalty='l1', solver='saga', tol=tolm)
clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
coef = clf.coef_.copy()
plt.figure(figsize=(10, 5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2, 5, i + 1)
    l1_plot.imshow(coef[i].reshape(8, 8), interpolation='nearest',
                   cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('%i' % i)
plt.suptitle('Coeficientes de la regresion  (C=%4.2f'%Cm+' tol=%4.2f)'%tolm)
plt.savefig("coeficientes.png")
MatConf=np.zeros((10,10))
y_pred=clf.predict(x_test)
for i in range(10):
    for i2 in range(10):
        MatConf[i,i2]=np.sum((y_test==i)*(y_pred==i2))
plt.figure(figsize=(10,10))
CNames=["0","1","2","3","4","5","6","7","8","9"]
Ejes=["Truth","Predict"]
plt.imshow(MatConf)
plt.ylabel("Truth")
plt.xlabel("Predict")
plt.xticks(np.arange(10),CNames)
plt.yticks(np.arange(10),CNames)
for i in range(10):
    for i2 in range(10):
        plt.text (i-0.3,i2,"%2.3f"%(MatConf[i,i2]/np.sum(MatConf[:,i2])))
plt.title("Matriz de Confusion (Score=%4.2f)"%score)
plt.savefig("confusion.png")