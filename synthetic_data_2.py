import pandas as pd
import numpy as np
import random,math
from scipy.spatial import distance

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def plot_3d(res_datapoints,m):

    fig = plt.figure()
    ax = Axes3D(fig)
    x=[]
    y=[]
    z=[]
    l=[]
    for d,cl in res_datapoints:
        x.append(d[0])
        y.append(d[1])
        z.append(d[2])
        l.append(cl+10)
    #print(cl)
    ax.scatter(x, y, z, c=l)
    xm=[]
    ym=[]
    zm=[]
    l=[]
    nm=20
    for mm in m:
        xm.append(mm[0])
        ym.append(mm[1])
        zm.append(mm[2])
        l.append(nm)
        nm +=1

    #print(xm,ym,zm)
    ax.scatter(xm, ym, zm, c="red",marker="X" , s= 100)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()





def plot_2d(res_datapoints,m):
    x=[]
    y=[]
    l=[]
    for d,cl in res_datapoints:
        x.append(d[0])
        y.append(d[1])
        l.append(cl+10)
    #print(cl)
    plt.scatter(x, y, c=l)
    xm=[]
    ym=[]
    l=[]
    nm=20
    for mm in m:
        xm.append(mm[0])
        ym.append(mm[1])
        l.append(nm)
        nm +=1

    #print(xm,ym)
    plt.scatter(xm, ym, c="red",marker="X")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



prob = [.68,.28,.04,.02]


k=8
dim_data = 2
N = 100000
means=np.random.randint(100, size=(k, dim_data))
std = np.random.randint(1,6,size=k)

param_set = list(zip(means,std))
print("parameter: ",param_set)

m = []
data = []
cluster_mark = 1
for p in param_set:
    meu,sigma = p
    m.append(meu)
    level_data = []
    nn=1
    for pr in prob:
        print("----",int((N/k)*pr),"---")
        x = np.random.randint(-nn,nn, size=(int((N/k)*pr), dim_data))
        level_data.extend((xx*sigma+meu,cluster_mark) for xx in x)
        nn +=1
    """
    x =np.random.randint(-2,2, size=(int(N/k), dim_data))
    #np.random.randint(sigma-3,sigma+3)
    
    """
    #data.extend([(np.random.randint(sigma-3,sigma+3)*xx + meu,cluster_mark) for xx in x])
    #data.extend([(level_data,cluster_mark)])
    data.extend(level_data)
    cluster_mark +=1

#print("datapoints: ",data)
print("centroids: ",m)

if len(m[0])==2:
    plot_2d(data,m)

elif len(m[0])==3:
    print("3D")
    plot_3d(data,m)  


#print("mue: ",means,"\nsigma: ",std)