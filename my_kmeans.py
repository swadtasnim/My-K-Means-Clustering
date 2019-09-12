import pandas as pd
import numpy as np
import random,math
from scipy.spatial import distance
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from synthetic_data import synthesize

from matplotlib.backends.backend_pdf import PdfPages

import os





def plot_3d(res_datapoints,m,pdf,kk):

    fig = plt.figure(kk)
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
    ax.scatter(xm, ym, zm, c="red",marker="X")

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    #plt.show()
    pdf.savefig(fig)





def plot_2d(res_datapoints,m,pdf,kk):


    fig = plt.figure(kk)
    x=[]
    y=[]
    l=[]
    #plt.subplot(number_of_k,2,kk)
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
    
    #plt.show()
    pdf.savefig(fig)



def dist(vec1,vec2):
    
    #print("x: ",vec1,"\n m: ",vec2)
    #d = [(a - b)**2 for a, b in zip(vec1, vec2)]
    #d = int(math.sqrt(sum(d)))+
    d = distance.euclidean(vec1,vec2)
    return d




def SSE(D,m,cluster_set):
    
    #print(D,m,cluster_set)
    #print(list(zip(D,cluster_set)))
    sqr_sum = 0

    for d,c in list(zip(D,cluster_set)):
        #print(d,c,m[c])
        #print(dist(d,m[c]))
        dis = dist(d,m[c])
        sqr_sum += dis*dis
    return sqr_sum




def disk_kmeans(k,datapoints):

    D = datapoints["points"]
    print(D)
    #print("datapoints: ", D)
    m=[D[i] for i in random.sample(range(len(D)),k) ]
    #print("Centroids: ",m)

    Cluster_set = {}
    #min_SSE = []

    prev_SSE = 1000000000
    relative_error = 1000000000
    #for nn in range(0,2500):
    nn = 0
    while(relative_error > .0005):
        print("----------ITERATION "+ str(nn+1) +" ---------------")
        nn +=1
        arr = [0 for i in range(len(D[0]))]
        ones = [1 for i in range(len(D[0]))]
        #print(arr)
        s = [arr for i in range(k)]
        n = [arr for i in range(k)]
        
        #print(s,n)

        
        cluster_set=[]
        for x in D:
            #print("distances from centroids: ",[dist(x,mm) for mm in m])
            j = np.argmin([dist(x,mm) for mm in m])
            #print("index j: ",j," s[j]: ",s[j],"data x: ",x)
            
            cluster_set.append(j)
            s[j]= list(map(lambda x,y:x+y,s[j],x))
            #print("value of s: ", s)
            n[j]= list(map(lambda x,y:x+y,n[j],ones))
            #print("value of n: ",n)
            
            #distance = [dd for dd in dist([d],m)]

        SSE(D,m,cluster_set)
        

        #print([np.array(c) for c in s])
        a = [np.array(c) for c in s]
        #print(a)
        b = [np.array(c) for c in n]
        m = [a/b for a,b in list(zip(a,b))  ]
        #print(m)




   
       
        #print("new Centroid m: ", m)

        #min_SSE.append(SSE(D,m,cluster_set))        

       # min_SSE = abs(SSE())
        #print("min SSE: ",min_SSE)

        
        curr_SSE = SSE(D,m,cluster_set)
       # sse_set.append(curr_SSE)
        print("Current SSE: ", curr_SSE)
        relative_error = abs(prev_SSE-curr_SSE)/prev_SSE

        print("relative error: ",relative_error)
    
        prev_SSE = curr_SSE        

    datapoints["Cluster"]=cluster_set
    #print("cluster set: ",cluster_set)



    
    #print(list(zip(datapoints["points"],datapoints["Cluster"])))
    #res_datapoints=[p,c for p,c in list(zip(points,labels))]
    return m, list(zip(datapoints["points"],datapoints["Cluster"])),cluster_set, sqrt(prev_SSE)


test1 = [[1],[2],[3],[7],[8],[9],[10],[11]]
test2 = [[1,2],[3,2],[7,8],[8,4],[9,0],[40,30], [33,6],[10,39]]
test3_scikit = [[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]]

test4 = [[1,2],[3,2],[7,8],[8,4],[9,0],[40,30], [33,6],[10,39], [33,55],[40,45],[90,34]]

test_random = np.random.randint(10000, size=(100, 2))
test_random3d = np.random.randint(100000, size=(100, 3))


#print(test_random3d)

k=5
dim_data=3
N=1000

pdf=PdfPages("cluster.pdf")
pdf0=PdfPages("synthetic_data.pdf")
test_synthetic = synthesize(k,dim_data,N,pdf0)
pdf0.close()
#print(test_synthetic)
datapoints = {"points":test_synthetic, "Cluster":[]}

#print(disk_kmeans(2,datapoints))


sse_set=[]
number_of_k= 15
#plt.subplots(number_of_k-1,1)
pdf=PdfPages("cluster.pdf")

for kk in range(1,number_of_k):
    m,res_datapoints,cluster_set,sse = disk_kmeans(kk,datapoints)
    #print("Centroids m: ",m," \nDatapoints: ",res_datapoints,"\nCluster set: ", cluster_set)
    #print("length m: ", len(m[0]))
    print("SSE: ",sse)
    sse_set.append(sse)



    #l=[name for name in mcolors.cnames]
    #print("color map",l)
    if len(m[0])==2:
        plot_2d(res_datapoints,m,pdf,kk)

    elif len(m[0])==3:
        print("3D")
        plot_3d(res_datapoints,m,pdf,kk)    

print(sse_set)
print(min(sse_set))
print(len(sse_set))

fig2 = plt.figure()
ax = plt.axes()


ax.plot([x for x in range(1,number_of_k)], sse_set)  
plt.show()
pdf.savefig(fig2) 
pdf.close()
