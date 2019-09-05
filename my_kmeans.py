import pandas as pd
import numpy as np
import random,math
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors







def dist(vec1,vec2):
    
    d = [(a - b)**2 for a, b in zip(vec1, vec2)]
    d = int(math.sqrt(sum(d)))
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
        relative_error = abs(prev_SSE-curr_SSE)/prev_SSE

        print("relative error: ",relative_error)
    
        prev_SSE = curr_SSE        

    datapoints["Cluster"]=cluster_set
    #print("cluster set: ",cluster_set)




    #print(list(zip(datapoints["points"],datapoints["Cluster"])))
    #res_datapoints=[p,c for p,c in list(zip(points,labels))]
    return m, list(zip(datapoints["points"],datapoints["Cluster"])),cluster_set


test1 = [[1],[2],[3],[7],[8],[9],[10],[11]]
test2 = [[1,2],[3,2],[7,8],[8,4],[9,0],[40,30], [33,6],[10,39]]
test3_scikit = [[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]]

test4 = [[1,2],[3,2],[7,8],[8,4],[9,0],[40,30], [33,6],[10,39], [33,55],[40,45],[90,34]]

test_random = np.random.randint(10000, size=(100, 2))

#print(test_random)
datapoints = {"points":test_random, "Cluster":[]}

#print(disk_kmeans(2,datapoints))

m,res_datapoints,cluster_set = disk_kmeans(2,datapoints)
print("Centroids m: ",m," \nDatapoints: ",res_datapoints,"\nCluster set: ", cluster_set)



#l=[name for name in mcolors.cnames]
#print("color map",l)


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

print(xm,ym)
plt.scatter(xm, ym, c="red",marker="X")

plt.xlabel('X')
plt.ylabel('Y')
plt.show()
    