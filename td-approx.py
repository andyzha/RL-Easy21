import numpy as np
import easy21
import collections
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def convertSA(a, dc, sum):
    indList = []
    for dci, dcr in enumerate(([1, 4], [4, 7], [7, 10])):
        for sumi, sumr in enumerate(([1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21])):
            if dc >= dcr[0] and dc <= dcr[1] and sum >= sumr[0] and sum <= sumr[1]:
                indList.append([a, dci, sumi])
    #print("SA convert from ", a, dc, sum, indList)
    return indList

# # Test non overlapping interval
# def convertSA(a, dc, sum):
#     indList = []
#     for dci, dcr in enumerate(([1, 4], [5, 7], [8, 10])):
#         for sumi, sumr in enumerate(([1, 5], [6, 9], [10, 12], [13, 15], [16, 18], [19, 21])):
#             if dc >= dcr[0] and dc <= dcr[1] and sum >= sumr[0] and sum <= sumr[1]:
#                 indList.append([a, dci, sumi])
#     #print("SA convert from ", a, dc, sum, indList)
#     return indList

def getFeatureMatrix(a, dc, sum):
    feature = np.zeros((2, 3, 6))
    indList = convertSA(a, dc, sum)
    for ind in indList:
        feature[ind[0], ind[1], ind[2]] = 1

    return feature 

def getQValue(a, dc, sum, w):
    f = getFeatureMatrix(a, dc, sum)
    res = np.dot(f.reshape(1, 36) , w)
    #print("getQValue ", a, dc, sum, res.shape, w.shape, f.reshape(1, 36).shape)
    #print(res)
    return res

def getQMatrix(w):
    q = np.zeros((2,11,22))
    for a in range(2):
        for dc in range(1, 11):
            for sum in range(1, 22):
                q[a, dc, sum] = getQValue(a, dc, sum, w)
    return q

def getAction(s, qw):
    dc, sum = s[1], s[2]
    # Hit: 0; Stick 1
    # s (type, dc, sum); type 1 == terminal
    ep = 0.05
    if random.random() < ep:
        #print("random action with ep: ", ep)
        return random.randint(0,1)
    return 1 if getQValue(0, dc, sum, qw) < getQValue(1, dc, sum, qw) else 0

def runTD(size, tdLambda):    
    # init q
    qw = np.zeros((36, 1)) # weight w of q
    # plot learning curve
    sqrErr = []
    mcq = np.load("montecarlo-qsa.npy")

    # each episode
    for i in range(size):
        # Eligibility
        e = np.zeros((2, 3, 6))
        # init one episode
        s0 = easy21.init()
        s = s0
        a = getAction(s, qw)
        #print("init ", s)
        # each step
        while s[0] != 1:                       
            #print("action: ", a)            
            # get s' and a'
            sprime = easy21.step(s, a)
            aprime = getAction(sprime, qw) if sprime[0] != 1 else 0 

            # err 
            sa, saprime = (a, s[1], s[2]), (aprime, sprime[1], sprime[2])
            alpha = 0.01

            r = 0 if sprime[0] != 1 else sprime[2]
            g = getQValue(aprime, sprime[1], sprime[2], qw) if sprime[0] != 1 else 0
            g += r
            err = g - getQValue(a, s[1], s[2], qw)
            indList = convertSA(a, s[1], s[2])
            for ind in indList:
                e[ind[0], ind[1], ind[2]] += 1

            # update q matric  
            qw = qw + alpha * err * e.reshape(36, 1)  
            e = e * tdLambda
            
            s,a = sprime, aprime
            #print("state: ", s)
        #print("result: ", s[2])
        sqrErr.append(np.power((getQMatrix(qw) - mcq), 2).mean())
    return qw, sqrErr

# # Run and plot q for certain lambda
# qw1, se1 = runTD(1000, 1)
# easy21.plot_qsa(getQMatrix(qw1))
# # -------------------------------------------------------------------------------

# # Run and plot err for each lambda
# def plot_err(errList):
#     print(errList)
#     fig = plt.figure()
#     plt.plot(np.arange(0, 1.1, 0.1), errList)
#     plt.show()

# mcq = np.load("montecarlo-qsa.npy")
# errList = []
# for la in np.arange(0, 1.1, 0.1):
#     #print(la)
#     qwla, errla = runTD(1000, la)
#     sqrErr = np.power((getQMatrix(qwla) - mcq), 2).mean()
#     errList.append(sqrErr)

# plot_err(errList)

# # -------------------------------------------------------------------------------

# # Run and plot learning curve
def plot_learningCurve(la):
    size = 10000
    qla, errList = runTD(size, la)
    plt.plot(range(1, size+1), errList)
    
#plot_learningCurve(0)
plot_learningCurve(1)
plt.show()