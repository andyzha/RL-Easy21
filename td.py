import numpy as np
import easy21
import collections
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getAction(s, q, counterState):  
    n0 = 100
    dc, sum = s[1], s[2]
    # Hit: 0; Stick 1
    # s (type, dc, sum); type 1 == terminal
    # print(q[hit], " and ", q[stick])
    ep = n0 / (n0 + counterState[dc, sum])
    if random.random() < ep:
        #print("random action with ep: ", ep)
        return random.randint(0,1)
    #return 1 if q[0, dc, sum] < q[1, dc, sum] else 0
    return np.argmax(q[:, dc, sum])

def runTD(size, tdLambda):    
    # init q
    q = np.zeros((2,11,22)) # action, dc, sum
    counterSA = np.zeros((2,11,22))
    counterState = np.zeros((11,22))
    # plot learning curve
    sqrErr = []
    mcq = np.load("montecarlo-qsa.npy")

    # each episode
    for i in range(size):
        # Eligibility
        e = np.zeros((2,11,22))
        # init one episode
        s0 = easy21.init()
        s = s0
        a = getAction(s, q, counterState)
        #print("init ", s)
        # each step
        while s[0] != 1:                       
            #print("action: ", a)
            counterState[s[1:]] += 1
            
            # get s' and a'
            sprime = easy21.step(s, a)
            aprime = getAction(sprime, q, counterState) if sprime[0] != 1 else 0 #  handle a' for terminal s'

            # err 
            sa, saprime = (a, s[1], s[2]), (aprime, sprime[1], sprime[2])
            counterSA[sa] += 1
            alpha = 1 / counterSA[sa]

            r = 0 if sprime[0] != 1 else sprime[2]
            g = q[saprime] if sprime[0] != 1 else 0
            g += r
            err = g - q 
            e[sa] += 1

            # update q matric  
            q = q + alpha * err * e  
            e = e * tdLambda
            
            s,a = sprime, aprime
            #print("state: ", s)
        #print("result: ", s[2])
        sqrErr.append(np.power((q - mcq), 2).mean())
    return q, sqrErr

def plot_err(errList):
    print(errList)
    fig = plt.figure()
    plt.plot(np.arange(0, 1.1, 0.1), errList)
    plt.show()

# # Run and plot q(s,a)
# q1 = runTD(10000, 1)
# easy21.plot_qsa(q1)
# # -------------------------------------------------------------------------------

# # Run and plot lambda error
# mcq = np.load("montecarlo-qsa.npy")
# errList = []
# for la in np.arange(0, 1.1, 0.1):
#     #print(la)
#     qla = runTD(10000, la)
#     sqrErr = np.power((qla - mcq), 2).mean()
#     errList.append(sqrErr)

# plot_err(errList)
# # -------------------------------------------------------------------------------

# # Run and plot learning curve
def plot_learningCurve(la):
    size = 100000
    qla, errList = runTD(size, la)
    plt.plot(range(1, size+1), errList)
    

#plot_learningCurve(0)
plot_learningCurve(1)
plt.show()