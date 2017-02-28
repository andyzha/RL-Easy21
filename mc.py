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

def runMC(size):
    q = np.zeros((2,11,22))
    counterSA = np.zeros((2,11,22))
    counterState = np.zeros((11,22))

    for i in range(size):
        # sample one episode
        s0 = easy21.init()
        s = s0
        #print("init ", s)
        episodes = []
        while s[0] != 1:
            counterState[s[1:]] += 1
            
            a = getAction(s, q, counterState)
            #print("action: ", a)
            sa = (a, s[1], s[2])
            counterSA[sa] += 1
            episodes.append(sa) 
            sprime = easy21.step(s, a)
            s = sprime
            #print("state: ", s)
        #print("result: ", s[2])

        for state in episodes:
            #print("s,a ", state)
            q[state] += (1/counterSA[state]) * (s[2] - q[state])
    return q

q1 = runMC(100000)

easy21.plot_qsa(q1)

#np.save("montecarlo-qsa.npy", q)
np.save("montecarlo.npy", qstar)
