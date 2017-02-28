import numpy as np
import collections
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def drawCard():
    val = random.randint(1, 10)
    col = random.randint(0,2)
    return (-1 if col==0 else 1) * val

def step(s, a):
    if s[0] == 1: return s
    dc, sum = s[1], s[2]
    # a == 0 hit
    if a == 0:
        sum  += drawCard()
        if sum > 21 or sum < 1:
            return (1, dc, -1)
        return (0, dc, sum)
    
    # a is stick
    while dc < 17: 
        dc += drawCard()
        if (dc > 21 or dc < 1):
            return (1, dc, 1)
    if sum == dc: return (1, dc, 0)
    return (1, dc, 1 if sum > dc else -1)

def init():
    dc = random.randint(1, 10)
    sum = 0
    while sum < 12:
        sum += random.randint(1, 10)
    return (0, dc, sum)

# s1 = (0, random.randint(1, 10), random.randint(1, 10))
# print(s1)
# s2 = step(s1, 1)
# print(s2)

def plot_qsa(q):
    print(q.shape)
    qstar = np.amax(q, axis=0)
    
    fig = plt.figure()
    ha = fig.add_subplot(111, projection='3d')
    x = range(10)
    y = range(10)
    X, Y = np.meshgrid(y, x)
    ha.plot_wireframe(X+1, Y+1, qstar[1:,12:])
    ha.set_ylabel("dealer card")
    ha.set_xlabel("player sum")
    plt.show()

