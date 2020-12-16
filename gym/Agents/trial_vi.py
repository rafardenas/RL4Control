import numpy as np

a = np.array([0.,1.,0.,0.])
p = 0.25
r = 0
gamma = 0.9
T = {0: [0,2,1,0], 1 : [1,1,1,1], 2 : [0,2,3,2], 3 : [1,3,3,2]} 
states = 4
actions = 4


iterations = 20

for eps in range(iterations):
    a_prev = a.copy()
    backup = []
    for i in range(states):
        #print(a[i])
        v = 0
        for j in T[i]:
            v += p * (r + gamma * a_prev[j])
        backup.append(gamma * a_prev[j])
        #    print(a_prev[j])
        #print(v)
        #print(T[i])
        #
        a[i] = v
        #print(v)
    #print("backup:" + str(gamma * a_prev[j]))
    print(backup)
    print(a.reshape(2,2))
    
    print("difference:" + str(np.sum(np.fabs(a - a_prev))))
    
    print("\n")

