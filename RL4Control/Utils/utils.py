#RL for control, utils
#27.12.2020
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import scipy.integrate as scp
import numpy as np
import matplotlib.pyplot as plt
import time

def discrete_env(state, modulus, s, upper_bounds, stochasticity = True):
    """
    Discretisation of the system
    """  
    resid = state % modulus     
    resid = resid/modulus       
    UB = 1 - resid              
    
    if stochasticity:
        draw = np.random.randint(0,2, state.shape[0])
        for i in range(state.shape[0]):
            if draw[i] < UB[i]:
                state[i] = state[i] - resid[i] * modulus[i]
            else:
                state[i] = state[i] - resid[i] * modulus[i] + modulus[i]
    else:
        state[i] = state[i] - resid[i] * modulus[i]

    #Rouding precision errors for purpose of key values
    for i in range(len(state)):
        if state[i] < 0:
            state[i] = 0
        elif state[i] < modulus[i]/2:
            state[i] = 0
        elif state[i] > upper_bounds[i]:
            state[i] = upper_bounds[i]
    

        f = str(modulus[i])
        decimal = f[::-1].find('.')  
        state[i] = np.round(state[i], decimal)

    state = (*tuple(state), s)

    return state


def integrator(model, current_state, dt, ctrl, modulus, s):
    """
    Util function for integration of the dynamics
    """
    ode = scp.ode(model)                           # define ode, using model defined as np array
    ode.set_integrator('lsoda', nsteps=3000)                         # define integrator
    ode.set_initial_value(current_state,dt)                          # set initial value
    ode.set_f_params(ctrl[s])             #aka params                        # set control action
    
    current_state = ode.integrate(ode.t + dt)
    return current_state




def eps_decay(xi, ei, episodes):
    """
    Params:
        xi: decay constant. Set 1 for no decay
        ei: episode index, used for decay progress
        episodes: number of total episodes. (the decay rate will approach 0.1 when completion)
    """
    if xi == 1:
        return 1

    G = -np.log(0.1) * xi * episodes
    if ei < G:
        behave = np.exp(-ei / (episodes * xi))
    else:
        behave = 0.1
    
    return behave

    


