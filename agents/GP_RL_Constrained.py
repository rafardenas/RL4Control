#GP-RL Paper

import gym
import numpy as np
import matplotlib.pyplot as plt
import GPy
import scipy.integrate as scp
from scipy.optimize import minimize
from collections import namedtuple
from collections import deque
from numpy.random import default_rng
from config_GP import configGP


class Env_base():
    r"""
    Base class for the environment
    Attributes:
        params(dic): parameters of the diff equations for the dynamics
        steps(int): number of discrete intervals in one episode (take it as equivalent of 'movements' in Agent's class, aka one movement per step)  
        t_f(int): lenght of the episode (unitless)
        x0(array): initial state, each item is the initial state for a variable. 
        control(dict or tuple): dictionary/tuple holding the values of the controlling variables, those variables are per se the controlling actions
        modulus(array): granularity of the states discretization (only for tabular cases)
        state_UB(array): upper bound
    """
    def __init__(self, parameters, steps, tf, x0, bounds, no_controls, noisy = False):
        self.parameters = parameters
        self.steps = steps 
        self.tf = tf
        self.x0 = x0
        self.noisy = noisy
        self.bounds = bounds
        self.no_controls = no_controls
        self.time_step = 0
        self.dt = tf/steps
    
    def model(self, t, state, control):
        params = self.parameters
        globals().update(params)
        nd = 5

        Sigma_v = [1e-4,1e-4,2e-4,0.1,0.2]

        CA  = state[0]
        CB  = state[1]
        CC  = state[2]
        T   = state[3]
        Vol = state[4] 

        F   =  control[0]
        T_a =  control[1]
            
        r1 = A1*np.exp(E1A*(1./Tr1-1./T))
        r2 = A2*np.exp(E2A*(1./Tr2-1./T))

        dCA   = -r1*CA + (CA0-CA)*(F/Vol)
        dCB   =  r1*CA/2 - r2*CB - CB*(F/Vol)
        dCC   =  3*r2*CB - CC*(F/Vol)
        dT    =  (UA*10.**4*(T_a-T) - CA0*F*CpA*(T-T0) + (HRA*(-r1*CA)+HRB*(-r2*CB\
                  ))*Vol)/((CA*CpA+CpB*CB+CpC*CC)*Vol + N0H2S04*CpH2SO4)
        dVol  =  F

        response = np.array([dCA, dCB, dCC, dT, dVol])
        if self.noisy: response += np.random.normal([0 for i in range(nd)], Sigma_v)

        return response

    def reset(self):
        state = self.x0
        self.time_step = 0
        return state

    def reward(self, state):
        return state[2] * state[4]                               #paid at every iteration

    def transition(self, state, action):
        self.time_step += 1
        ode   = scp.ode(self.model)                               # define ode, using model defined as np array
        ode.set_integrator('lsoda', nsteps=3000)                     # define integrator
        ode.set_initial_value(state, self.dt)                        # set initial value
        ode.set_f_params(action)                                     # set control action
        next_state = ode.integrate(ode.t + self.dt)                       # integrate system
        reward = self.reward(state)
        next_state = np.array(next_state)
        return next_state, reward, self.time_step



class GP_agent():
    def __init__(self, env, dims_input):
        self.env = env
        self.dims_input = dims_input
        self.kernel = GPy.kern.RBF(dims_input + self.env.no_controls, variance=1., ARD=True)
        self.inputs = []
        self.outputs = []
        self.valid_results = []
        self.core = None

    def update(self):
        #print(self.dims_input, self.env.no_controls)
        X = np.array(self.inputs).reshape(-1, self.dims_input + self.env.no_controls)
        Y = np.array(self.outputs).reshape(-1, 1)
        self.core = GPy.models.GPRegression(X, Y, self.kernel)     #building/updating the GP
        return self.core
        
    def add_input(self, state, action):
        "state and actions have to be lists"
        state, action = list(state), list(action)
        p = np.array([*state, *action])         
        self.inputs.append(p)
        return
    
    def add_val_result(self, state, action):
        "state and actions have to be lists"
        state, action = list(state), list(action)
        p = np.array([*state, *action])         
        self.valid_results.append(p)
        return
        
    def add_output(self, Y):
        self.outputs.append(Y)

    def get_outputs(self):
        return self.outputs

    


class experiment():
    def __init__(self, env, agent, config):
        self.env = env
        self.config = config
        self.models = []
        for i in range(self.env.steps): #intanziating one GP per time/control step
            self.models.append(agent(self.env, self.config.dims_input))
        self.opt_result = 0
          #number of runs with untrained models to gather information

    def select_action(self, state, model, pre_filling = False):
        eps = self.config.eps
        p = np.random.uniform()
        if pre_filling: p = 0
        if p > eps:
            action = self.best_action(state, model)
        else:
            action = self.random_action()
        in_bounds = True           #the minimizer ensures that the chosen action is in the set of feasible space
        assert in_bounds
        return action

    def best_action(self, state, model): #arg max over actions
        #return self.optimizer_control(model, state).x -- To be used with scipy
        max_a, max_q = self.random_search(state, model)
        return max_a

    def random_action(self): #random action from the action space
        actions = np.zeros((len(self.env.bounds)))
        for i in range(len(actions)):
            actions[i] = np.random.uniform(self.env.bounds[i,0], self.env.bounds[i,1])
        actions = actions
        #print(actions[0])
        return actions

    def wrapper_pred(self, guess, state, model):
        s_a = np.hstack((state, guess))
        guess = np.reshape(s_a, (-1,1))
        print("guess shape", guess.shape)
        return - model.core.predict(guess)[0] #negative to make it maximization

    def optimizer_control(self, model, state):  #fix state, go over actions
        action_guess = self.random_action()
        print(action_guess)
        print("Optimizing")
        assert isinstance(state, np.ndarray) #state has to be ndarray
        opt_result = minimize(self.wrapper_pred, action_guess, args = (state, model))#,bounds = self.env.bounds) #fixing states and maximizing over the actions
                                      #The optimization here is not constrained, have to set the boundaries explicitely?
        return opt_result

    def max_q(self, model, state):
        #return -self.optimizer_control(model, state).fun -- To be used with scipy
        max_a, max_q = self.random_search(state, model)
        return max_q
    
    def random_search(self, state, model):
        'model: regression model from GPy'
        actions = np.zeros(shape=(self.config.rand_search_cand, len(self.env.bounds)))
        for i in range(actions.shape[1]):
            actions[:,i] = np.random.choice(np.linspace(self.env.bounds[i,0], self.env.bounds[i,1]),\
                size=actions.shape[0],replace=False)
        state = state.reshape(1,-1)
        #print(state.shape)
        states = np.tile(state, (10,1))
        s_a = np.concatenate((states, actions), axis=1)
        #print(s_a.shape)
        landscape = model.core.predict(s_a)[0]
        #print(landscape.shape)
        optimum = np.argmax(landscape)
        #assert np.shape(optimum)[0] != 0  
        #print("iteration completed")
        return actions[optimum][:], landscape[optimum]
        

    def training_step(self):
        print("training")
        state = self.env.reset()
        for i in range(self.env.steps):
            action = self.best_action(state, self.models[i])
            ns, r, t_step = self.env.transition(state, action)
            Y = r + self.config.gamma * self.max_q((self.models[i]), ns)
            self.models[i].add_input(state, action)    #add new training inputs
            self.models[i].add_output(Y)               #add new training output
            m = self.models[i].update()                #fit GPregression
            m.optimize(messages=False)
            m.optimize_restarts(self.config.no_restarts)
            state = ns
        return

    def pre_filling(self):
        'Act randomly to initialise the initially empty GP'
        for i in range(self.config.pre_filling_iters):
            state = self.env.reset()
            for i in range(self.env.steps):
                action = self.select_action(state, self.models[i], pre_filling=True)
                ns, r, t_step = self.env.transition(state, action)
                Y = r
                self.models[i].add_input(state, action)    #add new training inputs
                self.models[i].add_output(Y)               #add new training output
                m = self.models[i].update()                #refit GPregression
                m.optimize(messages=False)
                m.optimize_restarts(self.config.no_restarts)
                state = ns
        return
            
    
    def training_loop(self):
        self.pre_filling()
        for i in range(config.training_iter):
            self.training_step()
        return

    def get_train_inputs(self, model):
        return model.inputs[self.config.pre_filling_iters::]

    def get_validation_data(self, model):
        return model.valid_results

    def get_train_outputs(self, model):
        return model.outputs[self.config.pre_filling_iters::]

    def get_trained_models(self):
        return self.models
    
    def validation_loop(self):
        for i in range(self.config.valid_iter):
            print('Validation')
            state = self.env.reset()
            for i in range(self.env.steps):
                action = self.best_action(state, self.models[i])
                ns, r, t_step = self.env.transition(state, action)
                self.models[i].add_val_result(state, action)    #add new training inputs
                state = ns
        return
    
params = {'CpA':30.,'CpB':60.,'CpC':20.,'CpH2SO4':35.,'T0':305.,'HRA':-6500.,'HRB':8000.,'E1A':9500./1.987,'E2A':7000./1.987,'A1':1.25,\
                 'Tr1':420.,'Tr2':400.,'CA0':4.,'A2':0.08,'UA':4.5,'N0H2S04':100.}
steps = 11
tf= 4
x0 = np.array([1,0,0,290,100])
bounds = np.array([[0,270],[298,500]])
config = configGP
config.dims_input = x0.shape[0]
config.training_iter = 20 #lets start with 1 iter
    
            
env   = Env_base(params, steps, tf, x0, bounds, config.no_controls, noisy=False)
#agent = GP_agent(env, config.input_dim)
agent = GP_agent
exp   = experiment(env, agent, config)
exp.training_loop()
exp.validation_loop()
outputs = np.zeros((config.training_iter, steps, config.dims_input+config.no_controls))
validation_data = np.zeros((config.valid_iter, steps, config.dims_input+config.no_controls))

for i in range(config.training_iter):
    for j in range(steps):
        outputs[i,j,:] = exp.get_train_inputs(exp.models[j])[i]

for i in range(config.valid_iter):
    for j in range(steps):
        validation_data[i,j,:] = exp.get_validation_data(exp.models[j])[i]



def plotting(data):
    fig, axs = plt.subplots(8, 1,sharex=True,figsize=(8,10))
    legend = ['$C_a$ (kmol m$^{-3}$)','$C_b$ (kmol m$^{-3}$)','$C_c$ (kmol m$^{-3}$)','$T$ (K)','$V$ (m$^3$)','$F$ (m$^3$hr$^{-1}$)','$T_a$ (K)']
    for j in range(data.shape[-1]):
        xx = data[:,:,j]
        for i in range(data.shape[0]):
            axs[j].plot(np.arange(len(xx[i,:])), xx[i,:])#, label = 'iteration #: {}'.format(str(i)))
            axs[j].set_ylabel(legend[j])
    plt.show()

plotting(validation_data)