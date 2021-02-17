#RL industrial case, inspired by one project I work on during my internship
#Rafael C
#31.01.2021

import numpy as np
import os
import gym
import sys
import math
import scipy.integrate as scp
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from envs.env_DRL import Environment2
from config_industrial_case import config

#changelog: 
# changed all the t_step for self.dt
# transformed the dif equ to integration form via scipy

# Class defining the model dynamics, costs, etc

class industrial_model(Environment2):
    def __init__(self, config, parameters, steps, tf, x0, control, modulus, state_UB, noisy = False):
        super(industrial_model, self).__init__(parameters, steps, tf, x0, control, modulus, state_UB, noisy = False) 
        self = config
        self.dt = tf/steps           #increased time every time step automatically by scipy(step in the integration of the kinetics equations)
        self.x0 = {
        'educt_2tank' : 0,
        'educt1' : 0,
        'educt2' : 0,
        'val_prod1' : 0,
        'prod_degradation' : 0,
        'temp_in_C' : 0,
        'yieldd' : 0,
        'volume': 0,
        'i' : 0                 #what is this???
        }

        self.simtime        = (4. * 60.) / self.self.dt  # hours * min/hour / min total = number of simulation steps
        self.minE2left_perc = 0.10 # Threshold to stop the batch. If x percentage are left the product is done
        self.val_prod1_degrades = True  # Product 1 degrades over time and the byproduct product2 is formed
        self.deltaH =  -150.  # [kJ/mol]     energy created during reaction
        self.minW   = -4000.
        self.maxW   =  2000.
        self.minD   =    10.
        # JPD changed maxD from 1500 to 2500
        self.maxD   =  2500.
        self.minT   =   -20
        self.maxT   =   100
        self.critTPerc = 0.2
        self.t_save    = 1  # save results ever t_save timestep
        
        # cost parameters (all fictional)
        self.t_setup           = 180  # Time to restart the batch in min / needed as otherwise the fastest heating and dosage will always lead to the best result
        self.educt1_Euro_kg    = 50   # costs Euro per kg
        self.educt2_Euro_kg    = 100  # costs Euro per kg
        self.val_prod1_Euro_kg = 300  # revenue Euro per kg
        # JPD we have changed here from 10 Euro to 20 Euro for the Energy
        self.Energy_Euro_kWh   = 10   # costs Euro per kWh
        self.fcosts_Euro_min   = 20   # costs Euro per minute (facilities, staff etc.)
        
        # -------------------------------------------------------------------------------------------
        # start values for energy, dosage and temperature (here fixed in comments random)
        self.startW = 0  #float(random.sample(range(int(self.minW), int(self.maxW)),1)[0])
        self.startD = 0  #float(random.sample(range(int(self.minD), int(self.maxD)),1)[0])
        self.startT = 40 #float(random.sample(range(int(self.minT)+int( self.critTPerc*float(self.maxT-self.minT)), int(self.maxT)-int( self.critTPerc*float(self.maxT-self.minT))),1)[0])

        self.rho = 1100.  # density [kg/m3]
        self.K = 0.01     # kinetics product formation
        self.K_deg_val_prod1 = 0.3 # kinetics value product degradation and formation of degradation product
        self.constant1 = 50    # constant for kinetics
        self.constant2 = 0.003 # constant for kinetics
        self.T_ref = 323.15  # reference temperature [K]
        self.Q = 4.190 * self.rho  # [kJ/m3]  energy required to increase Temperature by 1 _C
        self.educt2_kg   = 750  # [kg] total amount of educt 2 that can be added to educt1 and catalyst
        self.educt2_gmol = 200  # [g/mol]
        self.educt1_gmol = 100  # [g/mol]
        self.val_prod1_gmol = 300  # [g/mol]
        self.E_start = 1.3  # How much more educt 1 needs to be added in the beginning of the reaction in relation to educt 2
        self.educt2_mol = self.educt2_kg / self.educt2_gmol * 1000.  # kg * mol/kg =  [mol]
        self.educt1_mol = self.E_start * self.educt2_mol  # [mol]
        self.educt1_kg = self.educt1_mol * self.educt1_gmol / 1000.  # [kg]
        self.T_in_C = self.startT # temperature in tank [C]
        self.verbose = True # with or without output to screen
        self.dose_ed2 = self.startD  # dosage in kg/s
        self.ed2_tank_mol = self.educt2_mol # educt in storage tank
        self.W_input = self.startW 
        self.V = self.educt1_kg / self.rho *1000  # Volume in [l]
        self.educt2 = 0.
        self.educt1 = self.educt1_kg / self.educt1_gmol * 1000.  # educt1 in [mol]
        self.educt1_t0 = self.educt1  # educt1 at timestep 0
        self.val_prod1_t0 = -(self.educt1 - self.educt1_t0)  # val_prod1 at timestep 0 in [mol]
        self.val_prod1 = self.val_prod1_t0
        self.deg_prod  = 0.
        self.tot_yield = 0.
        self.costsaver = 0.
        self.revenueaver = 0.
        self.final_gain = 0.

        self.i = 0  # time counter


        # determin end of calculation
        self.ymax = 0.  # maximum yield
        self.tmax = 0.  # time of maximum yield
        self.valid = True  # process is not producing invalid numbers
        self.done = False  # less than 0.05 of educt2 is left
        self.critical = False

        # create empty list for output vars
        self.myvals = []
        self.mysets = []
        self.mycosts = []
        #self.maxgain = -100000
        self.rev_per_t = 0
        self.tot_gain = 0
        self.time_step = 0


    def reset(self):
        state = self.x0
        self.W_input = self.startW
        self.dose_ed2 = self.startD
        self.time_step = 0
        return state

    def model(self, t, state, control):
        r"""
        Generates the environment's response agains agents actions
        Does an integration of the kinetics equations
        """
        params = self.parameters
        FCn    = control

        #params for the ODEs #missing
        self.parameters = {'constant1' : 50, 'constant2' : 0.003, 'K' : 0.01, 'T_ref' : 323.15,
                    'K_deg_val_prod1' : 0.3
                    }
        
        # variable rate equations aka diff equations below

        R = (self.V * (-self.parameters['constant2'] * np.exp(-self.parameters['constant1'] / self.parameters['K'] * \
            (1. / (self.T_in_C + 273.15) - 1. / self.T_ref))) * (state['educt2'] / self.V) ** 2. * (state['educt1'] / self.V) ** 2.) #* self.dt ###Have to check the 'dt'
        
        if (self.val_prod1_degrades):
            R2  = - (self.V * (self.val_prod1 / self.V)**(2.) *
                    (-self.parameters['constant2'] * np.exp(-self.parameters['constant1'] / self.parameters['K_deg_val_prod1'] * \
                    (1. / (self.T_in_C + 273.15) - 1. / self.T_ref))))# * self.dt
        else:
            R2 = 0
        #===============================pending===============================
        #should use scipy for the integration, I am not sure I have to sum or the integration will take acocunt for that
        #self.educt1 = self.educt1 + R # total educt1 in [mol] 
        #self.educt2 = self.educt2 + R # total educt1 in [mol]
        #===============================#===============================

        return np.array([R, R2], dtype='float64')
        # dosage rate of educt 2 aka control action, check dt

        
    def transition(self, state, action):
        self.time_step += 1
        ode   = scp.ode(self.model)                               # define ode, using model defined as np array
        ode.set_integrator('lsoda', nsteps=3000)                     # define integrator
        ode.set_initial_value(state, self.dt)                        # set initial value
        ode.set_f_params(action)                                     # set control action
        next_state = ode.integrate(ode.t + self.dt)                       # integrate system
        noise = np.random.normal(0, 0.125, size = len(next_state)) if self.noisy else 0
        rewards = False
        assert rewards
        reward = self.reward(state)
        next_state = np.array((next_state) + noise).round(1)
        
        ##########WATCH this change############ 
        ##return next_state, reward, self.time_step
        #########################
        R, R2 = next_state[0], next_state[1] 
        ed2_mol_s = action * 1000. / self.educt2_gmol / 60. * self.dt           #divide over 60 because the dosage units are kg/s and the dt is in minutes
        dose_ed2 = min(ed2_mol_s * self.dt, self.ed2_tank_mol)                      #control action after applied naive safety constraints (min)
        
        # reduce educt 2 in dosage tank until its empty
        self.ed2_tank_mol = self.ed2_tank_mol - dose_ed2  # total educt2 in [mol] in dosage tank
        state['educt2']       = state['educt2']       + dose_ed2        # total educt2 in [mol]

        #product 1 degrading

        if (self.val_prod1_degrades):
            self.val_prod1 -= R - R2
            self.deg_prod +=      R2
        else:
            self.val_prod1 = -(self.educt1 - self.educt1_t0)


        #total yield
        self.tot_yield = (self.val_prod1 - self.val_prod1_t0) / (self.educt2_mol)

        #volume in tank

        self.V = (self.educt2    * self.educt2_gmol/1000 + 
                  self.educt1    * self.educt1_gmol/1000 + 
                  self.val_prod1 * self.val_prod1_gmol/1000 + 
                  self.deg_prod  * self.val_prod1_gmol/1000) / self.rho * 1000.

        #update T
        self.T_in_C += R * self.deltaH / ((self.V / 1000.) * self.Q)

        #compute level

        level = (self.V / 1000.) / (2 * math.pi * 0.5 ** 2.)  # assuming the tank is a cylinder with 1 m diameter
        S_tank = level * 2 * math.pi * 0.5  # assuming only the outer sides (not the bottom) are heated or cooled
        delta_T = (self.W_input * 1000 * self.self.dt) / (self.V * (self.rho / 1000) * 4.200) * S_tank / self.V
        self.T_in_C = self.T_in_C + delta_T

        #compute yield

        if (self.tot_yield > self.ymax):
            self.ymax = self.tot_yield
            self.tmax = self.i * self.self.dt
        
        #critical state?

        if ((self.T_in_C > self.minT + (self.maxT - self.minT) * self.critTPerc) & 
            (self.T_in_C < self.maxT - (self.maxT - self.minT) * self.critTPerc)):
            self.critical = False
        else:
            self.critical = True

        if ((self.T_in_C > self.maxT) | (self.T_in_C < self.minT)):
            self.valid = False
        if ((self.T_in_C > self.maxT - (self.maxT - self.minT) * self.critTPerc) |
            (self.T_in_C < self.minT + (self.maxT - self.minT) * self.critTPerc)):
            self.critical = True

        #validity of the batch

        if ((self.educt2 < 0) or (self.educt1 < 0) or (self.V <= 0)):
            self.valid = False
        if ((self.deg_prod == np.inf) or (self.val_prod1 == np.inf)):
            self.valid = False

        if (self.i > self.simtime):
            self.valid = False

        if ((self.ed2_tank_mol <= 1) & (self.educt2 / self.educt2_mol < self.minE2left_perc)):
            self.done = True

        state = [self.ed2_tank_mol, self.educt1, self.educt2, self.val_prod1, self.deg_prod, self.T_in_C, self.tot_yield, self.V, self.i]
        return state

    def get_costs_gains(self):
        # calculate costs per total runtime
        costs1 = self.educt1_kg * self.educt1_Euro_kg # costs for educt 1
        costs2 = (self.educt2_kg - (self.ed2_tank_mol * self.educt2_gmol / 1000)) * self.educt2_Euro_kg # costs for educt2 (the remaining educt 2 in the tank can be reused and is therefore excluded)
        # JPD changed the exponent from 1.0 to 1.5; 1.3 is working already
        costs3 = sum([abs(val[1])**1.5 for val in self.mysets]) / 1000 * self.self.dt * self.Energy_Euro_kWh  # heating and cooling is equally costly. The whole energy is summed up and multiplied by the costs
        costs4 = self.i * self.self.dt * self.fcosts_Euro_min  # fix costs per time
        costs5 = (self.t_setup * self.fcosts_Euro_min) # costs for setup
        costs = costs1 + costs2 + costs3 + costs4 + costs5

        revenue = self.val_prod1 * self.val_prod1_gmol / 1000 * self.val_prod1_Euro_kg # revenue from value product
        gains = revenue - costs 
        if self.i > 0:
            rev_per_time = gains / (self.i * self.self.dt )
        else:
            rev_per_time = 0.
        #if ((gains > self.maxgain) & (self.done)):
        self.maxgain = gains
        self.rev_per_t = rev_per_time
        self.final_gain = gains
        if (not self.valid):
            self.final_gain = 0

        return self.i * self.self.dt, costs, revenue, gains, rev_per_time, self.final_gain
    
    
    def reward(self, state):
        [time, costs, revenue, gains, _,_] = self.get_costs_gains()
        
        # long term reward (paid at the end if the batch was successful)
        self.tot_gain = gains/(time/60.)
        
        #short term reward for penalizing unwanted behaviour
        reward = 0
        if (not self.valid):
            reward = -1000
            self.tot_gain = 0
            self.done = True

        if (self.critical):
            reward = -1000

        return reward