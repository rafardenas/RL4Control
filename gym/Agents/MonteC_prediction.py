#Monte Carlo Implementation
#Rafael C
#22.12.2020

#based on https://github.com/dennybritz/reinforcement-learning/blob/master/MC/MC%20Prediction%20Solution.ipynb

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
    sys.path.append("../") 


#matplotlib.style.use('ggplot')

env = gym.make('Blackjack-v0')



def sample_policy(observation):
    """
    A policy that sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

def mc_prediction_first_visit(policy, env, num_episodes, discount_factor=1):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final value function
    V = defaultdict(float)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
            
        # Find all states the we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = [i for i,x in enumerate(episode) if x[0] == state]
            #first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx[0]:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]

    return V

#uncomment to run policy evaluation

#V = mc_prediction_first_visit(sample_policy, env, num_episodes=500000)
#sorted(V.items(), key = lambda x: x[1])


#####################################
###     MC First Visit Control with e-soft policy
#####################################


def esoft_policy(state, policy = None, eps = 1/6):
    """Random policy in first episode, then follow e-greedy policy, GLIE not implemented"""
    if len(policy) == 0:
        return np.random.randint(0,2)
    else:
        p = np.random.uniform()
        if p <= eps:
            return np.random.randint(0,2)
        else:
            return policy[state]
            
def e_soft_policy2(Q, state, eps = 1/6):
    A = np.zeros(env.action_space.n) * 1 + eps/env.action_space.n 
    arg = np.argmax(Q[state])
    A[arg] += (1-eps)
    return A

    
def mc_fv_control(policy, env, num_episodes, discount_factor=1):
    """
    MC first visit control algo. using e-soft policies to estimate optimal policy. Implemented as the pseudocode of the book page 101
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    
    Returns:
        A dictionary Q that maps from state-action -> value. 
        A dictionary P that holds the optimal policy 
    """
        


    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The final value function
    Q = defaultdict(list)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    P = defaultdict(int)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            probs = policy(Q,state)
            #action = policy(state,P)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))        #Store the action in a tuple
            if done:
                break
            state = next_state

        # States-actions in the episode
        # We convert each state-action to a tuple so that we can use it as a dict key
        states_actions_in_episode = set([tuple((x[0], x[1])) for x in episode])
        #print(states_actions_in_episode)
        for state_action in states_actions_in_episode:
            # Find the first occurance of the state_actions in the episode
            first_occurence_idx = [i for i,x in enumerate(episode) if x[0] == state_action[0]]
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx[0]:])])
            returns_sum[state_action[0]][state_action[1]] += G
            returns_count[state_action[0]][state_action[1]] += 1.0
            Q[state_action[0]][state_action[1]] = returns_sum[state_action[0]][state_action[1]] / returns_count[state_action[0]][state_action[1]]
            #Policy improvement
            P[state_action[0]] = np.argmax(Q[state_action[0]])
        
    return Q,P



#Uncomment to run Policy iteration with MC 

#Q,Opt_P = mc_fv_control(e_soft_policy2, env, num_episodes= 500000)
#print(sorted(Q.items(), key = lambda x: x[1][0])) #sorting by value
