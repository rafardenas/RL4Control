# Chemical Process Control with Reinforcement Learning
### Plannig:                                                                                                                                                                                                                                                                                                                                                                                                                                                               


|- Visualizations
    |- Interactive
        || Buttons: Play, pause, change control actions, choose reaction, introduce fault in reactor, etc.
    |- Agent's performance
    |- Control actions (Temperature changes, fluxes, etc)


|- Reactions
    |- Use "ideal" dynamics with equations from literature
    |- Use maybe Bioreactors? 


|- Agents
    |- MC
        |- Incremental
        |- First Visit
        |- Importance Sampling
    |- TD
        |- Q Learning
        |- SARSA
    |- DRL
        |- Deep Q network
        |- Oracle? By Elthon
        |- Gaussian Process

# Utils

Preprocessing: Discretize the states and add stochasticity


# Objects design
Classes: 

1. Envs: Reactions/Reactors

Attributes: 
Methods:
    Discretize:
    Generate the Observations:
    Reward:

2. Agents: RL Algorithms

Super Class: Agents
Classes: Algorithm
Attributes:
Methods:
    Act (policy eval)
    Learn (algos)

@class method?
@property?
@super class?


Identifying objects:

Proper noun - 	Instance (object)	
Common noun	 - Class (or attribute)	Field Officer, PlayingCard, value
Doing verb	 - Operation	Creates, submits, shuffles
Being verb	 - Inheritance	Is a kind of, is one of either
Having verb	 - Aggregation/Composition	Has, consists of, includes
Modal verb	 - Constraint	Must be
Adjective	 - Helps identify an attribute	a yellow ball (i.e. color)


Nouns
    Agent
        Verbs:
        |- Act
        |- (train) Learn
    Adjective:
        |- Control actions  
        |- Eps greedy
        |- Alpha
        |- Gamma
        |- Exploration technique?
        |- Args


Nouns
    Environment (Reactors?)
        Verbs:
        |- Response (reward, wind blows) (Run the reaction)
        |- 
    Adjectives:
        |- Input to the Reactions
        |- 


Nouns
    Reactions
        Verbs:
        |- Run the reaction
        |- 
    Adjective:
        |- 





