import random
import math
import sys
import numpy as np
import csv
import os

from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.trial = 0
        
        self.log_filename = os.path.join("logs", "ob.csv")
        with open(self.log_filename, 'wb') as self.log_file:
            self.log_fields = ['trial', 'testing', 'state', 'action', 'reward', 'Q_before', 'Q_after']
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_fields)
            self.log_writer.writeheader()


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        self.trial += 1
        
        if testing:
            self.epsilon = 0
            self.alpha = 0
            #self.log_file.close()
        else:
            trialFactor = -0.75
            
            # calculate alpha
#             tempx = np.array([0, 200, 400])
#             tempy = np.array([1.0, 0.7, 0.5]) #([1.0, 0.55, 0.5]) #
#             coef = np.polyfit(tempx, tempy, 2)
            
#             if (self.alpha > 0.5):
#                 self.alpha = coef[0] * np.power(self.trial, 2) + coef[1] * self.trial + coef[2] 
#             else:
#                 self.alpha = 0.5
                
            #self.alpha = 0.5 + pow(0.5, self.trial + 1)
            
            # calculate epsilon
#             tempx = np.array([0, 50, 400])
#             tempy = np.array([1.0, 0.5, 0.0099]) #([1.0, 0.55, 0.5]) #
#             coef = np.polyfit(tempx, tempy, 2)
            
#             self.epsilon = coef[0] * np.power(self.trial, 2) + coef[1] * self.trial + coef[2] 
            
#             tempx = np.array([0, 70, 300, 400])
#             tempy = np.array([1.0, 0.5, 0.12, 0.0099])
#             coef = np.polyfit(tempx, tempy, 3)
#             self.epsilon = coef[0] * np.power(self.trial, 3)  + coef[1] * np.power(self.trial, 2) + coef[2] * self.trial + coef[3] 
            
#             self.epsilon = pow(self.trial, -0.7)
            
            # tempx = np.array([0, 70, 100, 400])
            # tempy = np.array([1.0, 0.7, 0.63, 0.1])
            

#             tempx = np.array([0, 100, 200])
#             tempy = np.array([1.0, 0.7,  0.3])
#             coef = np.polyfit(tempx, tempy, 2)
#             self.epsilon = coef[0] * pow(self.trial, 2) + coef[1] * self.trial + coef[2]
            
            #self.epsilon =2.0025e-05 * np.power(self.trial, 2) + -8.51e-03 * self.trial  + 1
            #self.epsilon = 2.250e-05 * np.power(self.trial, 2) + -0.009 * self.trial + 1
            #self.epsilon = math.pow(0.8, self.trial)
            
            if (self.trial <= 500):#(self.epsilon >= 0.1):
                #tempx = np.array([0, 300, 400]) #([0, -400, 400]) #
                #tempy = np.array([1.0, 0.2,  0.1]) #([1.0, 0.1,  0.1])#
                
                #tempx = np.array([0, 100, 200])
                #tempy = np.array([1.0, 0.5,  0.1])
                #coef = np.polyfit(tempx, tempy, 2)
                #self.epsilon = coef[0] * pow(self.trial, 2) + coef[1] * self.trial + coef[2]#coef[0] * np.power(x, 3)  + coef[1] * np.power(x, 2) + coef[2] * x + coef[3] 
                
                self.epsilon = 1
            else:
                #self.epsilon -= (0.1-0.01)/(500 - 200)
                self.epsilon = pow(self.trial - 500, trialFactor)
                                  
        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent        
        
        state = (waypoint, inputs['light'], inputs['oncoming'], inputs['left'], inputs['right']) 

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        maxQ = max(self.Q[state].values())
        
        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if not (state in self.Q):
            self.Q[state] = {theAction:0.0 for theAction in self.valid_actions}

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        
        
        randResult = random.random()
        maxQ = self.get_maxQ(state)
        #zeroAction = [theAction for theAction, Qvalue in self.Q[state].items() if 0 == Qvalue]
        
#         if self.env.trial_data['testing']:
            
#             actList = [theAction for theAction, value in self.Q[state].items() if value == maxQ]
#             action = np.random.choice(actList)
        print "state"
        print self.Q[state]
        if self.env.trial_data['testing']:
            
            actList = [theAction for theAction, value in self.Q[state].items() if value == maxQ]
            action = np.random.choice(actList)
        
        elif not self.learning:
        
            print "not learning:"
            action = np.random.choice(self.valid_actions)
            print "random choice: {}".format(action)
    
#         elif zeroAction.count(0) >= 1: # from > to >=
            
#             action = np.random.choice(zeroAction)
        
        elif (randResult < self.epsilon):
            
            print "learning, learn new"
            zeroAction = [theAction for theAction, Qvalue in self.Q[state].items() if 0 == Qvalue]
            print "zeroAction: {}".format(zeroAction)
            if len(zeroAction) >= 1: 
                action = np.random.choice(zeroAction)
                print "in zeroAction choose: {}".format(action)
            else:
                action = np.random.choice(self.valid_actions)
                print "in random choose: {}".format(action)
            
        else:

            print "learning, choose max:"
            actList = [theAction for theAction, value in self.Q[state].items() if value == maxQ]
            action = np.random.choice(actList)
            
        print "At last we choose: {}".format(action)    
            
        return action


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule       
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * reward
        
        return

    def logger(self, state, action, reward, Q_before, Q_after):
        with open(self.log_filename, 'a') as self.log_file:
            self.log_fields = ['trial', 'testing', 'state', 'action', 'reward', 'Q_before', 'Q_after']
            self.log_writer = csv.DictWriter(self.log_file, fieldnames=self.log_fields)
            self.log_writer.writerow({
                'trial': self.trial,
                'testing': self.env.trial_data['testing'],
                'state': state,
                'action': action,
                'reward': reward,
                'Q_before':  Q_before,
                'Q_after': Q_after
            })
        
        return

    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward

#         if (False == self.env.trial_data['testing']):
        
        Q_before = {}
        for k, v in self.Q[state].items():
            if isinstance(v, float):
                Q_before[k] = "{:.2f}".format(v)
            else:
                Q_before[k] = v
        self.learn(state, action, reward)   # Q-learn
        Q_after = {}
        for k, v in self.Q[state].items():
            if isinstance(v, float):
                Q_after[k] = "{:.2f}".format(v)
            else:
                Q_after[k] = v

        self.logger(state, action, reward, Q_before, Q_after)

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=True)
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, alpha=0.5)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent, enforce_deadline=True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.0001, log_metrics=True, optimized=True)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(n_test=100, tolerance=pow(500, -0.75))#0.1##1.0/math.sqrt(200)#0.01


if __name__ == '__main__':
    run()
