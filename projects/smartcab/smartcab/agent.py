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
#         self.before = {
#             'state':(),
#             'action':None,
#             'reward':0,
#             'alpha':0
#         }
        
        self.log_filename = os.path.join("logs", "ob.csv")
        self.log_file = open(self.log_filename, 'wb')
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
        #self.alpha -= ((1 - 0.7) / 200)
        
        if (True == testing):
            self.epsilon = 0
            self.alpha = 0
            self.log_file.close()
        else:
            trialFactor = -2
            # if (self.trial <= 50):
            #     self.epsilon = 0.9
            # elif (self.trial <= 100):
            #     self.epsilon = 0.5 #math.pow(self.trial - 50, trialFactor) #-(0.8/math.pow(250, 2)) * math.pow(self.trial - 50, 2) + 0.8 #math.exp(-trialFactor * self.trial) #1.0 / math.pow(self.trial, 2) # -= (1.0 / 1000)#
            # # elif (self.trial <= 200):
            # #    self.epsilon = math.pow(self.trial - 100, trialFactor)
            # else:
            #     self.epsilon = math.pow(self.trial - 100, trialFactor)
            
            #self.epsilon =2.0025e-05 * np.power(self.trial, 2) + -8.51e-03 * self.trial  + 1
            #self.epsilon = 2.250e-05 * np.power(self.trial, 2) + -0.009 * self.trial + 1
            #self.epsilon = math.pow(0.8, self.trial)
            self.epsilon = math.pow(self.trial, trialFactor)
            #self.epsilon = 1.0/math.sqrt(self.trial)
            
            # x = self.trial
            # intercept_y = -(120.0/400) * x + 120
            # intercept_amp =40
            # a = 0.75
            # amplitude = intercept_amp- (intercept_amp/np.power(300, 2)) * np.power(x, 2)
            # y = (amplitude * np.cos(a * x) + intercept_y) / 240
            
            # self.epsilon = y

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
        
        state = (waypoint, inputs['light'], inputs['oncoming']) #  (waypoint, inputs['light'], deadline)#(waypoint, inputs['light'], deadline)  #(inputs['light'], inputs['left'],inputs['right'], inputs['oncoming']) # 

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state

        maxQ = self.Q[state][self.valid_actions[random.randint(0, 3)]] # maxQ for Q(self.valid_actions[random.randint(0, 3)]s', a')

        for theAction in self.valid_actions:
             if (self.Q[state][theAction] > maxQ):
                maxQ = self.Q[state][theAction]
        
        # The direction(waypoint) choose last time, is the direction now
        # To change the state at the moment, we should only change the waypoint of the state
        # ###Include them in a list named stateNextList
        # ###No need to do so.
        # ###We can compare the corresponding Q-value directly.
        # Each of them is a state[s'] that can be reached from the state[s] through an action[a]
        
        #stateNextList = []
        
        
#         for theAction in self.valid_actions:
#             stateNextUnit = list(state)
#             stateNextUnit[0] = theAction
#             stateNextUnit[-1] -= 1
            
# #            start search next waypoint in the next place
# #            tempself = self
# #            tempenv = self.env
            
# #             heading = self.env.agent_states[self]['heading']
# #             if theAction is not None:
# #                     # Agent wants to drive left:
# #                     if theAction == 'left':
# #                         heading = (heading[1], -heading[0])
# #                     # Agent wants to drive right:
# #                     elif theAction == 'right':
# #                         heading = (-heading[1], heading[0])
                   
# #                     location = self.env.agent_states[self]['location']
# #                     location = ((location[0] + heading[0] - self.env.bounds[0]) % (self.env.bounds[2] - self.env.bounds[0] + 1) + self.env.bounds[0],
# #                                 (location[1] + heading[1] - self.env.bounds[1]) % (self.env.bounds[3] - self.env.bounds[1] + 1) + self.env.bounds[1])  # wrap-around
# #                     self.env.agent_states[self]['location'] = location
# #                     self.env.agent_states[self]['heading'] = heading
           
# #             stateNextUnit[0] = self.planner.next_waypoint() # The next waypoint 
           
# #            self = tempself
# #             self.env = tempenv
#            #end for the next waypoint in the next place
           
#            #stateNextUnit[1] = ['green', 'red'][random.randint(0, 1)]
            
#             stateNextUnit = tuple(stateNextUnit)
#             #stateNextList.append(stateNextUnit)
#             if (stateNextUnit in self.Q.keys()):
#                 for ii in self.Q[stateNextUnit].keys():
#                     if (self.Q[stateNextUnit][ii] > maxQ):
#                         maxQ = self.Q[stateNextUnit][ii]
        
        return maxQ 


    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if not (state in self.Q.keys()):
            self.Q[state] = {}
            for theAction in self.valid_actions:
                self.Q[state][theAction] = 0.0

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
        
        #chooseAction = open('chooseAction.txt', 'a')#fileChooseAction
        
        action = self.valid_actions[random.randint(0, 3)]
        randResult = random.random()
        maxQ = self.get_maxQ(state)
        
#         if (False == self.learning):
#             action = self.valid_actions[random.randint(0, 3)]
#         elif (randResult < self.epsilon):
#             action = self.valid_actions[random.randint(0, 3)]
#             self.Q[state][action] += (randResult * maxQ)
#         else:
#             actList = [theAction for theAction in self.Q[state].keys() if (maxQ == self.Q[state][theAction])]
            
#             if (len(actList) > 1):
#                 action = actList[random.randint(0, len(actList) - 1)]
#             else:
#                 action = actList[0]
        
        if (
            (True == self.env.trial_data['testing']) or 
            (
                (True == self.learning) and 
                (randResult <= (1 - self.epsilon))
            )
           ):
            #actList = [action] #[None] #[self.valid_actions[random.randint(0, 3)]]
            #maxQ = self.get_maxQ(state)
            actList = [theAction for theAction in self.Q[state].keys() if (maxQ == self.Q[state][theAction])]
            
            if (len(actList) > 1):
                action = actList[random.randint(0, len(actList) - 1)]
            else:
                action = actList[0]
        
#         if (False == self.env.trial_data['testing']):
#             if ((True == self.learning) and (random.random() <= (1 - self.epsilon))):
#                 actList = [self.valid_actions[random.randint(0, 3)]]#[None] #self.Q[state].keys()[0]
#                 for theAction in self.Q[state].keys():
#                     if (self.Q[state][theAction] > self.Q[state][actList[0]]):
#                         actList = [theAction]
#                     elif (self.Q[state][theAction] == self.Q[state][actList[0]]):
#                         actList.append(theAction)
                 
#                 if (len(actList) > 1):
#                     action = actList[random.randint(0, len(actList) - 1)]
#                 else:
#                     action = actList[0]
#             else:
#                 action = self.valid_actions[random.randint(0, 3)]
#         else:
#             actList = [self.valid_actions[random.randint(0, 3)]] #[None]
#             for theAction in self.Q[state].keys():
#                 if (self.Q[state][theAction] > self.Q[state][actList[0]]):
#                     actList = [theAction]
#                 elif (self.Q[state][theAction] == self.Q[state][actList[0]]):
#                     actList.append(theAction)
            
#                 if (len(actList) > 1):
#                     action = actList[random.randint(0, len(actList) - 1)]
#                 else:
#                     action = actList[0]
            
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
        
        #state_next = self.build_state()
        #self.createQ(state_next)
        #self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.get_maxQ(state_next))
        
#         if (self.trial > 1):
#             self.Q[self.before['state']][self.before['action']] = (1 - self.before['alpha']) * self.Q[self.before['state']][self.before['action']] + \
#                                                                                self.before['alpha'] * (self.before['reward'] )#+ 0 * self.get_maxQ(state))
#         self.before = {
#             'state':state,
#             'action':action,
#             'reward':reward,
#             'alpha':self.alpha
#         }
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * reward
        
        return

    def logger(self, state, action, reward, Q_before, Q_after):
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

        if (False == self.env.trial_data['testing']):
        
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
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning=True, alpha=0.7)
    
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
    sim.run(n_test=200, tolerance=math.pow(200, -2))#0.1##1.0/math.sqrt(200)#


if __name__ == '__main__':
    run()
