import numpy as np
import copy
import matplotlib.pyplot as plt

class Arm:
    '''
    An Arm returns a reward value from a stationary probability distribution.
    '''
    def __init__(self, id):
        '''
        Initialize the reward probability distribution for
        an arm for the n-armed bandit problem
        '''
        self.id = id

        mu, sigma = 0.0, 1.0
        self.true_mean_reward = np.random.normal(mu, sigma)

        self.num_pulls = 0
        

    def pull(self):
        '''
        A pull constitutes a single action that returns a reward.
        '''
        reward = np.random.normal(self.true_mean_reward, 1.0)
        self.num_pulls += 1.0

        return(reward, self.num_pulls)

class Generic_Test_Bench:
    '''
    The generic test bench is a skeleton object for creating a
    set of N arms.
    '''
    def __init__(self, n=2):
        '''
        '''

        #The arm list allows for indexing the arm to play
        #The of the arm is the arms 'id'
        self.arm_list = []
        self.optimal_arm_id = 0
        self.n = n

        highest_reward = -1*np.inf

        #generate n-arms and identify the 'optimal arm'...arm with highest mean reward.
        for i in range(n):
            
            arm = Arm(i)
            self.arm_list.append(arm)

            if(arm.true_mean_reward > highest_reward):
                self.optimal_arm_id = copy.deepcopy(i)
                highest_reward = copy.deepcopy(arm.true_mean_reward)

    def play(self, arm_id):
        '''
        Pull the arm with id 'arm_id'
        '''

        reward, num_pulls = self.arm_list[arm_id].pull()

        #The last returned parameter indicates if the 
        #statistically optimal arm for the bandit was choosen of not.
        if(arm_id == self.optimal_arm_id):
            return reward, num_pulls, True
        else:
            return reward, num_pulls, False