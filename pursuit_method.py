from arm import Generic_Test_Bench
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import stats

class Pursuit_Method:
    '''
    Pursuit Method used action preference probabilities
    to pursue choosing the greedy action. This will eventually
    converge on selecting the greediest actions.
    '''

    def __init__(self, beta=0.01, n=10, num_plays=1000, num_bandits=1000):
        '''
        Initialize the pursuit method bandit
        '''

        #The beta parameter needs to be small to ensure the agent
        #does prematurely converge on a greedy action that is not the
        #optimal action.
        self.beta = beta
        self.n = n
        self.num_plays = 1000
        self.num_bandits = num_bandits
        self.bandits = [Generic_Test_Bench(self.n) for i in range(num_bandits)]

        #Keep an estimated value for each arm (action) for each bandit
        self.est_value = np.zeros((self.num_bandits, self.n))

        self.greedy_arms = [0 for i in range(self.num_bandits)]

        self.average_reward = np.zeros(self.num_plays)
        self.optimal_action_perc = np.zeros(self.num_plays)

    def run(self, data_queue=None):
        '''
        Run all the (num_bandits) bandits for num_plays number
        of plays. Gather the average reward and optimal action
        taken percentage data per play

        data_queue: A Queue object for exchanging data between
                    process. If None, then the average reward data
                    and optimal action perc data will not be placed
                    on the queue.

        '''
        choices = np.arange(self.n)

        #initialize the preference probabilities as a uniform distribution
        preference_probs = np.ones((self.num_bandits, self.n)) / float(self.n)
        for play_i in range(self.num_plays):
            
            num_optimal_action_taken = 0
            reward_cum = 0.0

            #perform a play for each bandit
            for bandit_i, bandit in enumerate(self.bandits):
                
                #Choose arm from preference probabilities for the given bandit
                choosen_arm = np.random.choice(choices, p=preference_probs[bandit_i, :])
                
                #play this bandit
                reward, num_pulls, opt_action_taken = bandit.play(choosen_arm) 
                reward_cum += reward

                #action-value update
                #update the value for the action using incremental 
                #sample averaging.
                prev_est_val = self.est_value[bandit_i, choosen_arm]
                self.est_value[bandit_i, choosen_arm] = prev_est_val + (1/float(num_pulls)) * (reward - prev_est_val)

                #action preference update using pursuit method
                self.greedy_arms[bandit_i] = np.argmax(self.est_value[bandit_i, :])
                
                for i in range(self.n):
                    
                    #increment the greedy arm preference towards
                    #probability of one
                    prev_prob = preference_probs[bandit_i, i]
                    if(i == self.greedy_arms[bandit_i]):
                        preference_probs[bandit_i, i] = prev_prob + self.beta * (1.0 - prev_prob)

                    #decrement non-greddy  action preferences toward
                    #0
                    else:
                        preference_probs[bandit_i, i] = prev_prob + \
                            self.beta * (0.0 - prev_prob)
                
                if(opt_action_taken):
                    num_optimal_action_taken += 1
            
            #print("Completed Play: ",play_i)

            self.optimal_action_perc[play_i] = float(num_optimal_action_taken) / float(self.num_bandits)
            self.average_reward[play_i] = reward_cum / float(self.num_bandits)

        if(data_queue != None):
            data_queue.put(self.average_reward)
            data_queue.put(self.optimal_action_perc)


if __name__ == "__main__":

    pursuit_method = Pursuit_Method(beta=0.05, num_bandits=500)
    
    pursuit_method.run()

    plt.plot(pursuit_method.average_reward)
    plt.show()