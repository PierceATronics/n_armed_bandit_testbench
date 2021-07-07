from arm import Generic_Test_Bench
import numpy as np
import copy
import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe, Queue

class Epsilon_Greedy:
    '''
    Epsilon Greedy action-value method for RL.
    '''
    def __init__(self, epsilon=0.1, n=10, num_plays=1000, num_bandits=1000):
        '''
        '''
        self.epsilon = epsilon
        self.n = n
        self.num_plays = 1000
        self.num_bandits = num_bandits
        self.bandits = [Generic_Test_Bench(self.n) for i in range(num_bandits)]

        #Keep an estimated value for each arm for each bandit
        self.est_value = np.zeros((self.num_bandits, self.n))

        self.greedy_arms = [0 for i in range(self.num_bandits)]

        self.average_reward = np.zeros(self.num_plays)
        self.optimal_action_perc = np.zeros(self.num_plays)

    def run(self, data_queue):

        '''
        Run all the (num_bandits) bandits for num_plays number of plays.
        Gather the average reward and optimation action percentage 
        data per play

        conn: A child connection for Piping to send data via multiprocessing.
        '''
        for play_i in range(self.num_plays):

            #keep track of how many bandits take the optimal
            #action
            num_optimal_action_taken=0
            reward_cum = 0.0

            #perform a play for each bandit
            for bandit_i, bandit in enumerate(self.bandits):
                
                #Action selection based on the greedy selection.
                choosen_arm = copy.deepcopy(self.greedy_arms[bandit_i])
                prob_explore = np.random.uniform()

                #probability epsilon of exploring a non-greedy arm
                if(prob_explore <= self.epsilon):
                    while(True):
                        explore_arm = np.random.randint(0, self.n)
                        if(explore_arm == self.greedy_arms[bandit_i]):
                            continue
                        else:
                            choosen_arm = explore_arm
                            break
                
                #play this bandit
                reward, num_pulls, opt_action_taken = bandit.play(choosen_arm) 
                reward_cum += reward
                
                #update the value for the action using incremental 
                #sample averaging.
                prev_est_val = self.est_value[bandit_i, choosen_arm]
                self.est_value[bandit_i, choosen_arm] = prev_est_val + (1/float(num_pulls)) * (reward - prev_est_val)

                #update which arm is now the greedy arm to choose
                self.greedy_arms[bandit_i] = np.argmax(self.est_value[bandit_i, :])

                if(opt_action_taken):
                    num_optimal_action_taken += 1
            
            self.optimal_action_perc[play_i] = float(num_optimal_action_taken) / float(self.num_bandits)
            self.average_reward[play_i] = reward_cum / float(self.num_bandits)
        
        data_queue.put(self.average_reward)
        data_queue.put(self.optimal_action_perc)
        

if __name__ == "__main__":
    ep_greedy_1 = Epsilon_Greedy(0.1, 10, 1000, 1000)
    ep_greedy_2 = Epsilon_Greedy(0.01, 10, 1000, 1000)

    #q1, q2 = Queue(), Queue()
    #ep_greedy_1.run(q1)
    #ep_greedy_2.run(q2)
    
    q1, q2 = Queue(), Queue()
    
    p1 = Process(target=ep_greedy_1.run, args=(q1,))
    p2 = Process(target=ep_greedy_2.run, args=(q2,))
    p1.start(); p2.start()
    average_reward_1 = q1.get()
    optimal_action_perc_1 = q1.get()
    average_reward_2 = q2.get()
    optimal_action_perc_2 = q2.get()
    p1.join()
    p2.join()
    
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(average_reward_1)
    ax1.plot(average_reward_2)
    ax1.set_ylabel("Average Reward")

    ax2.plot(optimal_action_perc_1)
    ax2.plot(optimal_action_perc_2)
    ax2.set_ylabel("% Optimal Action Taken")
    ax2.set_xlabel("Plays")
    plt.show()
    