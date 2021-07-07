'''
This script is used to compare the action-value
methods using the n-armed bandit testbed.
'''

from epsilon_greedy import Epsilon_Greedy
from pursuit_method import Pursuit_Method

from multiprocessing import Process, Queue
import matplotlib.pyplot as plt


if __name__ == "__main__":
    num_bandits = 1000
    num_plays = 1000
    n=10

    #Compare standard e-greedy with pursuit method
    ep_greedy_1 = Epsilon_Greedy(epsilon=0.1, n=n, num_plays=num_plays, num_bandits=num_bandits)

    pursuit_agent = Pursuit_Method(beta=0.01, n=n, num_plays=num_plays, num_bandits=num_bandits)

    #Multiprocessing used to speed up evaluation
    q1, q2 = Queue(), Queue()

    p1 = Process(target=ep_greedy_1.run, args=(q1,))
    p2 = Process(target=pursuit_agent.run, args=(q2,))

    p1.start(); p2.start()
    ep_1_avg_reward = q1.get()
    ep_1_opt_action_perc = q1.get()
    pursuit_avg_reward = q2.get()
    pursuit_opt_action_perc = q2.get()
    p1.join()
    p2.join()

    #Plot the results
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(ep_1_avg_reward, label="e-greedy(0.1)")
    ax1.plot(pursuit_avg_reward, label="pursuit(beta = 0.01)")
    ax1.legend()
    ax1.set_ylabel("Average Reward")

    ax2.plot(ep_1_opt_action_perc)
    ax2.plot(pursuit_opt_action_perc)
    ax2.set_ylabel("% Optimal Action Taken")
    ax2.set_xlabel("Plays")

    plt.show()



