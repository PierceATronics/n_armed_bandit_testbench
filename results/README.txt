This directory contains various results from the N-Armed bandit experiments with different action-value methods.

Figure_1.png --> Epsilon-greedy method compared with the Pursuit Method (as presented in Sutton & Barto).
		 For the both methods, action value estimates were updated using sampling-averaging. 
		 Epsilon-greedy - epsilon=0.1		Pursuit Method - beta-0.01
	         num_bandits=1000, num_plays=1000, num_arms=10
		 
		 I noticed that if beta is to high, the pursuit method converges on greedy arm that is
                 not the optimal action. Therefore the optimal action percentage plateaus...effectively
		 learning/exploring stops. So, beta should be fairly low to slow down the convergence.