import numpy as np
import random

def avgmoa(choices, probs):
    trials = choices.shape[0]
    bestarm = np.argmax(probs)
    right = np.sum(choices == bestarm)
    return right/trials

def egreedy(epochs, epsilon, alpha, probs):
    n_arms = len(probs)
    q = np.ones(n_arms) * 0.5
    choice = -1
    
    estimates = np.ones((epochs, n_arms)) 
    choices = np.zeros(epochs)
    rewards = np.zeros(epochs)
    
    for i in range(epochs):
        if random.random() < epsilon:
            #choose randomly
            choice = random.choice(range(n_arms))
        else:
            #choose highest, breaking ties randomly
            maxima = np.argwhere(q == np.amax(q))
            choice = random.choice(maxima)
            
        #sample the reward for the chosen arm
        if random.random() < probs[choice]:
            reward = 1
        else:
            reward = 0
            
        q[choice] = q[choice] + alpha*(reward - q[choice])
        
        choices[i] = choice
        rewards[i] = reward
        estimates[i,:] = q
    
    return (choices, rewards, estimates)

"""
Choose a distribution of probabilities with one 'arm' the highest, another close behind it (by the 'gap'),
and the rest falling under the second higheset and minimum.
"""
def get_rewards(rand_state, n_actions, lowest_r, highest_r, gap):
    
    highest = rand_state.randint(lowest_r, highest_r, 1)
    second_highest = highest - gap
    rest = rand_state.randint(0, second_highest, n_actions-2)
    
    probabilities = np.concatenate((highest, second_highest, rest))
    rand_state.shuffle(probabilities)
    
    return probabilities/100

def plot_avg_reward(rewards, **kwargs):
    plt.figure(**kwargs)
    plt.plot(np.convolve((rewards+1)*0.5, np.ones(100), mode='valid')/100)
    plt.xlabel("Epoch")
    plt.ylabel("Average reward over 100 epochs")

def plot_choices(actions, probabilities, **kwargs):
    episodes = actions.shape[0]
    optimal_arm = np.argmax(probabilities)

    plt.figure(**kwargs)

    plt.subplot(121)
    plt.scatter(np.arange(episodes), actions)
    plt.xlabel("Epoch")
    plt.ylabel("Arm Chosen")

    plt.subplot(122)
    plt.plot(np.convolve((actions == optimal_arm).ravel()*1, np.ones(100), mode='valid')/100)
    plt.xlabel("Epoch")
    plt.ylabel("Mean optimal action over 100 epochs")

def run_epsilon_trials(probabilities, epsilons, trials=5, episodes=1000):
    n_epsilons = len(epsilons)

    results = {}
    makearr = lambda: np.zeros((n_epsilons, trials))
    loihi_moa = makearr()
    loihi_rwd = makearr()
    
    cpu_moa = makearr()
    cpu_rwd = makearr()
    
    INTMAX = np.iinfo(np.int32).max
    
    for (i, epsilon) in enumerate(epsilons):
        for t in range(trials):
            #run the Loihi trial with a new seed
            seed = np.random.randint(INTMAX)
            bandit = Bandit(probabilities, epsilon=epsilon, l_epoch = l_epoch, n_replicates = 2, n_epochs = episodes, seed=seed)
            actions, rewards, _ = bandit.run()
            bandit.board.disconnect()
            
            loihi_moa[i,t] = avgmoa(actions, probabilities)
            #remap loihi (-1, 1) rewards to (0,1) before averaging
            loihi_rwd[i,t] = np.mean(rewards == 1)
            
            #run the CPU bandit, seed automatically advances
            actions, rewards, _ = egreedy(episodes, epsilon, 0.01, probabilities)
            cpu_moa[i,t] = avgmoa(actions, probabilities)
            cpu_rwd[i,t] = np.mean(rewards)
            
    results['loihi_moa'] = loihi_moa
    results['loihi_rwd'] = loihi_rwd
    results['cpu_moa'] = cpu_moa
    results['cpu_rwd'] = cpu_rwd
    
    return results
            
            