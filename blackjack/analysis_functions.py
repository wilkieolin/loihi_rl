import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

"""
Analysis functions used for the blackjack example.
"""

def read_history(state_actions):
    state_conversion = {0: False, 1: True}
    action_conversion = {0: 'Hit', 1: 'Stick'}
    
    data = {'Player Sum' : state_actions[:,0], 'Dealer Card' : state_actions[:,1]}
    data['Usable Ace'] = [state_conversion[x] for x in state_actions[:,2]]
    data['Action'] = [action_conversion[x] for x in state_actions[:,3]]
    
    df = pd.DataFrame(data)
    return df

def to_gap(rates):
    action_rates = rates.mean(axis=2)
    actions = action_rates[1,...] - action_rates[0,...]
    return actions.reshape(2,10,10)

def normalize(estimates, conditions):
    scale = (conditions['dynrange']) * (2**8-1) * 2**5
    #scale the estimates within [-1,1]
    estimates = (estimates / scale)
    return estimates

def open_julia_series(filename, dataset):
    file = h5py.File(filename,"r")
    dset = file[dataset]
    policy = np.array(dset[()].transpose(3,2,1,0))
    return policy

def open_julia_single(filename, dataset):
    file = h5py.File(filename,"r")
    dset = file[dataset]
    policy = np.array(dset[()]).transpose(2,1,0)
    return policy

def unitary_normalize(x):
    return x * 0.5 + 1

def data_to_values(data):
    estimates = data['final_estimates']
    transform = lambda x: np.mean(normalize(x, data['conditions']),axis=2)
    return list(map(transform, estimates))

def data_to_policy(data):
    estimates = data['final_estimates']
    transform = lambda x: to_gap(normalize(x, data['conditions']))
    return list(map(transform, estimates))

def sigmoid(x,l):
    x = np.clip(x,-50,50)
    return np.exp(l*x)/(np.exp(l*x)+1)

def policy_divergence(policies, optimal, border = 0.01):
    scaled_optimal = unitary_normalize(optimal)
    scaled_policies = list(map(unitary_normalize, policies))
    distances = [dst.jensenshannon(scaled_optimal.ravel() > border, scaled_policies[i].ravel() > border) for i in range(len(policies))]
    return distances

def plot_policy(g):
    g = g.transpose(0,2,1)
    cards = ["A"]
    [cards.append(str(i+2)) for i in range(10)]
    ticks = np.arange(0,10)+0.5
    dynrange = np.min(g), np.max(g)
    
    nua_actions = g[0,:]
    ua_actions = g[1,:]
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(121)
    ax.pcolor(nua_actions, vmin=dynrange[0], vmax=dynrange[1])
    ax.set_title("Non-usable Ace")
    ax.set_xlabel("Dealer showing")
    plt.xticks(ticks, cards)
    ax.set_ylabel("Player Sum")
    plt.yticks(ticks, np.arange(12,22))
    
    ax = fig.add_subplot(122)
    col = ax.pcolor(ua_actions, vmin=dynrange[0], vmax=dynrange[1])
    ax.set_title("Usable Ace")
    ax.set_xlabel("Dealer showing")
    plt.xticks(ticks, cards)
    ax.set_ylabel("Player Sum")
    plt.yticks(ticks, np.arange(12,22))
    
    fig.colorbar(col, ax=ax)


