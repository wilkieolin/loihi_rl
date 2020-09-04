import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_history(state_actions):
    conversion = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}
    
    data = {'X' : state_actions[:,0], 'Y' : state_actions[:,1]}
    data['Action'] = [conversion[x] for x in state_actions[:,2]]
    
    df = pd.DataFrame(data)
    return df

def plot_values(qdata, scale = 0.4, greedy=False, grid_size = (5,5)):
    (grid_x, grid_y) = grid_size
    fig, ax = plt.subplots(figsize=(10,10),dpi=100)
    
    qvals = np.mean(qdata, axis=2).transpose()
    rwd_ind = (2) * grid_x + 2
    
    n_states = qvals.shape[0]
    n_actions = qvals.shape[1]
    
    plt.xlim(0,grid_x+1)
    plt.ylim(0,grid_y+1)

    for i in range(n_states):
        x = i % grid_x + 1
        y = i // grid_x + 1
        
        maxq = np.max(qvals[i,:])
        minq = np.min(qvals[i,:])
        dr = maxq - minq
        
        def normalize(x):
            if dr > 0:
                return (x - minq)/dr
            else:
                return 0
            
        #plot a circle if we're at the reward loc
        if i == rwd_ind:
            xx = np.linspace(0,2*np.pi,100)
            xs = np.sin(xx)*scale + x
            ys = np.cos(xx)*scale + y
            plt.plot(xs, ys)
            plt.text(x, y, "Reward", horizontalalignment='center')
            continue
        
        for j in range(n_actions):
            dx = 0
            dy = 0
            q = qvals[i,j]
            
            if greedy and q != maxq:
                continue
            
            if j == 0:
                #NORTH
                dy = scale*normalize(q)
                c='black'
            elif j == 1:
                #EAST
                dx = scale*normalize(q)
                c='g'
            elif j == 2:
                #SOUTH
                dy = -scale*normalize(q)
                c='b'
            else:
                #WEST
                dx = -scale*normalize(q)
                c='r'
                
            plt.arrow(x,y,dx,dy,color=c,width=0.02)