import nxsdk.api.n2a as nx
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from nxsdk.graph.monitor.probes import SpikeProbeCondition
import pickle as p
sys.path.append("..")

from blackjackAgent import *
from analysis_functions import *

n_cards = 10
n_states = n_cards**2 * 2
n_actions = 2
n_estimates = n_states * n_actions

dynrange = 10
replicates = 3
runs = 100
l_epoch = 128
episodes = int(1e4)
useProbe = False

#init_vals = np.zeros((n_actions,2,10,10,replicates),dtype=np.int)
# init_vals[0,:,:,8:] = -1.0
# init_vals[1,:,:,8:] = 1.0

init_data = p.load(open("blackjack_run13_1.p","rb"))
init_vals = init_data['final_estimates'][-1]

conditions = {'dynrange' : dynrange,
            'replicates' : replicates,
            'l_epoch' : l_epoch,
            'episodes' : episodes,
            'init_policy' : init_vals}

player = BlackjackAgent(n_actions, 
                        n_states, 
                        n_epochs = episodes,
                        l_epoch = l_epoch,
                        starting_values = init_vals,
                        dynrange = dynrange,
                        n_replicates = replicates)


results = {}

results['conditions'] = conditions
results['states'] = []
results['outcomes'] = []
results['action_values'] = []
results['final_estimates'] = []

if (useProbe):
    results['ctx_spks'] = []
    probeCond = SpikeProbeCondition(tStart=(l_epoch*(episodes-1)))
    ctx_spks = player.cortex.get_outputs().probe(nx.ProbeParameter.SPIKE, probeCond)

for i in range(runs):
    result = player.run(episodes)

    results['states'].append(result[0])
    results['outcomes'].append(result[1])
    results['action_values'].append(result[2])
    results['final_estimates'].append(player.final_estimates)
    if (useProbe):
        ctxdata = ctx_spks[0].data
        results['ctx_spks'].append(ctxdata)

player.board.disconnect()

p.dump(results, open("blackjack_run13_2.p", "wb"))