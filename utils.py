def create_state_spikes(stategen, states, time_per_state, delay=1):
    n_states = stategen.numPorts
    s_times = [[] for i in range(n_states)]
    
    for (i,a) in enumerate(states):
        offset = i*time_per_state
        s_times[a].append(delay + offset)
        
    for i in range(n_states):
        stategen.addSpikes([i], [s_times[i]])

def create_action_spikes(actiongen, actions, time_per_state, delay=8):
    n_actions = actiongen.numPorts
    a_times = [[] for i in range(n_actions)]
    
    for (i,a) in enumerate(actions):
        offset = i*time_per_state
        a_times[a].append(time_per_state - delay + offset)
        
    for i in range(n_actions):
        actiongen.addSpikes([i], [a_times[i]])

def create_feedback_spikes(rwdgen, pungen, feedback, time_per_state, delay=2):
    r_times = []
    p_times = []
    
    for (i,f) in enumerate(feedback):
        offset = i*time_per_state
        time = time_per_state - delay + offset
        if f == -1:
            p_times.append(time)
        else:
            r_times.append(time)
    
    rwdgen.addSpikes([0], [r_times])
    pungen.addSpikes([0], [p_times])