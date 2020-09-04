import os
import re
import sys
import numpy as np
sys.path.append("..")

from agent import *


"""
Agent which is used for the multi-arm bandit task.
"""
class Bandit(FullAgent):
    def __init__(self, probabilities, **kwargs):
        #force n_states to 1 for a non-conditional bandit
        n_actions = len(probabilities)
        n_states = 1
        #print(n_action, n_states)
        super().__init__(n_actions, n_states, **kwargs)
        #get starting values to form the initial greedy policy
        self.start_values = kwargs.get("starting_values", np.zeros((n_actions,1), dtype='float'))
        #scale values from (-1,1) to the dynamic memory range & convert to int
        self.start_values *= self.dynrange*127*2**6
        self.start_values = self.start_values.astype('int')

        #set the chance a non-optimal action will be sampled (epsilon)
        self.epsilon = int(kwargs.get("epsilon", 0.10) * 100)
        #initialize the reward probabilities
        self.probabilities = np.array(probabilities) * 100
        self.probabilities = self.probabilities.astype('int')
        #debug flag which will print out each epoch in detail
        self.debug = kwargs.get("debug", False)
        #the number of data points which will be read from the board each epoch
        self.data_points = 1

    def _create_SNIPs(self):
        #create the SNIP which will manage rewards and actions
        assert hasattr(self, 'board'), "Must compile net to board before creating SNIP."
        includeDir = os.getcwd()
        self.snip = self.board.createSnip(Phase.EMBEDDED_MGMT,
                                    includeDir=includeDir,
                                    cFilePath = includeDir + "/management.c",
                                    funcName = "run_cycle",
                                    guardName = "check")

    def _create_channels(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating channels."
        assert hasattr(self, 'snip'), "Must add SNIP before creating channels."
        self.outChannels = []
        self.inChannels = []

        def connect(out, channel):
            if out:
                self.outChannels.append(channel)
                channel.connect(None, self.snip)
            else:
                self.inChannels.append(channel)
                channel.connect(self.snip, None)

        #need to send: 
        # keep the setupchannel for later use
        n_outData = (10 + 9 * self.n_actions + 4 * self.n_states + 5 * self.n_memories)
        setupChannel = self.board.createChannel(b'setupChannel', "int", n_outData)
        connect(True, setupChannel)

        #create the data channels to return location & action at each epoch
        dataChannel = self.board.createChannel(b'dataChannel', "int", self.data_points * self.n_epochs)
        connect(False, dataChannel)

        #return reward at each step
        rewardChannel = self.board.createChannel(b'rewardChannel', "int", self.n_epochs)
        connect(False, rewardChannel)

        #return the action values at each location
        spikeChannel = self.board.createChannel(b'spikeChannel', "int", self.n_epochs * self.n_actions)
        connect(False, spikeChannel)

    def get_data(self, n_epochs):
        dataChannel = self.inChannels[0]
        rewardChannel = self.inChannels[1]
        spikeChannel = self.inChannels[2]

        #get the state/action data
        self.data = np.array(dataChannel.read(n_epochs*self.data_points)).reshape(n_epochs, self.data_points)
        #get the rewards data
        self.rewards = np.array(rewardChannel.read(n_epochs))
        singlewgt = self.action_buffer.prototypes['s_prototypes']['single'].weight
        #get the action value data
        self.values = np.array(spikeChannel.read(n_epochs*self.n_actions), dtype='int').reshape(n_epochs, self.n_actions)/(singlewgt*2**6)


    def _send_config(self):
        #get the locations of axons where we need to send updates to/from the SNIP
        stateLocations = self.get_state_locations()
        actionLocations = self.get_action_locations()
        valueLocations = self.get_value_locations()
        RPLocations = self.get_RP_locations()

        #get the setup channel
        setupChannel = self.outChannels[0]
        #send the epoch length
        setupChannel.write(1, [self.l_epoch])

        #send the random seed
        setupChannel.write(1, [self.seed])

        #send the probability distributions
        for i in range(self.n_actions):
            setupChannel.write(1, [self.probabilities[i]])

        #send the reward stub
        setupChannel.write(4, RPLocations[0])
        #send the punishment stub
        setupChannel.write(4, RPLocations[1])
        #send the draw stub
        setupChannel.write(4, RPLocations[2])

        #send the action stubs
        for i in range(self.n_actions):
            setupChannel.write(4, actionLocations[i][:4])

        #send the state stubs
        for i in range(self.n_states):
            setupChannel.write(4, stateLocations[i][:4])

        #send the value locations
        for i in range(self.n_actions):
            setupChannel.write(4, valueLocations[i][:4])

        estimateLocations = self.get_estimate_locations()
        #send the estimate locations
        for i in range(self.n_memories):
            setupChannel.write(4, estimateLocations[i][:4])
        
        #send the initial values that will be used to form the policy
        for i in range(self.n_replicates):
            setupChannel.write(self.n_estimates, self.start_values.ravel(order='c'))

    def set_params_file(self):
        filename = os.getcwd()+'/parameters.h'

        with open(filename) as f:
            data = f.readlines()

        f = open(filename, "w")
        for line in data:

            #update numarms
            m = re.match(r'^#define\s+N_ACTIONS', line)
            if m is not None:
                line = '#define N_ACTIONS ' + str(self.n_actions) + '\n'

            #update epsilon
            m = re.match(r'^#define\s+EPSILON', line)
            if m is not None:
                line = '#define EPSILON ' + str(self.epsilon) + '\n'

            f.write(line)

        f.close()