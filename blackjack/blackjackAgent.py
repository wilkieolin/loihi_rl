import os
import re
import sys
sys.path.append("..")
from agent import *
import numpy as np

"""
Agent which is used for the blackjack task.
"""
class BlackjackAgent(FullAgent):
    def __init__(self, n_actions, n_states, **kwargs):
        #print(n_action, n_states)
        super().__init__(n_actions, n_states, **kwargs)
        #get starting values to form the initial greedy policy
        self.start_values = kwargs.get("starting_values", np.zeros((2,2,10,10,self.n_replicates), dtype='int'))
        #scale values from (-1,1) to the dynamic memory range & convert to int

        self.debug = kwargs.get("debug", False)


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

        #return the estimates for each pair at the end
        estimateChannel = self.board.createChannel(b'estimateChannel', "int", self.n_memories)
        connect(False, estimateChannel)

    def _create_SNIPs(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating SNIP."
        includeDir = os.getcwd()
        self.snip = self.board.createSnip(Phase.EMBEDDED_MGMT,
                                    includeDir=includeDir,
                                    cFilePath = includeDir + "/management.c",
                                    funcName = "run_cycle",
                                    guardName = "check")

    def get_data(self, n_epochs):
        dataChannel = self.inChannels[0]
        rewardChannel = self.inChannels[1]
        spikeChannel = self.inChannels[2]
        estimateChannel = self.inChannels[3]

        #get the state/action data
        self.data = np.array(dataChannel.read(n_epochs*self.data_points)).reshape(n_epochs, self.data_points)
        #get the rewards data
        self.rewards = np.array(rewardChannel.read(n_epochs))
        singlewgt = self.action_buffer.prototypes['s_prototypes']['single'].weight
        #get the action value data
        self.values = np.array(spikeChannel.read(n_epochs*self.n_actions), dtype='int').reshape(n_epochs, self.n_actions)
        #get the final estimate values
        self.final_estimates = np.array(estimateChannel.read(self.n_memories), dtype='int').reshape(self.n_actions, self.n_states, self.n_replicates)

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

        #send the number of epochs to run (tell program when to probe/send final values)
        setupChannel.write(1, [self.n_epochs])

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
        setupChannel.write(self.n_memories, self.start_values.ravel(order='c'))

    def set_params_file(self):
        filename = os.getcwd()+'/parameters.h'

        with open(filename) as f:
            data = f.readlines()

        f = open(filename, "w")
        for line in data:

            #update debug
            m = re.match(r'^#define\s+DEBUG', line)
            if m is not None:
                line = '#define DEBUG ' + str(int(self.debug)) + '\n'

            #update n_replicates
            m = re.match(r'^#define\s+N_REPLICATES', line)
            if m is not None:
                line = '#define N_REPLICATES ' + str(int(self.n_replicates)) + '\n'

            f.write(line)

        f.close()