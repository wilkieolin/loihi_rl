import os
import re
import sys
import numpy as np
sys.path.append("..")

from agent import *

class GridAgent(FullAgent):
    def __init__(self, n_actions, n_states, **kwargs):
        super().__init__(n_actions, n_states)

        self.debug = kwargs.get("debug", False)

    def _create_channels(self):
        assert hasattr(self, 'board'), "Must compile net to board before creating channels."
        assert hasattr(self, 'snip'), "Must add SNIP before creating channels."
        self.outChannels = []
        self.inChannels = []

        #need to send: 
        # keep the setupchannel for later use
        n_outData = (10 + 8 * self.n_actions + 4 * self.n_states)
        setupChannel = self.board.createChannel(b'setupChannel', "int", n_outData)
        setupChannel.connect(None, self.snip)
        self.outChannels.append(setupChannel)

        #create the data channels to return location & action at each epoch
        dataChannel = self.board.createChannel(b'dataChannel', "int", 3 * self.n_epochs)
        dataChannel.connect(self.snip, None)
        self.inChannels.append(dataChannel)

        #return reward at each step
        rewardChannel = self.board.createChannel(b'rewardChannel', "int", 2 * self.n_epochs)
        rewardChannel.connect(self.snip, None)
        self.inChannels.append(rewardChannel)

        #return the action values at each location
        spikeChannel = self.board.createChannel(b'spikeChannel', "int", self.n_epochs * self.n_actions)
        spikeChannel.connect(self.snip, None)
        self.inChannels.append(spikeChannel)

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

        #send the reward stub
        setupChannel.write(4, RPLocations[0])
        #send the punishment stub
        setupChannel.write(4, RPLocations[1])

        #send the action stubs
        for i in range(self.n_actions):
            setupChannel.write(4, actionLocations[i][:4])

        #send the state stubs
        for i in range(self.n_states):
            setupChannel.write(4, stateLocations[i][:4])

        #send the value locations
        for i in range(self.n_actions):
            setupChannel.write(4, valueLocations[i][:4])

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
