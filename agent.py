import decoder
import hippocampus
import cortex
import encoder

from abc import ABC, abstractmethod
import numpy as np
import os
import re

import nxsdk.api.n2a as nx
from nxsdk.graph.monitor.probes import *
from nxsdk.graph.processes.phase_enums import Phase
from primitives import connect_one_to_one, dense_along_axis, connect_full, OrNode

"""
Abstract class which defines the necessary parameters for the agent framework.
"""
class Agent(ABC):
    def __init__(self, n_actions, n_states):
        super().__init__()
        #initialize network
        self.network = nx.NxNet()
        self.started = False

        #get problem parameters
        self.n_actions = n_actions
        self.n_states = n_states
        self.n_estimates = n_actions*n_states
        self.shape = (n_actions, n_states)

"""
Used for testing the abstractclass framework alone.
"""
class DummyAgent(Agent):
    def __init__(self, n_actions, n_states):
        super().__init__(n_actions, n_states)

"""
Used to construct an agent which is being tested/used before being ported to the Loihi chip.
"""
class ManualAgent(Agent):
    def __init__(self, n_actions, n_states, **kwargs):
        super().__init__(n_actions, n_states)

        #get estimation parameters
        self.multiplicity = kwargs.get("multiplicity", 1)
        self.l_epoch = kwargs.get("l_epoch", 128)
        self.n_epochs = kwargs.get("n_epochs", 100)
        self.seed = kwargs.get("seed", 341257896)

        self.recordWeights = kwargs.get('recordWeights', False)
        self.recordSpikes = kwargs.get('recordSpikes', False)

        self._create_blocks()
        self._connect_blocks()

    def _create_blocks(self):
        #create the environment-agent interface (decoder)
        self.decoder = decoder.Decoder(self, 0)
        #create the "hippocampus" / state tracer
        self.hippocampus = hippocampus.Hippocampus(self, 1)
        #create the "cortex" / reward estimator
        self.cortex = cortex.Cortex(self, 2)
        #create the action encoder 
        self.encoder = encoder.Encoder(self, 3)

    def _connect_blocks(self):
        self.connections = {}

        hc_SA_in, self.rwd_in, self.pun_in = self.hippocampus.get_inputs()
        hc_SA_proto, self.feedback_proto, _ = self.hippocampus.get_synproto()
        hc_rwd_out, hc_pun_out = self.hippocampus.get_outputs()

        ctx_rwd_in, ctx_pun_in = self.cortex.get_inputs()

        #connect DEC to HC (state -> s/a tracker)
        self.connections['DEC_HC'] = dense_along_axis(self.decoder.get_outputs(),
                                                        self.decoder.output_shape,
                                                        0,
                                                        hc_SA_in,
                                                        self.hippocampus.input_shape,
                                                        1,
                                                        hc_SA_proto)
        #connect DEC to ENC (state -> action select (Q, state))
        self.connections['DEC_ENC'] = dense_along_axis(self.decoder.get_outputs(),
                                                        self.decoder.output_shape,
                                                        0,
                                                        self.encoder.get_inputs(),
                                                        self.encoder.input_shape,
                                                        1,
                                                        self.encoder.get_synproto())
        #connect HC to CTX (reward and punishment feedback on Q)
        self.connections['HC_CTX_rwd'] = connect_one_to_one(hc_rwd_out, ctx_rwd_in, self.cortex.get_synproto())
        self.connections['HC_CTX_pun'] = connect_one_to_one(hc_pun_out, ctx_pun_in, self.cortex.get_synproto())
        #connect CTX to ENC (Q -> action select (Q, state))
        self.connections['CTX_ENC'] = connect_one_to_one(self.cortex.get_outputs(), self.encoder.get_inputs(), self.encoder.get_synproto())
        #connect ENC to HC (if it selecting actions automatically)
        #not yet implemented

"""
Agent class which extends the framework and defines additional functions/variables which are required to port the newtork and execute on Loihi. 
"""
class FullAgent(Agent):
    def __init__(self, n_actions, n_states, **kwargs):
        super().__init__(n_actions, n_states)

        #get estimation parameters
        self.multiplicity = kwargs.get("multiplicity", 1)
        self.l_epoch = kwargs.get("l_epoch", 128)
        self.n_epochs = kwargs.get("n_epochs", 100)
        self.n_replicates = kwargs.get("n_replicates", 1)
        self.dynrange = kwargs.get("dynrange", 1)
        self.n_memories = self.n_estimates * self.n_replicates

        self.noisy = kwargs.get("noisy", False)
        self.seed = kwargs.get("seed", 341257896)

        self.recordWeights = kwargs.get('recordWeights', False)
        self.recordSpikes = kwargs.get('recordSpikes', False)

        self.connections = {}
        self.stubs = {}
        self._create_blocks()
        self._connect_blocks()


    def _create_blocks(self):
        #create the environment-agent interface (decoder)
        self.decoder = decoder.Decoder(self, -1)
        #create the "hippocampus" / state tracer
        self.action_buffer = OrNode(self.network, (self.n_actions,1), -1)
        self.hippocampus = hippocampus.Hippocampus(self, -1)
        #create the "cortex" / reward estimator
        #create the action encoder 
        if self.n_replicates > 1:
            self.cortex = cortex.MultiCortex(self, -1, noisy=self.noisy, n_replicates=self.n_replicates, dynrange=self.dynrange)
            self.encoder= encoder.MultiEncoder(self, -1)
        else:
            self.cortex = cortex.Cortex(self, -1, noisy=self.noisy)
            self.encoder = encoder.Encoder(self, -1)
        

        #create stubs for SNIP
        self.stubs['state'] = self.network.createInputStubGroup(size=self.n_states)
        self.stubs['action'] = self.network.createInputStubGroup(size=self.n_actions)
        self.stubs['reward'] = self.network.createInputStubGroup(size=1)
        self.stubs['punishment'] = self.network.createInputStubGroup(size=1)
        self.stubs['draw'] = self.network.createInputStubGroup(size=1)

    def _connect_blocks(self):
        hc_SA_in, self.rwd_in, self.pun_in = self.hippocampus.get_inputs()
        hc_reset = self.hippocampus.blocks['feedback_sum'].get_inputs()
        hc_SA_proto, self.feedback_proto, _ = self.hippocampus.get_synproto()
        hc_rwd_out, hc_pun_out = self.hippocampus.get_outputs()

        ctx_rwd_in, ctx_pun_in = self.cortex.get_inputs()

        #connect DEC to HC (state -> s/a tracker)
        self.connections['DEC_HC'] = dense_along_axis(self.decoder.get_outputs(),
                                                        self.decoder.output_shape,
                                                        0,
                                                        hc_SA_in,
                                                        self.hippocampus.input_shape,
                                                        1,
                                                        hc_SA_proto)
        #connection action buffer to HC (action -> s/a tracker)
        self.connections['ACT_HC'] = dense_along_axis(self.action_buffer.get_inputs(), 
                                                        self.action_buffer.shape,
                                                        0,
                                                        hc_SA_in, 
                                                        self.hippocampus.input_shape, 
                                                        0,
                                                        hc_SA_proto)
        #connect DEC to ENC (state -> action select (Q, state))
        self.connections['DEC_ENC'] = dense_along_axis(self.decoder.get_outputs(),
                                                        self.decoder.output_shape,
                                                        0,
                                                        self.encoder.get_inputs(),
                                                        self.encoder.input_shape,
                                                        1,
                                                        self.encoder.get_synproto())
        #connect HC to CTX (reward and punishment feedback on Q)
        self.connections['HC_CTX_rwd'] = connect_one_to_one(hc_rwd_out, ctx_rwd_in, self.cortex.get_synproto())
        self.connections['HC_CTX_pun'] = connect_one_to_one(hc_pun_out, ctx_pun_in, self.cortex.get_synproto())
        #connect CTX to ENC (Q -> action select (Q, state))
        self.connections['CTX_ENC'] = connect_one_to_one(self.cortex.get_outputs(), self.encoder.get_inputs(), self.encoder.get_synproto())
        #connect ENC to HC (if it selecting actions automatically)
        #not yet implemented

        #connect the stubs
        self.connections['state_stub_DEC'] = connect_one_to_one(self.stubs['state'],
                                                            self.decoder.get_inputs(),
                                                            self.decoder.get_synproto())

        self.connections['action_stub_ACT'] = connect_one_to_one(self.stubs['action'],
                                                            self.action_buffer.get_inputs(),
                                                            self.action_buffer.get_synproto())

        self.connections['rwd_stub_HC'] = connect_full(self.stubs['reward'], self.rwd_in, self.feedback_proto)
        self.connections['pun_stub_HC'] = connect_full(self.stubs['punishment'], self.pun_in, self.feedback_proto)
        self.connections['draw_stub_HC'] = connect_full(self.stubs['draw'], hc_reset, self.feedback_proto)

    def _compile(self):
        self.compiler = nx.N2Compiler()
        self.board = self.compiler.compile(self.network)
        self.board.sync = True

    @abstractmethod
    def _create_channels(self):
        pass

    @abstractmethod
    def _create_SNIPs(self):
        pass

    @abstractmethod
    def _send_config(self):
        pass

    def _start(self):
        assert hasattr(self, 'board') and hasattr(self, 'snip'), "Must have compiled board and snips before starting."
        self.board.startDriver()

    def get_action_locations(self):
        get_axonid = lambda x: self.connections['action_stub_ACT'][x].inputAxon.nodeId
        get_axon = lambda x: self.network.resourceMap.inputAxon(x)[0]

        axonIds = [get_axonid(x) for x in range(self.n_actions)]
        axons = list(map(get_axon, axonIds))

        return axons 

    def get_estimate_locations(self):
        locs = []

        compartments = self.cortex.blocks['estimates'].compartments['memory']

        for i in range(compartments.numNodes):
            compartmentId = compartments[i].nodeId
            compartmentLoc = self.network.resourceMap.compartmentMap[compartmentId]

            locs.append(compartmentLoc)

        return locs

    def get_state_locations(self):
        get_axonid = lambda x: self.connections['state_stub_DEC'][x].inputAxon.nodeId
        get_axon = lambda x: self.network.resourceMap.inputAxon(x)[0]

        axonIds = [get_axonid(x) for x in range(self.n_states)]
        axons = list(map(get_axon, axonIds))

        return axons 

    def get_value_locations(self):
        locs = []
        compartments = self.encoder.get_outputs()

        for i in range(self.n_actions):
            compartmentId = compartments[i].nodeId
            compartmentLoc = self.network.resourceMap.compartmentMap[compartmentId]

            locs.append(compartmentLoc)

        return locs

    def get_RP_locations(self):
        get_axonid = lambda x: self.connections[x][0].inputAxon.nodeId
        get_axon = lambda x: self.network.resourceMap.inputAxon(x)[0]

        reward = get_axon(get_axonid('rwd_stub_HC'))
        punishment = get_axon(get_axonid('pun_stub_HC'))
        draw = get_axon(get_axonid('draw_stub_HC'))

        return (reward, punishment, draw)

    @abstractmethod
    def get_data(self, n_epochs):
        pass        

    def init(self):
        self._compile()
        self._create_SNIPs()
        self._create_channels()
        
        self.set_params_file()
        self._start()
        self._send_config()

    def run(self):
        #only reserve hardware once we actually need to run the network
        if not self.started:
            self.init()
            self.started = True

        self.board.run(self.l_epoch * self.n_epochs)
        self.get_data(self.n_epochs)

        #return (self.data, self.rewards)
        return (self.data, self.rewards, self.values)

    @abstractmethod
    def set_params_file(self):
        pass
