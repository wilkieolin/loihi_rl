import os
import prototypes
from primitives import *
import nxsdk.api.n2a as nx
import numpy as np
import re
from nxsdk.graph.monitor.probes import *

class Cortex(ProcessNode):
    def __init__(self, agent, logicalCore=-1, **kwargs):
        self.n_actions = agent.n_actions
        self.n_states = agent.n_states
        self.input_shape = agent.shape
        self.output_shape = agent.shape
        self.noisy = kwargs.get("noisy", False)

        super().__init__(agent.network, self.output_shape, logicalCore)
        self.blocks = {}

        self._create_prototypes()
        self._create_blocks()
    
    def _create_blocks(self):
        self.blocks['estimates'] = TrackerNode(self.network, self.output_shape, self.logicalCore, noisy=self.noisy)

    def get_inputs(self):
        return self.blocks['estimates'].get_inputs()

    def get_outputs(self):
        return self.blocks['estimates'].get_outputs()

    def get_synproto(self):
        return self.blocks['estimates'].get_synproto()

class MultiCortex(ProcessNode):
    def __init__(self, agent, logicalCore=-1, **kwargs):
        self.n_actions = agent.n_actions
        self.n_states = agent.n_states
        self.input_shape = agent.shape

        self.n_replicates = kwargs.get("n_replicates", 2)
        self.dynrange = kwargs.get("dynrange", 1)
        self.output_shape =  (self.n_actions, self.n_states, self.n_replicates)

        super().__init__(agent.network, self.output_shape, logicalCore)
        self.blocks = {}

        self._create_prototypes()
        self._create_blocks()
        self._connect_blocks()
    
    def _create_blocks(self):
        self.blocks['excite_buffer'] = OrNode(self.network, self.input_shape, self.logicalCore)
        self.blocks['inhibit_buffer'] = OrNode(self.network, self.input_shape, self.logicalCore)
        self.blocks['estimates'] = TrackerNode(self.network, self.output_shape, self.logicalCore, noisy=True, dynrange=self.dynrange)
    
    def _connect_blocks(self):
        tracker = self.blocks['estimates']
        tracker_excite, tracker_inhibit = tracker.get_inputs()
        synproto = tracker.get_synproto()

        #connect the buffers to each of the replicated trackers
        self.connections['excite_trackers'] = expand_along_axis(self.blocks['excite_buffer'].get_outputs(), self.input_shape, tracker_excite, self.output_shape, 2, synproto)
        self.connections['inhibit_trackers'] = expand_along_axis(self.blocks['inhibit_buffer'].get_outputs(), self.input_shape, tracker_inhibit, self.output_shape, 2, synproto)

    def get_inputs(self):
        return self.blocks['excite_buffer'].get_inputs(), self.blocks['inhibit_buffer'].get_inputs()

    def get_outputs(self):
        return self.blocks['estimates'].get_outputs()

    def get_synproto(self):
        return self.blocks['excite_buffer'].get_synproto()