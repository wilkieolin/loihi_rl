import os
import prototypes
from primitives import *
import nxsdk.api.n2a as nx
import numpy as np
import re
from nxsdk.graph.monitor.probes import *

"""
Takes the reward estimates from the cortex and selects only action estimates relevant to current state,
"encodes" the rewards into a decision on what action to take that is fed back into the environment.
"""
class Encoder(ProcessNode):
    def __init__(self, agent, logicalCore=-1):
        self.n_actions = agent.n_actions
        self.n_states = agent.n_states
        self.input_shape = agent.shape
        self.output_shape = (self.n_actions,1)

        super().__init__(agent.network, self.output_shape, logicalCore)
        self.blocks = {}

        self._create_blocks()
        self._connect_blocks()

    def _create_blocks(self):
        self.blocks['filter'] = AndNode(self.network, self.input_shape, self.logicalCore)
        self.blocks['summator'] = OrNode(self.network, self.output_shape, self.logicalCore)
        self.blocks['counter'] = CounterNode(self.network, self.output_shape, self.logicalCore)

    def _connect_blocks(self):
        #connect the summator along the action axis (over all possible actions,
        # filtered by active state)
        self.connections['filter_summator'] = dense_along_axis(self.blocks['filter'].get_outputs(), 
                                                            self.blocks['filter'].shape,
                                                            0,
                                                            self.blocks['summator'].get_inputs(),
                                                            self.blocks['summator'].shape,
                                                            0,
                                                            prototype=self.blocks['summator'].get_synproto())

        self.connections['summator_counter'] = connect_one_to_one(self.blocks['summator'].get_outputs(),
                                                                self.blocks['counter'].get_inputs(),
                                                                prototype=self.blocks['counter'].get_synproto())

    def get_inputs(self):
        return self.blocks['filter'].get_inputs()

    def get_outputs(self):
        return self.blocks['counter'].get_outputs()

    def get_synproto(self):
        return self.blocks['filter'].get_synproto()

class MultiEncoder(ProcessNode):
    def __init__(self, agent, logicalCore=-1):
        self.n_actions = agent.n_actions
        self.n_states = agent.n_states
        self.input_shape = (self.n_actions, self.n_states, agent.n_replicates)
        self.output_shape = (self.n_actions,1)

        super().__init__(agent.network, self.output_shape, logicalCore)
        self.blocks = {}

        self._create_blocks()
        self._connect_blocks()

    def _create_blocks(self):
        self.blocks['filter'] = AndNode(self.network, self.input_shape, self.logicalCore)
        self.blocks['counter'] = CounterNode(self.network, self.output_shape, self.logicalCore)

    def _connect_blocks(self):
        #connect the summator along the action axis (over all possible actions,
        # filtered by active state)
        self.connections['filter_counter'] = dense_along_axis(self.blocks['filter'].get_outputs(),
                                                            self.input_shape,
                                                            0,
                                                            self.blocks['counter'].get_inputs(),
                                                            self.output_shape,
                                                            0,
                                                            prototype=self.blocks['counter'].get_synproto())

    def get_inputs(self):
        return self.blocks['filter'].get_inputs()

    def get_outputs(self):
        return self.blocks['counter'].get_outputs()

    def get_synproto(self):
        return self.blocks['filter'].get_synproto()
