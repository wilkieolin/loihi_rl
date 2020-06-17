import os
import prototypes
from primitives import *
import nxsdk.api.n2a as nx
import numpy as np
import re
from nxsdk.graph.monitor.probes import *

"""
A node which outputs a constant spike train representing the current state
in which the agent finds itself.

Input: Data which determines state (Input image or direct from SNIP)
Output: Spikes representing state
"""
class Decoder(ProcessNode):
    def __init__(self, agent, logicalCore=-1):
        self.n_states = agent.n_states
        self.input_shape = (self.n_states, 1)
        self.output_shape = (self.n_states,1)

        super().__init__(agent.network, self.output_shape, logicalCore)
        self.blocks = {}

        self._create_prototypes()
        self.prototypes['s_prototypes']['activation_conn'] = nx.ConnectionPrototype(weight=self.prototypes['vth'],
                                                                    delay=5)

        self._create_blocks()
        self._connect_blocks()
    
    def _create_blocks(self):
        self.blocks['input_buffer'] = OrNode(self.network, self.output_shape, self.logicalCore)
        self.blocks['input_sum'] = OrNode(self.network, 1, self.logicalCore)
        self.blocks['memory'] = FlipFlopNode(self.network, self.output_shape, self.logicalCore)
        
    def _connect_blocks(self):
        #connect the inputs to the summator
        self.connections['input_inputsum'] = connect_full(self.blocks['input_buffer'].get_outputs(),
                                                    self.blocks['input_sum'].get_inputs(),
                                                    prototype = self.prototypes['s_prototypes']['single'])

        (set_compartment, reset_compartment) = self.blocks['memory'].get_inputs()
        #connect the summator to the memory reset nodes
        self.connections['inputsum_memory_reset'] = connect_full(self.blocks['input_sum'].get_outputs(),
                                                    reset_compartment,
                                                    prototype = self.prototypes['s_prototypes']['single'])
        #connect the input to the memory with a delay to change its state after the reset
        self.connections['input_memory_set'] = connect_one_to_one(self.blocks['input_buffer'].get_outputs(),
                                                    set_compartment,
                                                    prototype = self.prototypes['s_prototypes']['activation_conn'])

    def get_inputs(self):
        return self.blocks['input_buffer'].get_inputs()

    def get_outputs(self):
        return self.blocks['memory'].get_outputs()

    def get_synproto(self):
        return self.prototypes['s_prototypes']['single']