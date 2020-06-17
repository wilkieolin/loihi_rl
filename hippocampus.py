import os
import prototypes
from primitives import *
import nxsdk.api.n2a as nx
import numpy as np
import re
from nxsdk.graph.monitor.probes import *

"""
Tracks which states the agent has entered and delivers reward/punishment feedback for those
states' representations in cortex once reward/punishment is received. 
"""
class Hippocampus(ProcessNode):
    def __init__(self, agent, logicalCore=-1):
        self.n_actions = agent.n_actions
        self.n_states = agent.n_states
        self.input_shape = agent.shape
        self.output_shape = agent.shape

        super().__init__(agent.network, self.output_shape, logicalCore)
        self.blocks = {}
        self.s_prototypes = {}

        self._create_blocks()
        self._add_prototypes()
        self._connect_blocks()

    def _add_prototypes(self):
        or_w = self.blocks['reward_buffer'].get_synproto().weight
        and_w = self.blocks['reward_gate'].get_synproto().weight

        self.s_prototypes['or_delayed'] = nx.ConnectionPrototype(weight = or_w, delay=3)
        self.s_prototypes['or_delayed_long'] = nx.ConnectionPrototype(weight = or_w, delay=6)
        self.s_prototypes['and_delayed'] = nx.ConnectionPrototype(weight = and_w, delay=5)

    def _create_blocks(self):
        self.blocks['filter'] = AndNode(self.network, self.input_shape, self.logicalCore)
        self.blocks['state_memory'] = FlipFlopNode(self.network, self.input_shape, self.logicalCore)

        self.blocks['reward_buffer'] = OrNode(self.network, 1, self.logicalCore)
        self.blocks['punishment_buffer'] = OrNode(self.network, 1, self.logicalCore)
        self.blocks['feedback_sum'] = OrNode(self.network, 1, self.logicalCore)
        self.blocks['feedback_delay'] = OrNode(self.network, 1, self.logicalCore)

        self.blocks['reward_gate'] = AndNode(self.network, self.input_shape, self.logicalCore)
        self.blocks['punishment_gate'] = AndNode(self.network, self.input_shape, self.logicalCore)

    def _connect_blocks(self):
        (excite_smemory, inhibit_smemory) = self.blocks['state_memory'].get_inputs()
       
        #Main blocks
        #connect the current state gated by action to the state/action memory block
        self.connections['filter_smemory'] = connect_one_to_one(self.blocks['filter'].get_outputs(),
                                                                excite_smemory,
                                                                self.blocks['state_memory'].get_synproto())

        #connect the state/action memory block to the reward-gated output
        self.connections['smemory_rgate'] = connect_one_to_one(self.blocks['state_memory'].get_outputs(),
                                                                self.blocks['reward_gate'].get_inputs(),
                                                                self.blocks['reward_gate'].get_synproto())

        #connect the state/action memory block to the punishment-gated output
        self.connections['smemory_pgate'] = connect_one_to_one(self.blocks['state_memory'].get_outputs(),
                                                                self.blocks['punishment_gate'].get_inputs(),
                                                                self.blocks['punishment_gate'].get_synproto())
        #Auxiliary reward/punishment blocks
        #connect the reward buffer to the reward-gated output
        self.connections['reward_rgate'] = connect_full(self.blocks['reward_buffer'].get_outputs(),
                                                        self.blocks['reward_gate'].get_inputs(),
                                                        self.s_prototypes['and_delayed'])

        #connect the punishment buffer to the punishment-gated output
        self.connections['punishment_pgate'] = connect_full(self.blocks['punishment_buffer'].get_outputs(),
                                                        self.blocks['punishment_gate'].get_inputs(),
                                                        self.s_prototypes['and_delayed'])

        #connect the reward to the feedback summator
        self.connections['reward_feedbacksum'] = connect_full(self.blocks['reward_buffer'].get_outputs(),
                                                        self.blocks['feedback_sum'].get_inputs(),
                                                        self.s_prototypes['or_delayed'])

        #connect the reward to the feedback summator
        self.connections['punishment_feedbacksum'] = connect_full(self.blocks['punishment_buffer'].get_outputs(),
                                                        self.blocks['feedback_sum'].get_inputs(),
                                                        self.s_prototypes['or_delayed'])

        #connect the reward to the feedback summator's reset
        self.connections['feedbacksum_smemory'] = connect_full(self.blocks['feedback_sum'].get_outputs(),
                                                        inhibit_smemory,
                                                        self.s_prototypes['or_delayed_long'])

        # #connect the reward to the feedback summator's reset
        # self.connections['feedbacksum_delay'] = connect_full(self.blocks['feedback_sum'].get_outputs(),
        #                                                 self.blocks['feedback_delay'].get_inputs(),
        #                                                 self.s_prototypes['or_delayed_long'])

        # #add a delay neuron to make the timing work out...
        # self.connections['delay_smemory'] = connect_full(self.blocks['feedback_delay'].get_outputs(),
        #                                                 inhibit_smemory,
        #                                                 self.s_prototypes['or_delayed_long'])

        
    def get_inputs(self):
        return self.blocks['filter'].get_inputs(), self.blocks['reward_buffer'].get_inputs(), self.blocks['punishment_buffer'].get_inputs()

    def get_outputs(self):
        return self.blocks['reward_gate'].get_outputs(), self.blocks['punishment_gate'].get_outputs()

    def get_synproto(self):
        #delayed = self.blocks['reward_buffer'].prototypes['s_prototypes']['hc_delayed']
        return self.blocks['filter'].get_synproto(), self.blocks['reward_buffer'].get_synproto(), self.blocks['punishment_buffer'].get_synproto()