from abc import ABC, abstractmethod
import os
import prototypes
import nxsdk.api.n2a as nx
import numpy as np
import re
from functools import reduce

"""
Abstract class which declares the functions and parameters which all nodes in the spiking computation graph
must define.
"""
class ProcessNode(ABC):
    def __init__(self, network, shape, logicalCore=-1):
        super().__init__()
        self.network = network
        self.shape = shape
        self.numNodes = np.prod(shape)
        self.logicalCore = logicalCore

        self.compartments = {}
        self.connections = {}
        self.neurons = {}
        self.stubs = {}

    def _create_prototypes(self, **kwargs):
        self.prototypes = prototypes.create_prototypes(logicalCoreId=self.logicalCore, **kwargs)

    #Returns the shape of the node
    def get_shape(self):
        return self.shape

    #Get the compartments which represent the node's output(s)
    @abstractmethod
    def get_outputs(self):
        pass

    #Get the compartments which represent the node's input(s)
    @abstractmethod
    def get_inputs(self):
        pass

    #Return a synaptic prototype which may be required for the node to operate correctly (e.g. weight for 3-way AND)
    @abstractmethod
    def get_synproto(self):
        pass

"""
A flip-flop like node which will spike all the time after reciving a spike on its 'excite' compartment
and return to a quiet state after reciving a spike on its 'inhibit' compartment.
"""
class FlipFlopNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)

        self._create_prototypes()
        self._create_compartments()
        self._create_connections()


    def _create_compartments(self):
        self.neurons['flipflops'] = self.network.createNeuronGroup(size=self.numNodes,
                                                prototype=self.prototypes['n_prototypes']['ffProto'])
        self.compartments['soma'] = self.neurons['flipflops'].soma
        self.compartments['memory'] = self.neurons['flipflops'].dendrites[0]

        self.compartments['inverter'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                prototype=self.prototypes['c_prototypes']['invProto'])

        self.compartments['excite_ands'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                prototype=self.prototypes['c_prototypes']['andProto'])

        self.compartments['inhibit_ands'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                prototype=self.prototypes['c_prototypes']['andProto'])

        self.compartments['excite'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                prototype=self.prototypes['c_prototypes']['bufferProto'])

        self.compartments['inhibit'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                prototype=self.prototypes['c_prototypes']['bufferProto'])

    def _create_connections(self):
        #connect output to the inverter input
        self.connections['soma_inv'] = connect_one_to_one(self.compartments['soma'], self.compartments['inverter'], self.prototypes['s_prototypes']['invconn'])
        #connect the excite/inhibit buffers to their respective ands
        self.connections['excite_exciteands'] = connect_one_to_one(self.compartments['excite'], self.compartments['excite_ands'], self.prototypes['s_prototypes']['halfconn'])
        self.connections['inhibit_inhands'] = connect_one_to_one(self.compartments['inhibit'], self.compartments['inhibit_ands'], self.prototypes['s_prototypes']['halfconn'])
        #connect the output gating to the excite/inhibit ands
        self.connections['inv_exciteands'] = connect_one_to_one(self.compartments['inverter'], self.compartments['excite_ands'], self.prototypes['s_prototypes']['halfconn'])
        self.connections['soma_inhands'] = connect_one_to_one(self.compartments['soma'], self.compartments['inhibit_ands'], self.prototypes['s_prototypes']['halfconn'])
        #connect the ands' output to the memory compartment
        self.connections['exciteand_memory'] = connect_one_to_one(self.compartments['excite_ands'], self.compartments['memory'], self.prototypes['s_prototypes']['econn'])
        self.connections['inhand_memory'] = connect_one_to_one(self.compartments['inhibit_ands'], self.compartments['memory'], self.prototypes['s_prototypes']['iconn'])

    def get_outputs(self):
        return self.compartments['soma']

    def get_inputs(self):
        return self.compartments['excite'], self.compartments['inhibit']

    def get_synproto(self):
        return self.prototypes['s_prototypes']['single']

"""
3-compartment neuron which integrates the currents provided at the input stub and does a soft reset
when the integrator compartment fires. 
"""
class SoftResetNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)
        self.noisy = int(kwargs.get("noisy", False))
        self.vth = int(kwargs.get("vth", 255))
        
        self._create_prototypes(noisy=self.noisy, vth=self.vth)
        self._create_compartments()
        self._create_connections()
        
    def _create_compartments(self):
        self.neurons['srneurons'] = self.network.createNeuronGroup(size=self.numNodes,
                                                            prototype=self.prototypes['n_prototypes']['srProto'])

        self.compartments['soma'] = self.neurons['srneurons'].soma
        self.compartments['integrator'] = self.neurons['srneurons'].dendrites[0]
        self.compartments['input'] = self.neurons['srneurons'].dendrites[0].dendrites[0]

    def _create_connections(self):
       
        #connect the soma output to soft reset the integrator
        self.connections['softreset'] = connect_one_to_one(self.compartments['soma'], self.compartments['integrator'], self.prototypes['s_prototypes']['vthconn'])

    def get_outputs(self):
        return self.compartments['soma']

    def get_inputs(self):
        return self.compartments['input']

    def get_synproto(self):
        return self.prototypes['s_prototypes']['single']

    
"""
Tracker node which fires at a rate proportional to the number of reward spikes over spikes received at the 
reward and punishment nodes.
"""
class TrackerNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)
        self.noisy = int(kwargs.get("noisy", False))
        self.vth = kwargs.get("vth", 255)
        self.dynrange = kwargs.get("dynrange", 1)
        
        self._create_prototypes()
        self.tracker_prototypes = prototypes.create_prototypes(noisy=self.noisy, vth=self.vth*self.dynrange, synscale=self.dynrange)
        self._create_compartments()
        self._create_connections()

    def _create_compartments(self):

        self.neurons['qneurons'] = self.network.createNeuronGroup(size=self.numNodes,
                                                            prototype=self.tracker_prototypes['n_prototypes']['qProto'])

        self.compartments['soma'] = self.neurons['qneurons'].soma
        self.compartments['integrator'] = self.neurons['qneurons'].dendrites[0]
        self.compartments['memory'] = self.neurons['qneurons'].dendrites[0].dendrites[0]

        self.compartments['inverter'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                            prototype=self.prototypes['c_prototypes']['invProto'])
                                                            
        self.compartments['excite_ands'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                                    prototype=self.prototypes['c_prototypes']['andProto'])
        self.compartments['inhibit_ands'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                                    prototype=self.prototypes['c_prototypes']['andProto'])

        self.compartments['excite'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                prototype=self.prototypes['c_prototypes']['bufferProto'])

        self.compartments['inhibit'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                prototype=self.prototypes['c_prototypes']['bufferProto'])

    def _create_connections(self):
        #connect output to the inverter input
        self.connections['soma_inv'] = connect_one_to_one(self.compartments['soma'], self.compartments['inverter'], self.prototypes['s_prototypes']['invconn'])
        #connect the soma output to soft reset the integrator
        self.connections['softreset'] = [connect_one_to_one(self.compartments['soma'], self.compartments['integrator'], self.prototypes['s_prototypes']['vthconn']) for i in range(self.dynrange)]
        #connect the excite/inhibit buffers to their respective ands
        self.connections['excite_exciteands'] = connect_one_to_one(self.compartments['excite'], self.compartments['excite_ands'], self.prototypes['s_prototypes']['halfconn'])
        self.connections['inhibit_inhands'] = connect_one_to_one(self.compartments['inhibit'], self.compartments['inhibit_ands'], self.prototypes['s_prototypes']['halfconn'])
        #connect the output gating to the excite/inhibit ands
        self.connections['inv_exciteands'] = connect_one_to_one(self.compartments['inverter'], self.compartments['excite_ands'], self.prototypes['s_prototypes']['halfconn'])
        self.connections['soma_inhands'] = connect_one_to_one(self.compartments['soma'], self.compartments['inhibit_ands'], self.prototypes['s_prototypes']['halfconn'])
        #connect the ands' output to the memory compartment
        self.connections['exciteand_memory'] = connect_one_to_one(self.compartments['excite_ands'], self.compartments['memory'], self.prototypes['s_prototypes']['econn'])
        self.connections['inhand_memory'] = connect_one_to_one(self.compartments['inhibit_ands'], self.compartments['memory'], self.prototypes['s_prototypes']['iconn'])

    def get_outputs(self):
        return self.compartments['soma']
    
    def get_inverted_outputs(self):
        return self.compartments['inverter']

    def get_inputs(self):
        return self.compartments['excite'], self.compartments['inhibit']

    def get_synproto(self):
        return self.prototypes['s_prototypes']['single']
        
"""
Creates a compartment which only fires when all inputs are active in a single cycle.
"""
class AndNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)

        self.numInputs = kwargs.get("numInputs", 2)
        self._create_prototypes()
        and_weight = int(self.prototypes['vth'] / self.numInputs) + 1
        self.prototypes['s_prototypes']['andconn'] = nx.ConnectionPrototype(weight=and_weight)
        self._create_compartments()
    
    def _create_compartments(self):
        self.compartments['and'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                                prototype=self.prototypes['c_prototypes']['andProto'])

    def get_synproto(self):
        return self.prototypes['s_prototypes']['andconn']

    def get_inputs(self):
        return self.compartments['and']

    def get_outputs(self):
        return self.compartments['and']

"""
Creates a compartment which fires when any input is active in a single cycle. Also can be used as buffer.
"""
class OrNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)

        self._create_prototypes()
        self._create_compartments()
    
    def _create_compartments(self):
        self.compartments['or'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                                prototype=self.prototypes['c_prototypes']['bufferProto'])

    def get_synproto(self):
        return self.prototypes['s_prototypes']['single']

    def get_inputs(self):
        return self.compartments['or']

    def get_outputs(self):
        return self.compartments['or']

"""
Creates a node which is inhibited on an input spike and otherwise fires (effectively inverts the input signal with a 1 cycle delay).
"""
class InvNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)

        self._create_prototypes()
        self._create_compartments()
    
    def _create_compartments(self):
        self.compartments['inv'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                                prototype=self.prototypes['c_prototypes']['invProto'])

    def get_synproto(self):
        return self.prototypes['s_prototypes']['invconn']

    def get_inputs(self):
        return self.compartments['inv']

    def get_outputs(self):
        return self.compartments['inv']

"""
Create a ring of compartments which will repeatedly fire in sequential order.
"""
class RingOscNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        assert np.prod(shape) > 1, "Must have at least 2 nodes in ring oscillator"

        super().__init__(network, shape, logicalCore)

        self._create_prototypes()
        self._create_special_prototypes()
        self._create_compartments()
        self._create_connections()

    def _create_special_prototypes(self):
        selfBiasMant=5
        startupCycles=2
        starterThMant = selfBiasMant*startupCycles-1

        self.prototypes['c_prototypes']['starterProto'] = nx.CompartmentPrototype(vThMant=starterThMant,
                                    biasMant=selfBiasMant,
                                    biasExp=6,
                                    functionalState=2,
                                    compartmentVoltageDecay=0,
                                    compartmentCurrentDecay=4095,
                                    logicalCoreId=self.logicalCore,
                                    **prototypes.noise_kwargs)
            

        self.prototypes['s_prototypes']['starterInhConn'] = nx.ConnectionPrototype(weight=-selfBiasMant)
    
    def _create_compartments(self):
        #create the starter compartment
        self.compartments['starter'] = self.network.createCompartmentGroup(size=1,
                                    prototype=self.prototypes['c_prototypes']['starterProto'])

        #create the compartments which will respond to its initial self-generated signal
        self.compartments['responder'] = self.network.createCompartmentGroup(size=self.numNodes,
                                    prototype=self.prototypes['c_prototypes']['bufferProto'])

        self.compartments['inhibitor'] = self.network.createCompartmentGroup(size=1,
                                    prototype=self.prototypes['c_prototypes']['bufferProto'])

    def _create_connections(self):
        #create the connection from the starter to the responder group's first element
        startMask = np.zeros((self.numNodes,1))
        startMask[0,0] = 1

        self.connections['start_to_resp'] = self.compartments['starter'].connect(self.compartments['responder'],
                                    connectionMask=startMask,
                                    prototype=self.prototypes['s_prototypes']['spkconn'])

        #create the connection from one responder to the next, starting with the first to the second and looping the last to first
        responderMask = np.diag(np.ones(self.numNodes-1),-1)
        responderMask[0,-1] = 1
        self.connections['responder_series'] = self.compartments['responder'].connect(self.compartments['responder'],
                                    connectionMask=responderMask,
                                    prototype=self.prototypes['s_prototypes']['spkconn'])


        #wire starter & all responders to the summator
        self.connections['starter_sum'] = connect_full(self.compartments['starter'], 
                                    self.compartments['inhibitor'],
                                    self.prototypes['s_prototypes']['spkconn'])

        self.connections['responder_sum'] = connect_full(self.compartments['responder'], 
                                    self.compartments['inhibitor'],
                                    self.prototypes['s_prototypes']['spkconn'])

        #the use the summator's output to inhibit the starter's self-excitation
        self.connections['sum_starter_inhibit'] = connect_full(self.compartments['inhibitor'], 
                                    self.compartments['starter'],
                                    self.prototypes['s_prototypes']['starterInhConn'])

    def get_outputs(self):
        return self.compartments['responder']

    def get_inputs(self):
        return None

    def get_synproto(self):
        return None

"""
Create a dummy counter node which only accumulates voltage as spikes are received. 
This is usually used to count spikes and is read and reset by a SNIP.

Effectively should be the same as the Lakemont registers for the same purpose, but 
can sometimes be more convenient to use and read-out.
"""
class CounterNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)

        self._create_prototypes()
        self._create_compartments()

    def _create_compartments(self):
        self.compartments['counters'] = self.network.createCompartmentGroup(size=self.numNodes,
                                                            prototype=self.prototypes['c_prototypes']['counterProto'])

    def get_outputs(self):
        return self.compartments['counters']

    def get_inputs(self):
        return self.compartments['counters']

    def get_synproto(self):
        return self.prototypes['s_prototypes']['single']

"""
Takes the average of incoming nodes by splitting their charge proportionally to the number of inputs and firing
at the average combined rate. Much simpler implementation then the sampling Avg node but has integer math based biases.
"""
class QAverageNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)
        self.n_replicates = kwargs.get("n_replicates", 1)
        self.blocks = {}
        self.vth = 2 * self.n_replicates - 1

        self._create_prototypes(vth=self.vth)
        self._create_blocks()

    def _create_blocks(self):
        self.blocks['averages'] = SoftResetNode(self.network, self.shape, self.logicalCore, vth=self.vth)

    def get_inputs(self):
        return self.blocks['averages'].get_inputs()

    def get_outputs(self):
        return self.blocks['averages'].get_outputs()

    def get_synproto(self):
        return self.prototypes['s_prototypes']['single']
    

"""
A node which follows the spiking rate of the inputs. Similar to Tracker but the inputs are used to update firing rates *every* step
instead of being gated by the presence/absence of a reward/punishment signal. 
"""
class FollowerNode(ProcessNode):
    def __init__(self, network, shape, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)
        self.blocks = {}

        self._create_prototypes()
        self._create_blocks()
        self.prototypes['s_prototypes']['delay_conn'] = nx.ConnectionPrototype(weight=self.blocks['tracker'].get_synproto().weight, delay=1)
        self._connect_blocks()

    def _create_blocks(self):
        self.blocks['buffer'] = OrNode(self.network, self.shape, self.logicalCore)
        self.blocks['inverter'] = InvNode(self.network, self.shape, self.logicalCore)
        self.blocks['tracker'] = TrackerNode(self.network, self.shape, self.logicalCore)

    def _connect_blocks(self):
        self.connections['buffer_inv'] = connect_one_to_one(self.blocks['buffer'].get_outputs(),
                                                            self.blocks['inverter'].get_inputs(),
                                                            self.blocks['inverter'].get_synproto())

        (exc_in, inh_in) = self.blocks['tracker'].get_inputs()
        self.connections['buffer_exc'] = connect_one_to_one(self.blocks['buffer'].get_outputs(),
                                                            exc_in,
                                                            self.prototypes['s_prototypes']['delay_conn'])

        self.connections['inverter_exc'] = connect_one_to_one(self.blocks['inverter'].get_outputs(),
                                                            inh_in,
                                                            self.blocks['tracker'].get_synproto())

    def get_inputs(self):
        return self.blocks['buffer'].get_inputs()

    def get_outputs(self):
        return self.blocks['tracker'].get_outputs()

    def get_synproto(self):
        return self.blocks['buffer'].get_synproto()
        
"""
A node which can create the average firing rate of several inputs at its output.
Currently this is done by sampling over each of the inputs.
To do it by splitting charge into a soft-reset neuron, use QAverageNode, but this can have errors for larger averages.
"""
class AveragePoolNode(ProcessNode):
    def __init__(self, network, shape, averageAxis, logicalCore=-1, **kwargs):
        super().__init__(network, shape, logicalCore)
        self.averageAxis = averageAxis
        self.variableAxes = np.delete(np.arange(len(shape)), averageAxis)
        self.output_shape = tuple(np.delete(np.array(shape), averageAxis))

        self.n_vars = reduce(np.multiply, [self.output_shape[i] for i in self.variableAxes])
        self.n_samples  = self.shape[self.averageAxis]

        self.blocks = {}
        self._create_prototypes()
        self._create_blocks()
        self._create_connections()

    def _create_blocks(self):
        
        self.blocks['filter'] = AndNode(self.network, self.shape, self.logicalCore)
        self.blocks['sampler'] = RingOscNode(self.network, [self.n_samples,1], self.logicalCore)
        self.blocks['summator'] = OrNode(self.network, self.output_shape, self.logicalCore)

    def _create_connections(self):
        self.connections['sampler_filter'] = dense_along_axis(self.blocks['sampler'].get_outputs(),
                                                            self.blocks['sampler'].shape,
                                                            0,
                                                            self.blocks['filter'].get_inputs(),
                                                            self.shape,
                                                            self.averageAxis,
                                                            prototype=self.blocks['filter'].get_synproto())

        self.connections['filter_summator'] = project_along_axis(self.blocks['filter'].get_outputs(),
                                                            self.shape,
                                                            self.averageAxis,
                                                            self.blocks['summator'].get_inputs(),
                                                            self.blocks['summator'].shape,
                                                            prototype=self.blocks['summator'].get_synproto())
        
    def get_inputs(self):
        return self.blocks['filter'].get_inputs()

    def get_outputs(self):
        return self.blocks['summator'].get_outputs()

    def get_synproto(self):
        return self.blocks['filter'].get_synproto()


# -- CONNECTIVITY -- #

def get_dim(object):
    if hasattr(object, "numNodes"):
        return object.numNodes
    elif hasattr(object, "numPorts"):
        return object.numPorts

"""
Connect each neuron in the source to a single corresponding neuron in the destination.
(1 -> 1), (2 -> 2), ... , (n -> n)
"""
def connect_one_to_one(source, target, prototype):
    assert get_dim(source) == get_dim(target), "Must have equal number of nodes to connect one-to-one."
    mask = np.identity(target.numNodes)

    return source.connect(target,
                        prototype=prototype,
                        connectionMask=mask)

"""
Connect two-dimensional nodes along a single axis. **DEPRECATED**
"""
def connect_along_axis(source, source_shape, source_axis, target, target_shape, target_axis, prototype):
    print("Deprecated to dense_along_axis")
    return dense_along_axis(source, source_shape, source_axis, target, target_shape, target_axis, prototype)


"""
Densely connect each slice of the source and target tensors along a single, matching axis.
Usually done to map connections to an AND block with multiple inputs.
"""
def dense_along_axis(source, source_shape, source_axis, target, target_shape, target_axis, prototype):
    assert source_shape[source_axis] == target_shape[target_axis], "Shapes must match along axis to be expanded:" + str(source_shape) + str(target_shape)

    len_dim = source_shape[source_axis]
    source_dims = len(source_shape)
    source_len = np.prod(source_shape)
    target_dims = len(target_shape)
    target_len = np.prod(target_shape)
    
    connections = []
    
    for i in range(len_dim):
        source_tensor = np.zeros(source_shape)
        target_tensor = np.zeros(target_shape)
    
        #select all the elements we want to connect (everything along the current slice)
        source_fill = [slice(None, None, None) for j in range(source_dims)]
        source_fill[source_axis] = i
        source_tensor[tuple(source_fill)] = 1
        
        #do the same for the target
        target_fill = [slice(None, None, None) for j in range(target_dims)]
        target_fill[target_axis] = i
        target_tensor[tuple(target_fill)] = 1
        
        #construct the tensor product of these connections and reshape to adj matrix
        product = np.tensordot(source_tensor, target_tensor, axes=0)
        product = product.reshape(source_len, target_len)
        connections.append(product)
        
    mask = reduce(np.add, connections).transpose()
    return source.connect(target,
                        prototype=prototype,
                        connectionMask=mask)
        

"""
Generate an adjacency matrix which generates the adjacencies required to connect all elements of tensor's projection
to all of its higher-order elements along the projected axis.
"""
def get_adjacency(shape, axis):
    dims = len(shape)
    new_dims = dims - 1
    new_shape = np.delete(np.array(shape), axis)
    new_basis = np.flip(np.cumprod(np.flip(new_shape, axis=0)), axis=0)
    new_basis = np.concatenate((new_basis[1:],[1]))

    n = np.prod(shape)
    n_proj = np.prod(new_shape)
    projections = np.zeros(n, dtype=np.int)

    #store how the projected axes map to old axes
    proj_map = {}
    for i in range(new_dims):
        proj_map[i] = i + 1 if i >= axis else i

    #for each element in the original tensor
    coords = np.unravel_index(np.arange(0, n), shape)
    #find its embedding in the projected tensor
    for i in range(new_dims):
        projections += (coords[proj_map[i]] * new_basis[i])

    #use the original index and projected index to set adjacency elements
    adjacency = np.zeros((n, n_proj), dtype=np.int)
    for i in range(n):
        adjacency[i,projections[i]] = 1

    return adjacency

"""
Project all source's compartments along an axis to a single compartment on the target.
Used to reduce averages, etc. 
"""
def project_along_axis(source, source_shape, source_axis, target, target_shape, prototype):
    dims = len(source_shape)
    new_dims = dims - 1
    new_shape = np.delete(np.array(source_shape), source_axis)
    
    assert target_shape == tuple(new_shape), "Target shape does not match shape of source projected along requested axis."
    
    mask = get_adjacency(source_shape, source_axis).transpose()
    return source.connect(target,
                        prototype=prototype,
                        connectionMask=mask)
        
"""
Project a source's compartments to all elements along a target axis in the target.
Used to create replicate data from one source to multiple targets.
"""
def expand_along_axis(source, source_shape, target, target_shape, target_axis, prototype):
    #ensure the target shape is compatible
    shape_check = tuple(np.delete(np.array(target_shape), target_axis))
    assert source_shape == shape_check, "Target shape must be the same as source shape except for the addition of a single axis the source is expanding over."

    mask = get_adjacency(target_shape, target_axis)
    return source.connect(target,
                        prototype=prototype,
                        connectionMask=mask)


"""
Create all-to-all connections between two nodes.
(1->1), (1->2), ... , (1->n), 
(2->1), (2->2), ... , (2->n)
                ... , (n->n)
"""
def connect_full(source, target, prototype):
    return source.connect(target, prototype=prototype)
