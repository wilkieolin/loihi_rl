import nxsdk.api.n2a as nx
noise_kwargs = {'randomizeCurrent' : 1,
                    'noiseMantAtCompartment': 0,
                    'noiseExpAtCompartment' : 6}

def create_prototypes(vth=255, logicalCoreId=-1, noisy=0, synscale=1):
    prototypes = {}
    prototypes['vth'] = vth
    
    #setup compartment prototypes
    c_prototypes = {}
    n_prototypes = {}
    s_prototypes = {}

    #Q Neuron
    c_prototypes['somaProto'] = nx.CompartmentPrototype(vThMant=vth,
                                  compartmentCurrentDecay=4095,
                                  compartmentVoltageDecay=0,
                                  logicalCoreId=logicalCoreId,
                                  enableNoise=0,
                                  **noise_kwargs
                                  )

    c_prototypes['spkProto'] = nx.CompartmentPrototype(vThMant=vth,
                                     biasMant=int(vth/2),
                                     biasExp=6,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=0,
                                     thresholdBehavior=2,
                                     logicalCoreId=logicalCoreId,
                                     enableNoise=noisy,
                                     **noise_kwargs
                                     )

    c_prototypes['ememProto'] = nx.CompartmentPrototype(vThMant=vth,
                                     #vMaxExp=15,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=0,
                                     thresholdBehavior=3,
                                     logicalCoreId=logicalCoreId,
                                     enableNoise=0,
                                     **noise_kwargs
                                     )

    c_prototypes['somaProto'].addDendrite([c_prototypes['spkProto']],
                                        nx.COMPARTMENT_JOIN_OPERATION.OR)

    c_prototypes['spkProto'].addDendrite([c_prototypes['ememProto']],
                                        nx.COMPARTMENT_JOIN_OPERATION.ADD)

    n_prototypes['qProto'] = nx.NeuronPrototype(c_prototypes['somaProto'])


    #Soft Reset Neuron
    c_prototypes['srSomaProto'] = nx.CompartmentPrototype(vThMant=vth,
                                  compartmentCurrentDecay=4095,
                                  compartmentVoltageDecay=0,
                                  logicalCoreId=logicalCoreId,
                                  enableNoise=0,
                                  **noise_kwargs
                                  )

    c_prototypes['srSpkProto'] = nx.CompartmentPrototype(vThMant=vth,
                                     biasMant=0,
                                     biasExp=6,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=0,
                                     thresholdBehavior=2,
                                     logicalCoreId=logicalCoreId,
                                     enableNoise=noisy,
                                     **noise_kwargs
                                     )

    c_prototypes['intProto'] = nx.CompartmentPrototype(vThMant=vth,
                                     #vMaxExp=15,
                                     compartmentCurrentDecay=4095,
                                     compartmentVoltageDecay=4095,
                                     thresholdBehavior=0,
                                     logicalCoreId=logicalCoreId,
                                     enableNoise=0,
                                     **noise_kwargs
                                     )

    c_prototypes['srSomaProto'].addDendrite([c_prototypes['srSpkProto']],
                                        nx.COMPARTMENT_JOIN_OPERATION.OR)

    c_prototypes['srSpkProto'].addDendrite([c_prototypes['intProto']],
                                        nx.COMPARTMENT_JOIN_OPERATION.ADD)

    n_prototypes['srProto'] = nx.NeuronPrototype(c_prototypes['srSomaProto'])

    #FF Neuron
    c_prototypes['ffSomaProto'] = nx.CompartmentPrototype(vThMant=1,
                                  compartmentCurrentDecay=4095,
                                  compartmentVoltageDecay=0,
                                  logicalCoreId=logicalCoreId,
                                  enableNoise=0,
                                  **noise_kwargs
                                  )

    c_prototypes['ffSomaProto'].addDendrite([c_prototypes['ememProto']],
                                        nx.COMPARTMENT_JOIN_OPERATION.ADD)

    n_prototypes['ffProto'] = nx.NeuronPrototype(c_prototypes['ffSomaProto'])

    #Inverter compartment
    c_prototypes['invProto'] = nx.CompartmentPrototype(vThMant=1,
                                    biasMant=2,
                                    biasExp=6,
                                    compartmentVoltageDecay=0,
                                    functionalState=2,
                                    logicalCoreId=logicalCoreId,
                                    enableNoise=0,
                                   **noise_kwargs
                                   )

    #buffer / OR

    c_prototypes['bufferProto'] = nx.CompartmentPrototype(vThMant=1,
                                    compartmentVoltageDecay=4095,
                                    compartmentCurrentDecay=4095,
                                    logicalCoreId=logicalCoreId,
                                    enableNoise=0,
                                    **noise_kwargs
                                    )
                                
    #AND
    c_prototypes['andProto'] = nx.CompartmentPrototype(vThMant=vth,
                                compartmentCurrentDecay=4095,
                                compartmentVoltageDecay=4095,
                                logicalCoreId=logicalCoreId,
                                enableNoise=0,
                                **noise_kwargs
                                )

    #WTA
    # c_prototypes['noisyWTAProto'] = nx.CompartmentPrototype(biasMant=10,
    #                                     biasExp = 6,
    #                                     vThMant = 100,
    #                                     logicalCoreId=0,
    #                                     compartmentVoltageDecay=0,
    #                                     compartmentCurrentDecay=128,
    #                                     enableNoise=1,
    #                                     noiseMantAtCompartment=0,
    #                                     noiseExpAtCompartment=11,
    #                                     functionalState = nx.COMPARTMENT_FUNCTIONAL_STATE.IDLE)

    #Counter
    v_th_max = 2**17-1
    c_prototypes['counterProto'] = nx.CompartmentPrototype(vThMant=v_th_max,
                                compartmentCurrentDecay=4095,
                                compartmentVoltageDecay=0,
                                logicalCoreId=logicalCoreId,
                                enableNoise=0,
                                **noise_kwargs
                                )

    #Connections
    vth = vth/synscale

    s_prototypes['econn'] = nx.ConnectionPrototype(weight=2)
    s_prototypes['iconn'] = nx.ConnectionPrototype(weight=-2)
    s_prototypes['invconn'] = nx.ConnectionPrototype(weight=-1)
    s_prototypes['vthconn'] = nx.ConnectionPrototype(weight=-vth)
    s_prototypes['spkconn'] = nx.ConnectionPrototype(weight=vth)
    s_prototypes['halfconn'] = nx.ConnectionPrototype(weight = int(vth/2)+1)
    s_prototypes['thirdconn'] = nx.ConnectionPrototype(weight = int(vth/3)+1)
    s_prototypes['single'] = nx.ConnectionPrototype(weight = 2)

    prototypes['c_prototypes'] = c_prototypes
    prototypes['n_prototypes'] = n_prototypes
    prototypes['s_prototypes'] = s_prototypes

    return prototypes
