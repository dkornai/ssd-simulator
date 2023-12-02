'''CLASS HOLDING REACTIONS THAT OCCUR IN THE NETWORK'''

from network import Node, ConstantBirthNode, DynamicBirthNode, Network
import numpy as np

class Reaction():
    '''base reaction class, only attribute is the node at which the reaction is ocurring'''
    def __init__(self, node:Node, etype:int, rate:float):
        self.node = node
        self.etype = etype # Entity type index
        self.rate = rate # Generic reaction rate
        self.varnamestr = f'n{node.id}_t{etype}'
        self.statevecindex = node.id*2 + etype # Which element of the statevector the reaction is applied to

        # These variables are set to 0 by default for all reactions, and are only changed for dynamic birth reactions
        self.birthrate = 0
        self.targetpop = 0
        self.delta = 0
        self.controlstrength = 0

    def __str__(self) -> str:
        return f'variable:{self.varnamestr}, state_i:{self.statevecindex}'

class DeathReaction(Reaction):
    '''death reactions occur at a given node with a given static rate'''
    def __init__(self, node:Node, etype:int, deathrate:float):
        super().__init__(node, etype, rate=deathrate)

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'DeathReaction({base_str}, rate:{self.rate})'
    
    def statevec_update(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = -1 #
        return statevec

class ConstantBirthReaction(Reaction):
    '''static birth reactions occur at a given node with a given static rate'''
    def __init__(self, node:Node, etype:int, birthrate:float):
        super().__init__(node, etype, rate=birthrate)

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'ConstantBirthReaction({base_str}, rate:{self.rate})'
    
    def statevec_update(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = 1
        
        return statevec
        
class DynamicBirthReaction(Reaction):
    '''static birth reactions occur at a given node with a given static rate'''
    def __init__(self, node:Node, etype:int, birthrate:float, targetpop:int, controlstrength:float, delta:float):
        super().__init__(node, etype, rate=-1)
        self.birthrate = birthrate
        self.targetpop = targetpop
        self.controlstrenth = controlstrength
        self.delta = delta

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'DynamicBirthReaction({base_str}, rate:dynamic)'
    
    def statevec_update(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = 1
        
        return statevec

class TransportReaction(Reaction):
    '''contstant rate transport ocurring at a given rate'''
    def __init__(self, source_node: Node, dest_node: Node, etype: int, transportrate:float):
        super().__init__(source_node, etype, rate=transportrate)
        self.dest_node = dest_node

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'TransportReaction({base_str}, sourceid: {self.node.id}, destid: {self.dest_node.id}, rate:{self.rate})'
    
    def statevec_update(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = -1
        statevec[self.dest_node.id*2 + self.etype] = 1
        
        return statevec

class NetworkReactions():
    '''class representing the set of reactions in a specified network'''
    
    def __init__(self, network:Network):
        '''initilise by going through the nodes and edges of a given network to collect birth, death, and transport reactions'''
        
        # Initilize the empty state vector, which will hold the number of particles in each node
        self.statevec = np.zeros(network.number_of_nodes()*2, dtype=np.int32) # *2 due to allocation for two types of entities at each node

        # Start collecting the reactions
        self.reactions = []

        # Add birth reactions if there is a nonzero birth rate
        for node in network.nodes:
            if node.birthrate > 0:
                for etype in range(2): # Iterate through both entity types, adding reactions for each 
                    if type(node) is ConstantBirthNode:
                        
                        self.reactions.append(ConstantBirthReaction(node, etype, node.birthrate))
                        
                    elif type(node) is DynamicBirthNode:
                        
                        self.reactions.append(DynamicBirthReaction(node, etype, node.birthrate, node.targetpop, node.controlstrength, node.delta))

        # Add death reactions if there is a nonzero death rate
        for node in network.nodes:
            if node.deathrate > 0:
                for etype in range(2): # Iterate through both entity types, adding reactions for each 
                        
                        self.reactions.append(DeathReaction(node, etype, node.deathrate))

        # Add transport reactions if there is a nonzero rate
        for source_node, dest_node, data in network.edges(data=True):
            transportrate = data['rate']
            if transportrate > 0:
                for etype in range(2): # Iterate through both entity types, adding reactions for each 
                        
                        self.reactions.append(TransportReaction(source_node, dest_node, etype, transportrate))

        self.n_reactions = len(self.reactions)

class CmeParameters():
    '''arrays of parameters required to simulate the chemical master equation (cme) corresponding to the reactions'''
    def __init__(self, networkreactions):
        '''generate the various arrays from the set of reactions in the network'''
        
        # The reaction matrix holds the update to the state vector corresponding to each reaction
        self.reaction_matrix = np.zeros((networkreactions.n_reactions, networkreactions.statevec.size), dtype=np.int32)
        for i, reaction in enumerate(networkreactions.reactions):
            self.reaction_matrix[i,:] = reaction.statevec_update(statevec=networkreactions.statevec)
        
        # The per-capita reaction rate vector holds the reaction rates for each reaction. dynamics rates are marked with -1
        self.percap_reaction_rates = np.zeros(networkreactions.n_reactions, dtype=np.float64)
        for i, reaction in enumerate(networkreactions.reactions):
            self.percap_reaction_rates[i] = reaction.rate

        # The reaction statevector indeces show which element of the statevector holds the particles undergoing each reaction
        self.reaction_statevec_index = np.zeros(networkreactions.n_reactions, dtype=np.int32)
        for i, reaction in enumerate(networkreactions.reactions):
            self.reaction_statevec_index[i] = reaction.statevecindex

        # Collect the indeces of nodes where dynamic birthrates occur
        self.dynamic_birth_statevec_index = [] # Locations in the staetvector that have dynamic births
        for i, reaction in enumerate(networkreactions.reactions):
            if type(reaction) == DynamicBirthReaction and reaction.etype == 0:
                self.dynamic_birth_statevec_index.append(i)
        self.dynamic_birth_statevec_index = np.array(self.dynamic_birth_statevec_index, dtype=np.int32)

        # These are the parameters that are used to update dynamic birthrates (will have 0 values for all other reaction types)
        self.birthrates          = np.zeros(networkreactions.n_reactions, dtype=np.float64)     
        self.targetpops          = np.zeros(networkreactions.n_reactions, dtype=np.float64)    
        self.controlstrengths    = np.zeros(networkreactions.n_reactions, dtype=np.float64)    
        self.deltas              = np.zeros(networkreactions.n_reactions, dtype=np.float64) 

        for i, reaction in enumerate(networkreactions.reactions):
            self.birthrates[i]       = reaction.birthrate
            self.targetpops[i]       = reaction.targetpop
            self.controlstrengths[i] = reaction.targetpop
            self.deltas[i]           = reaction.delta

    def __str__(self) -> str:
        txt = ''
        txt += f'Reaction Matrix:\n{str(self.reaction_matrix)}\n'
        txt += f'Reaction Rates:\n{str(self.percap_reaction_rates)}\n'
        txt += f'Statevec i:\n{str(self.reaction_statevec_index)}\n'
        txt += f'Dynamic Birth Parameters:\n{str(self.dynamic_birth_statevec_index)}\n'
        txt += f'{str(self.birthrates)}\n'
        txt += f'{str(self.targetpops)}\n'
        txt += f'{str(self.controlstrengths)}\n'
        txt += f'{str(self.deltas)}\n'
        return txt