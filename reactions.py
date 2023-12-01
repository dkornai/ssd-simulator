'''CLASS HOLDING REACTIONS THAT OCCUR IN THE NETWORK'''

from network import Node, ConstantBirthNode, DynamicBirthNode, Network
import numpy as np

class Reaction():
    '''base reaction class, only attribute is the node at which the reaction is ocurring'''
    def __init__(self, node:Node, etype:int):
        self.node = node
        self.etype = etype # Entity type index
        self.varnamestr = f'n{node.id}_t{etype}'
        self.statevecindex = node.id*2 + etype # Which element of the statevector the reaction is applied to

    def __str__(self) -> str:
        return f'variable:{self.varnamestr}, state_i:{self.statevecindex}'

class DeathReaction(Reaction):
    '''death reactions occur at a given node with a given static rate'''
    def __init__(self, node:Node, etype:int, deathrate:float):
        super().__init__(node, etype)
        self.deathrate = deathrate

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'DeathReaction({base_str}, rate:{self.deathrate})'
    
    def statevecupdate(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = -1 #
        return statevec

class ConstantBirthReaction(Reaction):
    '''static birth reactions occur at a given node with a given static rate'''
    def __init__(self, node:Node, etype:int, birthrate:float):
        super().__init__(node, etype)
        self.birthrate = birthrate

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'ConstantBirthReaction({base_str}, rate:{self.birthrate})'
    
    def statevecupdate(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = 1
        
        return statevec
        
class DynamicBirthReaction(Reaction):
    '''static birth reactions occur at a given node with a given static rate'''
    def __init__(self, node:Node, etype:int, birthrate:float, targetpop:int, controlstrength:float, delta:float):
        super().__init__(node, etype)
        self.birthrate = birthrate
        self.targetpop = targetpop
        self.controlstrenth = controlstrength
        self.delta = delta

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'DynamicBirthReaction({base_str}, rate:dynamic)'
    
    def statevecupdate(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = 1
        
        return statevec

class TransportReaction(Reaction):
    '''contstant rate transport ocurring at a given rate'''
    def __init__(self, source_node: Node, dest_node: Node, etype: int, transportrate:float):
        super().__init__(source_node, etype)
        self.dest_node = dest_node
        self.transportrate = transportrate

    def __str__(self) -> str:
        base_str = super().__str__()
        return f'TransportReaction({base_str}, sourcenode: {self.node.id}, destnode: {self.dest_node.id}, rate:{self.transportrate})'
    
    def statevecupdate(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = -1
        statevec[self.dest_node.id*2 + self.etype] = 1
        
        return statevec


class NetworkReactions():
    '''class representing the set of reactions in a specified network'''
    
    def __init__(self, network:Network):
        '''initilise by going through the nodes and edges of a given network to collect birth, death, and transport reactions'''
        
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