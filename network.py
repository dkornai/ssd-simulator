'''NETWORK AND NODE CLASSES'''

import networkx as nx
import numpy as np

class Node():
    '''base Node class (currently empty)'''
    def __init__(self):
        self.id = None

class ConstantBirthNode(Node):
    '''Node with births ocurring at constant (possibly zero) rate'''
    def __init__(self, deathrate:float, birthrate:float):
        super().__init__()
        self.deathrate = deathrate
        self.birthrate = birthrate

    def __str__(self) -> str:
        return f'ConstantBirthNode(id:{self.id})'

class DynamicBirthNode(Node):
    '''Node with dynamic birth rates targeting a given population size'''
    def __init__(self, deathrate:float, birthrate:float, targetpop:int, controlstrength:float, delta:float):
        super().__init__()
        self.deathrate = deathrate
        self.birthrate = birthrate
        self.targetpop = targetpop
        self.controlstrength = controlstrength
        self.delta = delta

    def __str__(self) -> str:
        return f'DynamicBirthNode(id:{self.id})'
        
class Network(nx.DiGraph):
    '''class used to hold the network of Nodes'''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.replicative_advantage = None
        self.slower_advantage = None
        self.extra_attributes = kwargs  # Store extra attributes in a dictionary

    def add_node(self, node, **attr):
        '''add a Node type object to the Network'''
        if type(node) not in [ConstantBirthNode, DynamicBirthNode]:
            raise TypeError("The Network class can only add nodes of the Node type, not arbitray objects.")
        
        return super().add_node(node, **attr)

    def add_edge(self):
        '''this overrides the .add_edge method, preventing possible problems that arise from misuse'''
        
        raise NotImplementedError("Do not use .add_edge() with the Network class, use .add_transport() instead.")

    def add_transport(self, source_node, dest_node, **attr):
        '''add a transport reaction between two nodes'''
        
        if source_node not in list(self.nodes):
            raise ValueError("Source node not in the network!")
        
        if dest_node not in list(self.nodes):
            raise ValueError("Destination node not in the network!")

        if 'rate' not in attr:
            raise ValueError("Edge must have a 'rate' attribute")
        
        super().add_edge(source_node, dest_node, **attr)

    def allocate_ids(self):
        '''allocate unique ids to all the nodes in the network'''
        
        for i, node in enumerate(self.nodes):
            node.id = i

    def set_replicative_advantage(self, advantage:float):
        '''configure replicative advantage for entity type 1'''

        if self.slower_advantage != None:
            raise ValueError("System cannot have replicative advantage and slower advantage at the same time!")
        self.replicative_advantage = advantage
    
    def set_slower_advantage(self, advantage:float):
        '''configure an increase in the birthrate and deathrate of entity type 0, leading to an advantage for entity type 1'''

        if self.replicative_advantage != None:
            raise ValueError("System cannot have slower advantage and replicative advantage at the same time!")
        self.slower_advantage = advantage

    