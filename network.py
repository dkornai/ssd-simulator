'''NETWORK AND NODE CLASSES'''

import networkx as nx
import numpy as np

class Node():
    '''base Node class (currently empty)'''
    def __init__(self, id=None):
        self.id = id

class ConstantBirthNode(Node):
    '''Node with births ocurring at constant (possibly zero) rate'''
    def __init__(self, deathrate:float, birthrate:float, id=None):
        super().__init__(id)
        self.deathrate = deathrate
        self.birthrate = birthrate

    def __str__(self) -> str:
        return f'ConstantBirthNode(id:{self.id})'
    
    def __repr__(self) -> str:
        return f'ConstantBirthNode(id:{self.id})'

class DynamicBirthNode(Node):
    '''Node with dynamic birth rates targeting a given population size'''
    def __init__(self, deathrate:float, birthrate:float, targetpop:int, controlstrength:float, delta:float, id=None):
        super().__init__(id)
        self.deathrate = deathrate
        self.birthrate = birthrate
        self.targetpop = targetpop
        self.controlstrength = controlstrength
        self.delta = delta

    def __str__(self) -> str:
        return f'DynamicBirthNode(id:{self.id})'
    
    def __repr__(self) -> str:
        return f'DynamicBirthNode(id:{self.id})'
        
class Network(nx.DiGraph):
    '''class used to hold the network of Nodes'''
    def __init__(self, **kwargs):
        
        
        super().__init__(**kwargs)
        self.replicative_advantage = None
        self.slower_advantage = None
        self.extra_attributes = kwargs  # Store extra attributes in a dictionary
        
        # Keep track of nodes in the system
        self.n_nodes = 0
        self.node_ids = []

    def add_node(self, node, **attr):
        '''add a Node type object to the Network'''
        
        if not isinstance(node, Node):
            raise TypeError("The Network class can only add nodes of the Node type, not arbitray objects.")
        
        # Verify that the node does not have an id
        if node.id is not None:
            raise ValueError("The node id property should only be accessed by the program, not set by the user")
        
        # Assign an ID to the node
        node.id = self.n_nodes
        
        # Add the id, and increment the number of nodes
        self.node_ids.append(node.id)
        self.n_nodes += 1 

        super().add_node(node, **attr)

    def get_node_by_id(self, node_id) -> Node:
        '''return the node with the specified id'''

        for node in self.nodes:
            if hasattr(node, 'id') and node.id == node_id:
                return node
        raise KeyError("The node with the requested id is not found in the network.")

    def add_edge(self):
        '''this overrides the .add_edge method, preventing possible problems that arise from misuse'''
        
        raise NotImplementedError("Do not use .add_edge() with the Network class, use .add_transport() instead.")

    def add_transport(self, source_node, dest_node, **attr):
        '''add a transport reaction between two nodes'''
        
        if source_node == dest_node:
            raise ValueError("Source and destination cannot be identical")

        # Behaviour if nodes are specified as objects
        if isinstance(source_node, Node) and isinstance(dest_node, Node):
            if source_node not in list(self.nodes):
                raise ValueError("Source node not in the network!")
            if dest_node not in list(self.nodes):
                raise ValueError("Destination node not in the network!")
        
        # Behaviour if nodes are specifed via their ids
        elif isinstance(source_node, int) and isinstance(dest_node, int):
            if source_node not in self.node_ids:
                raise ValueError("Source node not in the network!")
            if dest_node not in self.node_ids:
                raise ValueError("Destination node not in the network!")

            # Find the nodes matching the ids
            source_node = self.get_node_by_id(source_node)
            dest_node   = self.get_node_by_id(dest_node)

        # Other ways of determining edges are not supported
        else:
            raise NotImplementedError("Both nodes must be specified via their ids or as specific Node objects")

        if 'rate' not in attr:
            raise ValueError("Edge must have a 'rate' attribute")
        
        super().add_edge(source_node, dest_node, **attr)


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

    def get_empty_statevector(self) -> np.ndarray:
        '''get the empty (all 0s) statevector'''
        
        return np.zeros(self.n_nodes*2, dtype=np.int32)
    
    def get_statevector_heteroplasmy(self, heteroplasmy:float, delta:float) -> np.ndarray:
        ''''''
        if not 0<heteroplasmy<1:
            raise ValueError("heteroplasmy must be between 0 and 1")
        if not 0<delta<1:
            raise ValueError("delta must be between 0 and 1")
        