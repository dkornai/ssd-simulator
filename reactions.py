'''CLASS HOLDING REACTIONS THAT OCCUR IN THE NETWORK'''

from typing import List
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
        return f'DeathReaction({self.varnamestr}, rate: {self.rate})'
    
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
        return f'BirthReaction({self.varnamestr}, rate: {self.rate})'
    
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
        self.controlstrength = controlstrength
        self.delta = delta
        self.etype_0_varnamestr = f'n{node.id}_t0'
        self.etype_1_varnamestr = f'n{node.id}_t1'

    def __str__(self) -> str:
        return f'BirthReaction({self.varnamestr}, rate: Dynamic)'
    
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
        self.dest_varnamestr = f'n{dest_node.id}_t{etype}' # Add the variable name for the destination node

    def __str__(self) -> str:
        return f'TransportReaction(from: {self.varnamestr}, to: {self.dest_varnamestr},  rate: {self.rate})'
    
    def statevec_update(self, statevec):
        '''update to the state vector corresponding to the reaction'''
        statevec = np.array(statevec)
        statevec[self.statevecindex] = -1
        statevec[self.dest_node.id*2 + self.etype] = 1
        
        return statevec

class Reactions():
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

        # Set up the number of reactions
        self.n_reactions = len(self.reactions)

        # Set up the variable names of the various species in the reactions
        self.reaction_variables = []
        for node in network.nodes:
            for etype in range(2):
                self.reaction_variables.append(f'n{node.id}_t{etype}') # Each entity type within each node gets its own variable.

    def get_birth_reaction(self, varname:str) -> List[Reaction]:
        '''get the birth reaction associated with a given variable name'''
        if varname not in self.reaction_variables:
            raise KeyError('Requested variable does not exist in the network')
        result = []
        for reaction in self.reactions:
            if type(reaction) in [ConstantBirthReaction, DynamicBirthReaction]:
                if reaction.varnamestr == varname:
                    result.append(reaction)
                    
        return result
            
    def get_death_reaction(self, varname:str) -> List[Reaction]:
        '''get the death reaction associated with a given variable name'''
        if varname not in self.reaction_variables:
            raise KeyError('Requested variable does not exist in the network')
        result = []
        for reaction in self.reactions:
            if type(reaction) in [DeathReaction]:
                if reaction.varnamestr == varname:
                    result.append(reaction)
                    
        return result
                
    def get_transport_reaction_out(self, varname:str) -> List[Reaction]:
        '''get the reactions where a given variable is the source'''
        if varname not in self.reaction_variables:
            raise KeyError('Requested variable does not exist in the network')
        result = []
        for reaction in self.reactions:
            if type(reaction) in [TransportReaction]:
                if reaction.varnamestr == varname:
                    result.append(reaction)
                    
        return result
                
    def get_transport_reaction_in(self, varname:str) -> List[Reaction]:
        '''get the reactions where a given variable is the destination'''
        if varname not in self.reaction_variables:
            raise KeyError('Requested variable does not exist in the network')
        result = []
        for reaction in self.reactions:
            if type(reaction) in [TransportReaction]:
                if reaction.dest_varnamestr == varname:
                    result.append(reaction)

        return result

class CmeParameters():
    '''arrays of parameters required to simulate the chemical master equation (cme) corresponding to the reactions'''
    def __init__(self, networkreactions:Reactions):
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
            self.controlstrengths[i] = reaction.controlstrength
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
    

def get_ode_term(
        varname:str,
        birth_reaction:List[ConstantBirthReaction|DynamicBirthReaction], 
        death_reaction:List[DeathReaction], 
        transport_out_reaction:List[TransportReaction], 
        transport_in_reaction:List[TransportReaction]
        ) -> str:
    '''from the set of reactions ocurring on a given variable, get the corresponding term in the ode'''
    
    # Term corresponding to births
    birth_term = '' 
    if len(birth_reaction) == 1:
        br = birth_reaction[0]
        # Constant rate birth reactions have a very simple term
        if   type(br) == ConstantBirthReaction:
            birth_term += f'+{varname}*{br.rate} '
        
        # Dynamic birth reactions have a more complicated term
        elif type(br) == DynamicBirthReaction:
            birth_term += f'+{varname}*np.max([0, {br.birthrate}+{br.controlstrength}*({br.targetpop}-{br.etype_0_varnamestr}-({br.delta}*{br.etype_1_varnamestr}))]) '

    # Term corresponding to deaths
    death_term = ''
    if len(death_reaction) == 1:
        death_term += f'-{varname}*{death_reaction[0].rate} '
    
    # If a given node has outward transport, add the corresponding reactions
    transport_out_term = ''
    if len(transport_out_reaction) > 0:
        transport_out_term += f'-{varname}*('
        for reaction in transport_out_reaction:
            transport_out_term += f'+{reaction.rate}'
        transport_out_term += ') '

    #  If a given node has inward transport, add the corresponding reactions
    transport_in_term = ''
    if len(transport_in_reaction) > 0:
        transport_in_term += f'+('
        for reaction in transport_out_reaction:
            transport_in_term += f'+{reaction.dest_varnamestr}*{reaction.rate}'
        transport_in_term += ') '


    # Generate the ODE term
    ode_term = f'\t\t{birth_term}{death_term}{transport_out_term}{transport_in_term},\n'

    # Generate the corresponding comment term, with useful annotations
    comment_term = f'# Δ{varname}/Δt\t'
    if birth_term != '': 
        comment_term += f'<birth>{" "*(len(birth_term)-7)}'
    if death_term != '':
        comment_term += f'<death>{" "*(len(death_term)-7)}'
    if transport_in_term != '':
        comment_term += f'<outflow>{" "*(len(transport_out_term)-9)}'
    if transport_out_term != '':
        comment_term += f'<inflow>'
    comment_term += '\n'
    
    return comment_term+ode_term


    
class OdeParameters():
    '''get the parameters required for the automatic generation of the ODE representation of the reaction network'''
    def __init__(self, networkreactions:Reactions):
        '''
        the __init__ method generates the string version of the source code of an ODE model
        corresponding to the reactions.
        '''
        # Collect the equation corresponding to each variable
        ode_terms = ''
        for variable in networkreactions.reaction_variables:
            ode_terms += (
                get_ode_term(
                    variable,
                    networkreactions.get_birth_reaction(variable),
                    networkreactions.get_death_reaction(variable),
                    networkreactions.get_transport_reaction_out(variable),
                    networkreactions.get_transport_reaction_in(variable)
                    )
                )  
        
        # Initialize the program
        self.ode_program_text = "global ODE_model\ndef ODE_model(t, z):\n"

        # Set up tuple of variables (by replacing elements in the string version of the list of variables)
        vars_for_code = str((str(networkreactions.reaction_variables)[1:-1]).replace("'","")) 
        self.ode_program_text += f'# Variables (<node.id>_<etype>):\n'
        self.ode_program_text += f'\t{vars_for_code} = z\n'
        
        # set up how each variable changes
        self.ode_program_text += '# Terms for each variable:'
        self.ode_program_text += f"\n\treturn [\n{ode_terms}\t\t]"

    def __str__(self) -> str:
        return self.ode_program_text
            
            
    