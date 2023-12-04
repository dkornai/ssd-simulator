'''
THIS MODULE CONTAINS FUNCTIONS TO SIMULATE THE NETWORK OF REACTIONS
'''

from typing import Callable, Type
import numpy as np; np.set_printoptions(suppress=True)
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from tqdm import tqdm

import scipy.integrate as integrate

from reactions import CmeParameters, OdeParameters
import libsimbackend

# Set up custom types for hinting
npVec_i32 = Type[np.ndarray[np.int32, np.ndim(1)]]
npMat_i32 = Type[np.ndarray[np.int32, np.ndim(2)]]
npVec_f64 = Type[np.ndarray[np.float64, np.ndim(1)]]


# wrapper for the ode model
def simulate_ode(
        ode_param:      OdeParameters, 
        time_points:    np.ndarray,
        start_state:    np.ndarray,
        ) ->            np.ndarray:
    '''take a set of ODE parameters, and execute the corresponding program to simulate the system'''
    
    # Execute the source code corresponding to the Ode Model to generate a callable model
    global ODE_model
    ODE_model = None
    exec(ode_param.ode_program_text)

    # Use scipy to solve the corresponding IVP
    sol = integrate.solve_ivp(
        ODE_model, 
        [0, time_points[-1]], 
        start_state, 
        t_eval=time_points, 
        method = 'LSODA'  # Eq system is mostly stiff, needs a solver that can handle such cases
        ) 
    
    return sol.y


def gillespie_wrapper(
        cme_paramtuple: tuple[npVec_f64, npVec_i32, npMat_i32, npMat_i32, npVec_f64, npVec_i32, npVec_i32, npVec_f64, npVec_f64, npVec_f64, npVec_f64]
        ) ->            npVec_i32:
    '''
    Wrapper for the C++ gillespie simulator module
    
    void sim_gillespie(
        const   npVec_f64       ItimePoints,                // points in time where system state should be recorded
        
        const   npVec_i32       IstateVec,                  // starting state of the system
                npMat_i32_ptr   stateVecSample,             // array where the 'sys_state' is recorded at each time in 'timePoints'
                
        const   npMat_i32_ptr   reactionMatrix,             // the update to the statevector corresponding to each reaction
        const   npVec_f64       IpercapReactionRates,       // per capita reaction rates
        const   npVec_i32       IreactionStateVecIndex,     // indeces of system state variables from which propensity is calculated
        
        // Parameters used to update dynamic birth rates
        const   npVec_i32       IdynBirthStateVecIndex,      // stateVector indeces of the nodes where birthRates are dynamic
        const   npVec_f64       IbirthRates,                   
        const   npVec_f64       ItargetPops,
        const   npVec_f64       IcontrolStrenghts,
        const   npVec_f64       Ideltas   
        )
    '''

    # Unpack tuple of variables
    timePoints, stateVec, reactionMatrix, percapReactionRates, reactionStateVecIndex, dynBirthStateVecIndex, birthRates, targetPops, controlStrenghts, deltas = cme_paramtuple
    
    # Create arrays which will be modified in place
    stateVecSample  = np.zeros((timePoints.size, stateVec.size), dtype = np.int32, order = 'F')

    # Run C++ module, which modifies 'stateVecSample' in place
    libsimbackend.sim_gillespie(
        timePoints, 
        stateVec,
        stateVecSample,
        reactionMatrix, 
        percapReactionRates, 
        reactionStateVecIndex, 
        dynBirthStateVecIndex, 
        birthRates, 
        targetPops, 
        controlStrenghts, 
        deltas
        )

    # Transpose results and return 
    return stateVecSample.transpose(1,0)

# wrapper to simulate using gillespie
def simulate_gillespie(
        cme_param:      CmeParameters,
        time_points:    np.ndarray,
        start_state:    np.ndarray,
        replicates:     int = 100,
        n_cpu:          int = 0,
        ) ->            np.ndarray:

    # Create array for output
    replicate_results   = np.zeros((replicates, len(start_state), time_points.size), dtype = np.int32)

    # Create arrays holding simulation parameters
    timePoints              = np.array(time_points,                             dtype=np.float64)
    stateVec                = np.array(start_state,                             dtype=np.int32)
    reactionMatrix          = np.array(cme_param.reaction_matrix,               dtype=np.int32, order = 'F')
    percapReactionRates     = np.array(cme_param.percap_reaction_rates,         dtype=np.float64)
    reactionStateVecIndex   = np.array(cme_param.reaction_statevec_index,       dtype=np.int32)
    dynBirthStateVecIndex   = np.array(cme_param.dynamic_birth_statevec_index,  dtype=np.int32)
    birthRates              = np.array(cme_param.birthrates,                    dtype=np.float64)
    targetPops              = np.array(cme_param.targetpops,                    dtype=np.float64)
    controlStrenghts        = np.array(cme_param.controlstrengths,              dtype=np.float64)
    deltas                  = np.array(cme_param.deltas,                        dtype=np.float64)

    # If the number of cpus is not specified, use all cores
    if n_cpu == 0:
        n_cpu = cpu_count()


    print('Simulating using gillespie...')
    pbar = tqdm(total=replicates)

    with Pool(n_cpu) as pool:
        # prepare arguments as list of tuples
        param = [(timePoints, stateVec, reactionMatrix, percapReactionRates, reactionStateVecIndex, dynBirthStateVecIndex, birthRates, targetPops, controlStrenghts, deltas) for _ in range(replicates)]
        
        # make list that unordered results will be deposited to
        pool_results = []

        # execute tasks
        for result in pool.imap_unordered(gillespie_wrapper, param):
            pool_results.append(result)
            pbar.update(1)

        # write to output array
        for i in range(replicates): replicate_results[i,:,:] = pool_results[i]

    return replicate_results


# wrapper for the c++ tau leaping simulator module
def tauleaping_wrapper(
        vartup:         tuple[np.ndarray, float, list, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
        ) ->            np.ndarray:

    # unpack tuple of variables
    time_points, timestep, start_state, reactions, react_rates, state_index, birth_update_par, n_birth_updates = vartup
    
    # create arrays which will be modified in place
    sys_state           = np.array(start_state, dtype=np.int32)
    sys_state_sample    = np.zeros((time_points.size, sys_state.size), dtype = np.int32, order = 'F')

    # run c++ module, which modifies 'sys_state_sample' in place
    libsimbackend.sim_tauleaping(
        time_points,
        timestep, 
        sys_state, 
        sys_state_sample, 
        reactions, 
        react_rates, 
        state_index, 
        birth_update_par, 
        n_birth_updates
        )


    # transpose and return 
    return sys_state_sample.transpose(1,0)

# wrapper to simulate using tau leaping
def simulate_tauleaping(
        gill_param:     dict,
        time_points:    np.ndarray,
        start_state:    list,
        replicates:     int = 100,
        timestep:       float = 0.01,
        n_cpu:          int = 0,
        ) ->            np.ndarray:

    # create array for output
    replicate_results   = np.zeros((replicates, len(start_state), time_points.size), dtype = np.int32)

    # create arrays holding simulation parameters
    time_points         = np.array(time_points, dtype = np.float64)
    react_rates         = np.array(gill_param['reactions']['reaction_rates'], dtype=np.float64)
    state_index         = np.array(gill_param['reactions']['state_index'], dtype=np.int32)
    reactions           = np.array(gill_param['reactions']['reactions'], dtype=np.int32, order = 'F')
    birth_update_par    = np.array(gill_param['update_rate_birth']['rate_update_birth_par'][0], dtype = np.float64)
    n_birth_updates     = int(len(gill_param['update_rate_birth']['rate_update_birth_par'])*2)

    print('simulating using tau leaping...')
    pbar = tqdm(total=replicates)

    if n_cpu == 0:
        n_cpu = cpu_count()
    with Pool(n_cpu) as pool:
        # prepare arguments as list of tuples
        param = [(time_points, timestep, start_state, reactions, react_rates, state_index, birth_update_par, n_birth_updates) for _ in range(replicates)]
        
        # make list that unordered results will be deposited to
        pool_results = []

        # execute tasks
        for result in pool.imap_unordered(tauleaping_wrapper, param):
            pool_results.append(result)
            pbar.update(1)

        # write to output array
        for i in range(replicates): replicate_results[i,:,:] = pool_results[i]


    return replicate_results