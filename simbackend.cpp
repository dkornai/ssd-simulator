#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <numeric>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#define EIGEN_NO_DEBUG

namespace py = pybind11;


// CUSTOM TYPES //

using IntVec       = std::vector<int>;
using DoubleVec    = std::vector<double>;

typedef py::array_t<double> npVec_f64; // 1d np.float64 array
typedef py::array_t<int>    npVec_i32; // 1d np.int32 array 

typedef Eigen::Ref<Eigen::VectorXi> npVec_i32_ptr; // view on a 1d np.float64 array, which can be directly written via Eigen syntax
typedef Eigen::Ref<Eigen::VectorXd> npVec_f64_ptr; // view on a 1d np.float64 array, which can be directly written via Eigen syntax
typedef Eigen::Ref<Eigen::MatrixXi> npMat_i32_ptr; // view on a 2d np.int32 array, which can be directly written via Eigen syntax
typedef Eigen::Ref<Eigen::MatrixXd> npMat_f64_ptr; // view on a 2d np.int32 array, which can be directly written via Eigen syntax

typedef Eigen::DiagonalMatrix<int, Eigen::Dynamic> DiagMat; // diagonalized matrix

// HELPER FUNCTIONS //

// c++ vector from np.float64 array
DoubleVec get_double_vec_from_np(
    const   npVec_f64   arr
    )
{
    py::buffer_info info = arr.request();
    double* data = static_cast<double*>(info.ptr);
    std::size_t size = info.size;
    std::vector<double> vec(data, data + size);
    return vec;
}

// c++ vector from np.int64 array
IntVec get_int_vec_from_np(
    const   npVec_i32     arr
    )
{
    py::buffer_info info = arr.request();
    int* data = static_cast<int*>(info.ptr);
    std::size_t size = info.size;
    std::vector<int> vec(data, data + size);
    return vec;
}

// Eigen int vector from np.int32 array
Eigen::VectorXi get_eig_int_vec_from_np(
    py::array_t<int> input
    ) 
{
    py::buffer_info buf = input.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("Expected a 1D numpy array");
    }

    int* ptr = static_cast<int*>(buf.ptr);
    Eigen::Map<Eigen::VectorXi> eigenVector(ptr, buf.size);

    return eigenVector;
}


// GILLESPIE FUNCTION //
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
{

    // C++ versions of numpy arrays
    const   DoubleVec       timePoints              = get_double_vec_from_np(ItimePoints);
    
            Eigen::VectorXi stateVec                = get_eig_int_vec_from_np(IstateVec);
    
            DoubleVec       percapReactionRates     = get_double_vec_from_np(IpercapReactionRates);
    const   IntVec          reactionStateVecIndex   = get_int_vec_from_np(IreactionStateVecIndex);
    
    const   IntVec          dynBirthStateVecIndex   = get_int_vec_from_np(IdynBirthStateVecIndex);
    const   DoubleVec       birthRates              = get_double_vec_from_np(IbirthRates);
    const   DoubleVec       targetPops              = get_double_vec_from_np(ItargetPops);
    const   DoubleVec       controlStrenghts        = get_double_vec_from_np(IcontrolStrenghts);
    const   DoubleVec       deltas                  = get_double_vec_from_np(Ideltas);
    
    // Counts often accessed during loops
    const   int             n_timePoints            = timePoints.size();
    const   int             n_reactions             = reactionStateVecIndex.size();

    // Variables used for calculating event rates
            DoubleVec       globalReactionRates(n_reactions);     // global rate of each reaction
            double          totalReactionRate;                    // sum of global reaction rates


    // Init a random generator
    std::mt19937 gen(std::random_device{}());

    double t = timePoints[0];
    // Loop through the time points to sample
    for (int i = 0; i < n_timePoints; ++i) {
        while (t < timePoints[i]) {
                
            // Calculate dynamic birth rates in nodes with active birthRates control,
            // avoiding negative values, and set corresponding reaction rates
            for (int svi : dynBirthStateVecIndex) {
                percapReactionRates[svi] = percapReactionRates[svi+1] 
                = std::max(0.0, birthRates[svi]+controlStrenghts[svi]*(targetPops[svi]-stateVec[svi]-(deltas[svi]*stateVec[svi+1]))); 
            }
            
            // Calculate global reaction propensity by multiplyin per capita rates 
            // with the number of reactants, while keeping track of their sum
            totalReactionRate = 0.0;
            for (int ri = 0; ri < n_reactions; ri++) {
                globalReactionRates[ri] = percapReactionRates[ri]*stateVec[reactionStateVecIndex[ri]];
                totalReactionRate += globalReactionRates[ri];
            }

            // If there are no more reactions, break the loop
            if (totalReactionRate == 0.0) {
                t = timePoints[i];
                break;
            }

            // Set up the discrete distribution parametrized by the global reaction rates
            std::discrete_distribution<> reactionProbDist(globalReactionRates.begin(), globalReactionRates.end());
            
            // Sample from the distribution to select a reaction, and modify the state vector accordingly
            stateVec += reactionMatrix.row(reactionProbDist(gen));          
            
            // Increment time forward
            std::exponential_distribution<> expoDist(totalReactionRate);
            t += expoDist(gen);
 
        }
        
        // write the current state of the system to the output array
        stateVecSample.row(i) = stateVec;
        
    }
}


// // TAU LEAPING FUNCTION //
// void sim_tauleaping(
//     const   npVec_f64      in_timePoints,             // points in time where system state should be recorded
//     const   double          timestep,                   // simulation time step
    
//     const   npVec_i32      in_sys_state,               // starting 'sys_state'
//             npMat_i32_ptr sys_state_sample,           // array where the 'sys_state' is recorded at each time in 'timePoints'
            
//     const   npMat_i32_ptr reactions,                  // 'sys_state' updates corresponding to each possible reaction
//     const   npVec_f64      in_percap_r_rates,          // per capita reaction rates
//     const   npVec_i32      in_state_index,             // indeces of system state variables from which propensity is calculated

//     const   npVec_f64      in_birthRate_updates_par,   // parameters used to update dynamic birth rates
//     const   int             n_birthRate_updates         // number of birth rate reactions which must be updated
//     )
// {   
//     // ### VARIABLE SETUP #### //

//     // c++ versions of numpy arrays
//     const   DoubleVec      timePoints     = get_double_vec_from_np(in_timePoints);
//             DoubleVec      percap_r_rates  = get_double_vec_from_np(in_percap_r_rates);
//     const   IntVec         state_index     = get_int_vec_from_np(in_state_index);
//             Eigen::VectorXi sys_state       = get_eig_int_vec_from_np(in_sys_state);
            
//     // counts often accessed during loops
//     const   int             n_timePoints   = timePoints.size();
//     const   int             n_reactions     = state_index.size();
//     const   int             n_pops          = sys_state.size();

//     // variables used in calculating the dynamic birth rates
//     const   DoubleVec      br_up_par       = get_double_vec_from_np(in_birthRate_updates_par);
//     const   double          c_b             = br_up_par[0];
//     const   double          mu              = br_up_par[1];
//     const   double          nss             = br_up_par[2];
//     const   double          deltas           = br_up_par[3];
//     const   double          brh1            = mu + c_b * nss;   // precomputed values that do not change across iterations
//     const   double          brh2            = c_b * deltas;
    
//     // variables used to derive the rate of poisson processes
//             DiagMat  diagonalized_n_events(n_reactions); // a diagonalized matrix where each element represents the number of times a given reaction occures
//             Eigen::MatrixXi product(n_reactions, n_pops);       // a matrix matrix product of the diagonalized event count matrix and the reaction matrix


//     // ### SIMULATOR #### //

//     // init a random generator
//     std::mt19937 gen(std::random_device{}());

    
//     double t = timePoints[0]; 
//     // loop through the time points to sample
//     for (int i = 0; i < n_timePoints; ++i) {

//         // check if the system has completely exhausted, and exit if needed (input array is all 0s anyway)
//         if (sys_state.isZero()) {
//             break;
//         }

//         // while the next time point to sample the system state is reached
//         while (t < timePoints[i]) {
            
//             // avoiding negative values, calculate dynamic birth rates in nodes with active birthRates control, and set corresponding reaction rates
//             for (int j = 0; j < n_birthRate_updates; j+=2) {
//                 percap_r_rates[j] = percap_r_rates[j+1] = std::max(0.0, (mu + c_b*(nss - sys_state[j] - (deltas*sys_state[j+1]))));  
//             }

//             // calculate rates, and use as mean of poisson, and then generate the number of times each reaction occurs during the timestep.
//             for (int j = 0; j < n_reactions; ++j) {
//                 std::poisson_distribution<> dist(percap_r_rates[j]*sys_state[state_index[j]]*timestep);
//                 diagonalized_n_events.diagonal()(j) = dist(gen);
//             }

//             // update the state of the system by adding each reaction the correct number of times                     
//             product = diagonalized_n_events*reactions;              
//             sys_state += product.colwise().sum();         
//             sys_state = sys_state.cwiseMax(0); // guarantee that there are no underflows
            
//             // increment time forward
//             t += timestep;
//         }
        
//         // write the current state of the system to the output array
//         stateVecSample.row(i) = sys_state;

//     }
// }





// PYBIND // 
PYBIND11_MODULE(libsimbackend, m)
{
    m.def("sim_gillespie", &sim_gillespie, "Simulate CME using gillespie");
    // m.def("sim_tauleaping", &sim_tauleaping, "simulate CME using tau leaping");
}