import numpy as np

class CmeSimResults():
    def __init__(self, results:np.ndarray):
        self.raw = results

        # Collect basic information about the dimensionality of the resuls.
        self.n_samples = results.shape[0]
        self.n_vars = results.shape[1]
        self.n_timepoints = results.shape[2]

        # Average counts across replicates
        self.mean = np.nanmean(results, axis=0)

        # Collect the regions of the statevector corresponding to each entity type
        self.statevec_e0 = results[:,np.arange(0, self.n_vars, 2),:]
        self.statevec_e1 = results[:,np.arange(1, self.n_vars, 2),:]

        # Collect the mean statevector for each enetity type
        self.statevec_e0_mean = np.mean(self.statevec_e0, axis = 0)
        self.statevec_e1_mean = np.mean(self.statevec_e1, axis = 0)

        # Collect the total entity types across each system
        self.e0_count = np.sum(self.statevec_e0, axis=1)
        self.e1_count = np.sum(self.statevec_e1, axis=1)


