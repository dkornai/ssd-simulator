import numpy as np

class CmeSimResults():
    def __init__(self, results:np.ndarray):
        self.raw = results

        # Collect basic information about the dimensionality of the resuls.
        self.n_samples = results.shape[0]
        self.n_vars = results.shape[1]
        self.n_timepoints = results.shape[2]

        # Average counts across replicates
        self.average = np.nanmean(results, axis=0)

        # Collect entity types seperately
        self.e0_count = results[:,np.arange(0, self.n_vars, 2),:]
        self.e1_count = results[:,np.arange(1, self.n_vars, 2),:]


