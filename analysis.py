import numpy as np

class CmeSimResults():
    def __init__(self, results:np.ndarray):
        self.raw = results
       
        n_vars = results.shape[2]

        # Average counts across replicates
        self.average = np.nanmean(results, axis=0)

