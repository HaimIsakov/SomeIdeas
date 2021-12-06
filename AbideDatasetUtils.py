import os
import numpy as np
from nilearn import connectome

def load_connectivity(input_matrix):
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    conn_array = conn_measure.fit_transform([input_matrix])[0]
    conn_array = np.delete(conn_array, 82, axis=0)
    conn_array = np.delete(conn_array, 82, axis=1)
    np.fill_diagonal(conn_array, 0) # Since the diagonal has 1 correlation value
    network = conn_array
    return network
