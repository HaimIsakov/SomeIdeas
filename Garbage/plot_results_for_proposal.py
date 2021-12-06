import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind

only_values = [0.62, 0.59, 0.6, 0.54, 0.42, 0.55, 0.62, 0.64, 0.71, 0.64, 0.68]
only_graph_structure = [0.57, 0.61, 0.5, 0.64, 0.6, 0.5, 0.48, 0.57, 0.52, 0.55]
values_and_graph_structure = [0.53, 0.68, 0.59, 0.57, 0.54, 0.61, 0.55, 0.71, 0.52, 0.72]

only_values_mean = np.mean(only_values)
only_values_std = np.std(only_values)
print(f"only_values_mean {only_values_mean}")
print(f"only_values_std {only_values_std}")

only_graph_structure_mean = np.mean(only_graph_structure)
only_graph_structure_std = np.std(only_graph_structure)
print(f"only_graph_structure_mean {only_graph_structure_mean}")
print(f"only_graph_structure_std {only_graph_structure_std}")

values_and_graph_structure_mean = np.mean(values_and_graph_structure)
values_and_graph_structure_std = np.std(values_and_graph_structure)
print(f"values_and_graph_structure_mean {values_and_graph_structure_mean}")
print(f"values_and_graph_structure_std {values_and_graph_structure_std}")

stat, pvalue = ttest_ind(only_values, values_and_graph_structure)
print(f"{stat}, {pvalue}")
