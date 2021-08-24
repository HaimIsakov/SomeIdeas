from accelerated_graph_features.attractor_basin import AttractorBasinCalculator
from accelerated_graph_features.bfs_moments import BfsMomentsCalculator
from accelerated_graph_features.flow import FlowCalculator
from accelerated_graph_features.k_core import KCoreCalculator
from accelerated_graph_features.motifs import nth_nodes_motif
from accelerated_graph_features.page_rank import PageRankCalculator
from average_neighbor_degree import AverageNeighborDegreeCalculator
from betweenness_centrality import BetweennessCentralityCalculator
from closeness_centrality import ClosenessCentralityCalculator
from communicability_betweenness_centrality import CommunicabilityBetweennessCentralityCalculator
from eccentricity import EccentricityCalculator
from feature_calculators import FeatureMeta
from fiedler_vector import FiedlerVectorCalculator
from general import GeneralCalculator
from hierarchy_energy import HierarchyEnergyCalculator
from load_centrality import LoadCentralityCalculator
from louvain import LouvainCalculator
# new
from features_algorithms.vertices.eigenvector_centrality import EigenvectorCentralityCalculator
from features_algorithms.vertices.clustering_coefficient import ClusteringCoefficientCalculator
from features_algorithms.vertices.square_clustering_coefficient import SquareClusteringCoefficientCalculator
from features_algorithms.vertices.generalized_degree import GeneralizedDegreeCalculator
from features_algorithms.vertices.all_pairs_shortest_path_length import AllPairsShortestPathLengthCalculator
from features_algorithms.vertices.all_pairs_shortest_path import AllPairsShortestPathCalculator


class FeaturesMeta:
    """
    The following are the implemented features. This file includes the accelerated versions for the features which have
    the option, whereas the similar features_meta.py file includes the regular versions.
    For each feature, the comment to the right describes to which graph they are intended.
    We split the features into 3 classes by duration, below.
    """

    def __init__(self, gpu=False, device=0):
        self.NODE_LEVEL = {
            "attractor_basin": FeatureMeta(AttractorBasinCalculator, {"ab"}),  # Directed
            "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"avg_nd"}),  # Any
            "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),  # Any
            "bfs_moments": FeatureMeta(BfsMomentsCalculator, {"bfs"}),  # Any
            "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),  # Any
            "communicability_betweenness_centrality": FeatureMeta(CommunicabilityBetweennessCentralityCalculator,
                                                                  {"communicability"}),  # Undirected
            "eccentricity": FeatureMeta(EccentricityCalculator, {"ecc"}),  # Any
            "fiedler_vector": FeatureMeta(FiedlerVectorCalculator, {"fv"}),  # Undirected (due to a code limitation)
            "flow": FeatureMeta(FlowCalculator, {}),  # Directed
            # General - calculating degrees. Directed will get (in_deg, out_deg) and undirected will get degree only per vertex.
            "general": FeatureMeta(GeneralCalculator, {"gen"}),  # Any
            "hierarchy_energy": FeatureMeta(HierarchyEnergyCalculator, {"hierarchy"}),  # Directed (but works for any)
            "k_core": FeatureMeta(KCoreCalculator, {"kc"}),  # Any
            "load_centrality": FeatureMeta(LoadCentralityCalculator, {"load_c"}),  # Any
            "louvain": FeatureMeta(LouvainCalculator, {"lov"}),  # Undirected
            "motif3": FeatureMeta(nth_nodes_motif(3, gpu, device), {"m3"}),  # Any
            "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),  # Directed (but works for any)
            "motif4": FeatureMeta(nth_nodes_motif(4, gpu, device), {"m4"}),  # Any
            # new
            "eigenvector_centrality": FeatureMeta(EigenvectorCentralityCalculator, {"eigenvector"}),
            "clustering_coefficient": FeatureMeta(ClusteringCoefficientCalculator, {"clustering"}),
            "square_clustering_coefficient": FeatureMeta(SquareClusteringCoefficientCalculator, {"square_clustering"}),
            "generalized_degree": FeatureMeta(GeneralizedDegreeCalculator, {"generalized_degree"}),
            "all_pairs_shortest_path_length": FeatureMeta(AllPairsShortestPathLengthCalculator,
                                                          {"all_pairs_shortest_path_length"}),
            "all_pairs_shortest_path": FeatureMeta(AllPairsShortestPathCalculator, {"all_pairs_shortest_path"}),
        }

        self.MOTIFS = {
            "motif3": FeatureMeta(nth_nodes_motif(3, gpu, device), {"m3"}),
            "motif4": FeatureMeta(nth_nodes_motif(4, gpu, device), {"m4"})
        }

        """
        Features by duration:
        Short:
            - Average neighbor degree
            - General
            - Louvain
            - Hierarchy energy
            - Motif 3
            - K core
            - Attraction basin
            - Page Rank 
            
        Medium:
            - Fiedler vector
            - Closeness centrality 
            - Eccentricity
            - Load centrality
            - BFS moments
            - Flow
            - Motif 4
        Long:
            - Betweenness centrality
            - Communicability betweenness centrality
        """
