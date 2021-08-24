from features_algorithms.vertices.attractor_basin import AttractorBasinCalculator
from features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
from features_algorithms.vertices.bfs_moments import BfsMomentsCalculator
from features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
from features_algorithms.vertices.communicability_betweenness_centrality import \
    CommunicabilityBetweennessCentralityCalculator
from features_algorithms.vertices.eccentricity import EccentricityCalculator
from features_algorithms.vertices.fiedler_vector import FiedlerVectorCalculator
from features_algorithms.vertices.flow import FlowCalculator
from features_algorithms.vertices.general import GeneralCalculator
from features_algorithms.vertices.hierarchy_energy import HierarchyEnergyCalculator
from features_algorithms.vertices.k_core import KCoreCalculator
from features_algorithms.vertices.load_centrality import LoadCentralityCalculator
from features_algorithms.vertices.louvain import LouvainCalculator
# from features_algorithms.vertices.neighbor_nodes_histogram import nth_neighbor_calculator
from features_algorithms.vertices.motifs import nth_nodes_motif
from features_algorithms.vertices.page_rank import PageRankCalculator
from features_infra.feature_calculators import FeatureMeta, FeatureCalculator

# new
from features_algorithms.vertices.eigenvector_centrality import EigenvectorCentralityCalculator
from features_algorithms.vertices.clustering_coefficient import ClusteringCoefficientCalculator
from features_algorithms.vertices.square_clustering_coefficient import SquareClusteringCoefficientCalculator
from features_algorithms.vertices.generalized_degree import GeneralizedDegreeCalculator
from features_algorithms.vertices.all_pairs_shortest_path_length import AllPairsShortestPathLengthCalculator
from features_algorithms.vertices.all_pairs_shortest_path import AllPairsShortestPathCalculator


class FeaturesMeta:
    """
    The following are the implemented features. This file includes the regular versions for feature calculators, whereas
    the similar accelerated_features_meta.py file includes the the accelerated features (for the features that have the
    accelerated version).
    For each feature, the comment to the right describes to which graph they are intended.
    We split the features into 3 classes by duration, below.
    """

    def __init__(self):
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
            "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),  # Any
            "page_rank": FeatureMeta(PageRankCalculator, {"pr"}),  # Directed (but works for any)
            "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"}),  # Any
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
            "motif3": FeatureMeta(nth_nodes_motif(3), {"m3"}),
            "motif4": FeatureMeta(nth_nodes_motif(4), {"m4"})
        }

    """
    Features by duration:
    Short:
        - Average neighbor degree
        - General
        - Louvain
        - Hierarchy energy
    Medium:
        - Fiedler vector
        - K core
        - Motif 3
        - Closeness centrality 
        - Attraction basin
        - Eccentricity
        - Load centrality
        - Page rank
    Long:
        - Betweenness centrality
        - BFS moments
        - Communicability betweenness centrality
        - Flow
        - Motif 4
    """
