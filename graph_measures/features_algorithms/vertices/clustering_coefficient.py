import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class ClusteringCoefficientCalculator(NodeFeatureCalculator):
    def __init__(self, *args, **kwargs):
        super(ClusteringCoefficientCalculator, self).__init__(*args, **kwargs)

    def _calculate(self, include: set):
        self._features = nx.algorithms.cluster.clustering(self._gnx)

    def is_relevant(self):
        return True


feature_entry = {
    "clustering_coefficient": FeatureMeta(ClusteringCoefficientCalculator, {"clustering"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(ClusteringCoefficientCalculator, is_max_connected=True)