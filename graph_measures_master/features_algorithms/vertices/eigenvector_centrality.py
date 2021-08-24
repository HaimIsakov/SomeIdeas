import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class EigenvectorCentralityCalculator(NodeFeatureCalculator):
    def __init__(self, *args, max_iter=1000, **kwargs):
        super(EigenvectorCentralityCalculator, self).__init__(*args, **kwargs)
        self._max_iter = max_iter

    def _calculate(self, include: set):
        self._features = nx.eigenvector_centrality(self._gnx, max_iter=self._max_iter)

    def is_relevant(self):
        return True


feature_entry = {
    "eigenvector_centrality": FeatureMeta(EigenvectorCentralityCalculator, {"eigenvector"}),
}

if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature

    test_specific_feature(EigenvectorCentralityCalculator, is_max_connected=True)
