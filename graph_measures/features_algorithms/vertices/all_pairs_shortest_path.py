import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class AllPairsShortestPathCalculator(NodeFeatureCalculator):
    def __init__(self, *args, **kwargs):
        super(AllPairsShortestPathCalculator, self).__init__(*args, **kwargs)

    def _calculate(self, include: set):
        self._features = nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path(self._gnx)

    def is_relevant(self):
        return True


feature_entry = {
    "all_pairs_shortest_path": FeatureMeta(AllPairsShortestPathCalculator, {"all_pairs_shortest_path"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(AllPairsShortestPathCalculator, is_max_connected=True)