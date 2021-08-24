import networkx as nx

from features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta


class GeneralizedDegreeCalculator(NodeFeatureCalculator):
    def __init__(self, *args, **kwargs):
        super(GeneralizedDegreeCalculator, self).__init__(*args, **kwargs)

    def _calculate(self, include: set):
        self._features = nx.algorithms.cluster.generalized_degree(self._gnx)

    def is_relevant(self):
        return True


feature_entry = {
    "generalized_degree": FeatureMeta(GeneralizedDegreeCalculator, {"generalized_degree"}),
}


if __name__ == "__main__":
    from measure_tests.specific_feature_test import test_specific_feature
    test_specific_feature(GeneralizedDegreeCalculator, is_max_connected=True)