import os
import numpy as np
import pickle
from operator import itemgetter
from sklearn.linear_model import LinearRegression
from scipy.special import comb


# This class is intended to calculate all the additional features we used in the clique detection trials.
class AdditionalFeatures:
    def __init__(self, params, gnx, matrix, motifs=None):
        self._params = params
        self._matrix = matrix
        self._gnx = gnx
        assert params["subgraph"] == "clique", "This class is for cliques only"
        self._load_other_things(motifs)

    def _load_other_things(self, motifs):
        self._mp = MotifProbability(self._params['vertices'], self._params['probability'],
                                    self._params['subgraph_size'], self._params['directed'])
        if motifs is None:
            self._clique_motifs = self._mp.get_3_clique_motifs(3) + self._mp.get_3_clique_motifs(4)
        else:
            self._clique_motifs = motifs

    def _residual(self):
        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        res = np.zeros(self._matrix.shape)
        res_expected_nc = []
        res_expected_c = []
        degrees = np.array([j for (_, j) in self._gnx.degree()])
        reshaped_degrees = degrees.reshape(-1, 1)
        for motif in range(self._matrix.shape[1]):
            reg = LinearRegression(fit_intercept=True)
            reg.fit(reshaped_degrees, self._matrix[:, motif])
            res[:, motif] = self._matrix[:, motif] - ((reg.coef_[0] * degrees) + reg.intercept_)
            res_expected_nc.append(expected_clique[motif] - ((reg.coef_[0] * (
                    2 * self._params['probability'] * (self._params['vertices'] - 1))
                                                              ) + reg.intercept_))
            res_expected_c.append(expected_clique[motif] - ((reg.coef_[0] * (
                    2 * self._params['probability'] * (self._params['vertices'] - 1) + self._params['subgraph_size'] - 1)
                                                             ) + reg.intercept_))
        return res, res_expected_c, res_expected_nc

    def calculate_extra_ftrs(self):
        dot_excl = []  # dot product with expected clique
        dot_exncl = []  # dot product with expected non clique
        proj_excl = []  # projection on expected clique
        proj_exncl = []  # projection on expected non clique
        dist_excl = []  # distance from expected clique
        dist_exncl = []  # distance from expected non clique
        lgdist_excl = []  # distance of log vector from log expected clique
        lgdist_exncl = []  # distance of log vector from log expected non clique
        zproj_excl = []  # projection of z-scored vector on z-scored expected clique
        zproj_exncl = []  # projection of z-scored vector on z-scored expected non clique
        zdist_excl = []  # distance of z-scored vector from z-scored expected clique
        zdist_exncl = []  # distance of z-scored vector from z-scored expected non clique
        sum_motifs = []
        regsum = []  # sum all motif residuals after linear regression of motif(degree) for every motif.
        tnbr_sum = []  # num. neighbors to which a vertex is connected (<->) of top 10% vertices by sum motifs.
        cc4 = []  # clustering coefficient
        tcc = []  # mean of cc for |clique-size| neighbors (<->) with this largest value.

        expected_clique = [self._mp.motif_expected_clique_vertex(motif) for motif in self._clique_motifs]
        expected_non_clique = [self._mp.motif_expected_non_clique_vertex(motif)
                               for motif in self._clique_motifs]
        means = np.mean(self._matrix, axis=0)
        stds = np.std(self._matrix, axis=0)
        log_expected_clique = np.log(expected_clique)
        log_expected_non_clique = np.log(expected_non_clique)
        zscored_expected_clique = np.divide((expected_clique - means), stds)
        zscored_expected_non_clique = np.divide((expected_non_clique - means), stds)
        motif_matrix_residual, _, _ = self._residual()
        cc = np.divide(self._matrix[:, 0], np.array(
            [self._gnx.degree(v) * (self._gnx.degree(v) - 1) * (1 if self._params['directed'] else 0.5)
             for v in range(self._params['vertices'])]))
        sums = [(i, sum(motif_matrix_residual[i, :])) for i in range(self._params['vertices'])]
        sums.sort(key=itemgetter(1), reverse=True)
        top_sum = [v[0] for v in sums[:int(self._params['vertices'] / 10)]]
        bitmat = np.zeros((len(top_sum), self._params['vertices']))
        for i in range(len(top_sum)):
            for j in range(self._params['vertices']):
                if self._params['directed']:
                    bitmat[i, j] = 1 if self._gnx.has_edge(top_sum[i], j) and self._gnx.has_edge(j, top_sum[i]) else 0
                else:
                    bitmat[i, j] = 1 if self._gnx.has_edge(top_sum[i], j) else 0
        bitsum = np.sum(bitmat, axis=0)

        # Calculating
        tnbr_sum = tnbr_sum + [bitsum[i] for i in range(self._params['vertices'])]
        cc4 = cc4 + [cc[i] for i in range(self._params['vertices'])]
        for v in range(self._params['vertices']):
            motif_vector = self._matrix[v, :]
            log_motif_vector = np.log(motif_vector)
            zscored_motif_vector = np.divide((motif_vector - means), stds)
            reg_motif_vector = motif_matrix_residual[v, :]

            neighbors = set(self._gnx.successors(v)).intersection(set(self._gnx.predecessors(v))) \
                if self._params['directed'] else set(self._gnx.neighbors(v))
            neighbor_cc = [(v, cc[v]) for v in neighbors]
            neighbor_cc.sort(key=itemgetter(1), reverse=True)
            top_neighbors = neighbor_cc[:self._params['subgraph_size']]
            dot_excl.append(np.dot(motif_vector, np.transpose(expected_clique)))
            dot_exncl.append(np.dot(motif_vector, np.transpose(expected_non_clique)))
            proj_excl.append(np.vdot(motif_vector, expected_clique) / np.linalg.norm(expected_clique))
            proj_exncl.append(
                np.vdot(motif_vector, expected_non_clique) / np.linalg.norm(expected_non_clique))
            dist_excl.append(np.linalg.norm(motif_vector - expected_clique))
            dist_exncl.append(np.linalg.norm(motif_vector - expected_non_clique))
            lgdist_excl.append(np.linalg.norm(log_motif_vector - log_expected_clique))
            lgdist_exncl.append(np.linalg.norm(log_motif_vector - log_expected_non_clique))
            zproj_excl.append(
                np.vdot(zscored_motif_vector, zscored_expected_clique) / np.linalg.norm(
                    zscored_expected_clique))
            zproj_exncl.append(
                np.vdot(zscored_motif_vector, zscored_expected_non_clique) / np.linalg.norm(
                    zscored_expected_non_clique))
            zdist_excl.append(np.linalg.norm(zscored_motif_vector - zscored_expected_clique))
            zdist_exncl.append(np.linalg.norm(zscored_motif_vector - zscored_expected_non_clique))
            sum_motifs.append(sum(motif_vector))
            regsum.append(sum(reg_motif_vector))
            tcc.append(np.mean([j for i, j in top_neighbors]))
        extra_features_matrix = np.vstack((dot_excl, dot_exncl, proj_excl, proj_exncl, dist_excl, dist_exncl,
                                           lgdist_excl, lgdist_exncl, zproj_excl, zproj_exncl, zdist_excl,
                                           zdist_exncl, sum_motifs, regsum, tnbr_sum, cc4, tcc)).transpose()
        return extra_features_matrix


class MotifProbability:
    def __init__(self, size, edge_probability: float, clique_size, directed):
        self._is_directed = directed
        self._size = size
        self._probability = edge_probability
        self._cl_size = clique_size
        self._build_variations()
        self._motif_index_to_edge_num = {"motif3": self._motif_num_to_number_of_edges(3),
                                         "motif4": self._motif_num_to_number_of_edges(4)}
        self._gnx = None
        self._labels = {}

    def _build_variations(self):
        name3 = f"3_{'' if self._is_directed else 'un'}directed.pkl"
        variations_path = os.path.join(os.path.dirname(__file__), 'features_algorithms', 'motif_variations')
        path3 = os.path.join(variations_path, name3)
        self._motif3_variations = pickle.load(open(path3, "rb"))
        name4 = f"4_{'' if self._is_directed else 'un'}directed.pkl"
        path4 = os.path.join(variations_path, name4)
        self._motif4_variations = pickle.load(open(path4, "rb"))

    def _motif_num_to_number_of_edges(self, level):
        motif_edge_num_dict = {}
        if level == 3:
            variations = self._motif3_variations
        elif level == 4:
            variations = self._motif4_variations
        else:
            return
        for bit_sec, motif_num in variations.items():
            motif_edge_num_dict[motif_num] = bin(bit_sec).count('1')
        return motif_edge_num_dict

    def get_2_clique_motifs(self, level):
        if level == 3:
            variations = self._motif3_variations
            motif_3_with_2_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 6) if self._is_directed else np.binary_repr(number, 3)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[2]]]
                           + [(variations[number]) not in motif_3_with_2_clique]):
                        motif_3_with_2_clique.append(variations[number])
                else:
                    if variations[number] not in motif_3_with_2_clique:
                        motif_3_with_2_clique.append(variations[number])
            return motif_3_with_2_clique
        elif level == 4:
            variations = self._motif4_variations
            motif_4_with_2_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 12) if self._is_directed else np.binary_repr(number, 6)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[3]]] +
                           [(variations[number] + 13) not in motif_4_with_2_clique]):
                        motif_4_with_2_clique.append(variations[number] + 13)
                else:
                    if (variations[number] + 2) not in motif_4_with_2_clique:
                        motif_4_with_2_clique.append(variations[number] + 2)
            return motif_4_with_2_clique
        else:
            return []

    def get_3_clique_motifs(self, level):
        if level == 3:
            return [12] if self._is_directed else [1]
        elif level == 4:
            variations = self._motif4_variations
            motif_4_with_3_clique = []
            for number in variations.keys():
                if variations[number] is None:
                    continue
                bitnum = np.binary_repr(number, 12) if self._is_directed else np.binary_repr(number, 6)
                if self._is_directed:
                    if all([int(x) for x in [bitnum[0], bitnum[1], bitnum[3], bitnum[4], bitnum[6], bitnum[7]]] +
                           [(variations[number] + 13) not in motif_4_with_3_clique]):
                        motif_4_with_3_clique.append(variations[number] + 13)
                else:
                    if all([int(x) for x in [bitnum[5], bitnum[4], bitnum[2]]] +
                           [(variations[number] + 2) not in motif_4_with_3_clique]):
                        motif_4_with_3_clique.append(variations[number] + 2)
            return motif_4_with_3_clique
        else:
            return []

    def _for_probability_calculation(self, motif_index):
        if self._is_directed:
            if motif_index > 12:
                motif_index -= 13
                variations = self._motif4_variations
                num_edges = self._motif_index_to_edge_num['motif4'][motif_index]
                num_max = 12
                flag = 4
            else:
                variations = self._motif3_variations
                num_edges = self._motif_index_to_edge_num['motif3'][motif_index]
                num_max = 6
                flag = 3
        else:
            if motif_index > 1:
                motif_index -= 2
                variations = self._motif4_variations
                num_edges = self._motif_index_to_edge_num['motif4'][motif_index]
                num_max = 6
                flag = 4
            else:
                variations = self._motif3_variations
                num_edges = self._motif_index_to_edge_num['motif3'][motif_index]
                num_max = 3
                flag = 3
        return motif_index, variations, num_edges, num_max, flag

    def motif_probability_non_clique_vertex(self, motif_index):
        motif_index, variations, num_edges, num_max, _ = self._for_probability_calculation(motif_index)
        motifs = []
        for original_number in variations.keys():
            if variations[original_number] == motif_index:
                motifs.append(np.binary_repr(original_number, num_max))
        num_isomorphic = len(motifs)
        prob = num_isomorphic * (self._probability ** num_edges) * ((1 - self._probability) ** (num_max - num_edges))
        return prob

    def motif_expected_non_clique_vertex(self, motif_index):
        if self._is_directed:
            if motif_index > 12:
                to_choose = 4
            else:
                to_choose = 3
        else:
            if motif_index > 1:
                to_choose = 4
            else:
                to_choose = 3
        prob = self.motif_probability_non_clique_vertex(motif_index)
        return comb(self._size - 1, to_choose - 1) * prob

    @staticmethod
    def _second_condition(binary_motif, clique_edges):
        return all([int(binary_motif[i]) for i in clique_edges])

    def _clique_edges(self, flag, i):
        # Given i clique motifs (plus one we focus on) in fixed indices, get the edges that must appear in the motif.
        if self._is_directed:
            if flag == 3:
                if i == 0:
                    return []
                elif i == 1:
                    return [0, 2]
                else:
                    return [i for i in range(6)]
            else:
                if i == 0:
                    return []
                elif i == 1:
                    return [0, 3]
                elif i == 2:
                    return [0, 1, 3, 4, 6, 7]
                else:
                    return [i for i in range(12)]
        else:
            if flag == 3:
                if i == 0:
                    return []
                elif i == 1:
                    return [0]
                else:
                    return [i for i in range(3)]
            else:
                if i == 0:
                    return []
                elif i == 1:
                    return [0]
                elif i == 2:
                    return [0, 1, 3]
                else:
                    return [i for i in range(6)]

    def _specific_combination_motif_probability(self, motif_index, num_edges, num_max, flag, variations, i):
        # P(motif|i clique vertices except for the vertex on which we focus)
        clique_edges = self._clique_edges(flag, i)
        motifs = []
        for original_number in variations.keys():
            if variations[original_number] == motif_index:
                b = np.binary_repr(original_number, num_max)
                if self._second_condition(b, clique_edges):
                    motifs.append(b)
        num_iso = len(motifs)
        num_already_there = (i + 1) * i if self._is_directed else (i + 1) * i / 2
        return num_iso * self._probability ** (num_edges - num_already_there) * (
                    1 - self._probability) ** (num_max - num_edges)

    def motif_probability_clique_vertex(self, motif_index):
        motif_ind, variations, num_edges, num_max, flag = self._for_probability_calculation(motif_index)
        clique_non_clique = []
        for i in range(flag if self._cl_size > 1 else 1):
            # Probability that a specific set of vertices contains exactly i + 1 clique vertices.
            if i == 1:
                indicator = 1 if motif_index in self.get_2_clique_motifs(flag) else 0
            elif i == 2:
                indicator = 1 if motif_index in self.get_3_clique_motifs(flag) else 0
            elif i == 3:
                indicator = 1 if motif_index == 211 else 0
            else:
                indicator = 1
            if not indicator:
                clique_non_clique.append(0)
                continue
            cl_ncl_comb_prob = comb(max(self._cl_size - 1, 0), i) * comb(self._size - max(self._cl_size, 1),
                                                                         flag - 1 - i) / float(
                                   comb(self._size - 1, flag - 1))
            spec_comb_motif_prob = self._specific_combination_motif_probability(
                motif_ind, num_edges, num_max, flag, variations, i)

            clique_non_clique.append(cl_ncl_comb_prob * spec_comb_motif_prob)
        prob = sum(clique_non_clique)
        return prob

    def motif_expected_clique_vertex(self, motif_index):
        if self._is_directed:
            if motif_index > 12:
                to_choose = 4
            else:
                to_choose = 3
        else:
            if motif_index > 1:
                to_choose = 4
            else:
                to_choose = 3
        prob = self.motif_probability_clique_vertex(motif_index)
        return comb(self._size - 1, to_choose - 1) * prob

    def clique_non_clique_angle(self, motifs):
        clique_vec = [self.motif_expected_clique_vertex(m) for m in motifs]
        non_clique_vec = [self.motif_expected_non_clique_vertex(m) for m in motifs]
        return self._angle(clique_vec, non_clique_vec)

    def clique_non_clique_zscored_angle(self, mean_vector, std_vector, motifs):
        clique_vec = np.array([self.motif_expected_clique_vertex(m) for m in motifs])
        non_clique_vec = np.array([self.motif_expected_non_clique_vertex(m) for m in motifs])
        normed_clique_vec = np.divide(clique_vec - mean_vector, std_vector)
        normed_non_clique_vec = np.divide(non_clique_vec - mean_vector, std_vector)
        return self._angle(normed_clique_vec, normed_non_clique_vec)

    @staticmethod
    def _angle(v1, v2):
        cos = np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cos)
