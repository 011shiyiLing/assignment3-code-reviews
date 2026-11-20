"""Regression checks for neighbor selection and simultaneous label repair."""

import unittest

import numpy as np

from noisy_classification_template import repair_labels_knn


def reference_repair(features, labels, neighbors):
    """Small exhaustive oracle: sort other rows by distance, then row index."""
    result = []
    for row, point in enumerate(features):
        others = [index for index in range(len(features)) if index != row]
        others.sort(key=lambda index: (np.linalg.norm(features[index] - point), index))
        result.append(1 if labels[others[:neighbors]].sum() > 0 else -1)
    return np.array(result)


class RepairLabelsTests(unittest.TestCase):
    def assert_matches_reference(self, features, labels, neighbors=5):
        original = labels.copy()
        actual = repair_labels_knn(features, labels, neighbors)
        np.testing.assert_array_equal(actual, reference_repair(features, labels, neighbors))
        np.testing.assert_array_equal(labels, original)

    def test_random_points(self):
        rng = np.random.default_rng(42)
        self.assert_matches_reference(rng.normal(size=(100, 2)), rng.choice([-1, 1], 100))

    def test_equidistant_boundary_and_order_dependence(self):
        features = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
        labels = np.array([-1, 1, 1, 1, -1, -1, -1])
        self.assert_matches_reference(features, labels)
        self.assertEqual(repair_labels_knn(features, labels, 5)[0], 1)
        permutation = [0, 6, 2, 3, 4, 5, 1]
        self.assert_matches_reference(features[permutation], labels[permutation])
        self.assertEqual(repair_labels_knn(features[permutation], labels[permutation], 5)[0], -1)

    def test_duplicate_points_exclude_only_self(self):
        self.assert_matches_reference(np.zeros((12, 2)), np.array([1, -1] * 6))

    def test_all_other_points_are_neighbors(self):
        self.assert_matches_reference(np.zeros((6, 2)), np.array([1, -1] * 3))

    def test_invalid_neighbor_count(self):
        for neighbors in (0, 6):
            with self.assertRaises(ValueError):
                repair_labels_knn(np.zeros((6, 2)), np.ones(6), neighbors)


if __name__ == "__main__":
    unittest.main()
