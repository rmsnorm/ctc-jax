import unittest

import metrics


class TestCollapseRepetitions(unittest.TestCase):
    def test_no_blank(self):
        x = [1, 1, 1, 1, 2, 2, 2]
        blank_id = 0
        self.assertEqual(metrics.collapse_repetitions(x, blank_id), [1, 2])

    def test_blank_between_unique(self):
        x = [1, 1, 1, 1, 0, 2, 2, 2]
        blank_id = 0
        self.assertEqual(metrics.collapse_repetitions(x, blank_id), [1, 2])

    def test_blank_between_same(self):
        x = [1, 1, 1, 0, 1, 0, 2, 2, 2]
        blank_id = 0
        self.assertEqual(metrics.collapse_repetitions(x, blank_id), [1, 1, 2])

    def test_blank_at_begin(self):
        x = [0, 1, 1, 1, 1, 2, 2, 2]
        blank_id = 0
        self.assertEqual(metrics.collapse_repetitions(x, blank_id), [1, 2])

    def test_blank_at_end(self):
        x = [1, 1, 1, 1, 2, 2, 2, 0]
        blank_id = 0
        self.assertEqual(metrics.collapse_repetitions(x, blank_id), [1, 2])

    def test_repeated_blanks(self):
        x = [0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2]
        blank_id = 0
        self.assertEqual(metrics.collapse_repetitions(x, blank_id), [1, 2])

    def test_complex(self):
        x = [0, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 0, 2, 0]
        blank_id = 0
        self.assertEqual(metrics.collapse_repetitions(x, blank_id), [1, 1, 2, 2])


class TestLabelError(unittest.TestCase):
    def test_noblank(self):
        x = [1, 1, 1, 1, 2, 2]
        y = [1, 2, 3]
        self.assertEqual(metrics.label_error(x, y, 0), 1)

    def test_blank(self):
        x = [0, 0, 1, 1, 0, 1, 1, 2, 2]
        y = [1, 2, 3]
        self.assertEqual(metrics.label_error(x, y, 0), 2)
