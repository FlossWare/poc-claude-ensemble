import random
import unittest
import ranking_service as rs


class RankingTests(unittest.TestCase):
    def test_candidate_in_both_lists_ranks_first(self):
        result = rs.fuse({"lexical": [{"key": "a", "rank": 1}, {"key": "b", "rank": 2}],
                          "vector": [{"key": "b", "rank": 1}, {"key": "c", "rank": 2}]}, 3)
        self.assertEqual(result[0]["key"], "b")

    def test_contributions_explain_the_score(self):
        result = rs.fuse({"lexical": [{"key": "a", "rank": 1}], "vector": [{"key": "a", "rank": 3}]}, 1)
        self.assertEqual(set(result[0]["contributions"]), {"lexical", "vector"})
        self.assertAlmostEqual(result[0]["score"], round(1 / 61 + 1 / 63, 6), places=6)

    def test_order_independent_of_input_order(self):
        items = [{"key": f"k{i}", "rank": i + 1} for i in range(20)]
        shuffled = items[:]
        random.Random(7).shuffle(shuffled)
        self.assertEqual(rs.fuse({"v": items}, 20), rs.fuse({"v": shuffled}, 20))

    def test_ties_break_on_key(self):
        result = rs.fuse({"a": [{"key": "y", "rank": 1}], "b": [{"key": "x", "rank": 1}]}, 2)
        self.assertEqual([r["key"] for r in result], ["x", "y"])

    def test_duplicate_within_a_list_counts_once(self):
        result = rs.fuse({"a": [{"key": "x", "rank": 1}, {"key": "x", "rank": 5}]}, 1)
        self.assertAlmostEqual(result[0]["score"], round(1 / 61, 6), places=6)

    def test_top_k_limits_output(self):
        items = [{"key": f"k{i}", "rank": i + 1} for i in range(10)]
        self.assertEqual(len(rs.fuse({"v": items}, 3)), 3)

    def test_rejections(self):
        for args in [({}, 5), ({"a": "nope"}, 5), ({"a": [{"key": "", "rank": 1}]}, 5),
                     ({"a": [{"key": "x", "rank": 0}]}, 5), ({"a": []}, 0), ({"a": []}, 501)]:
            with self.assertRaises(ValueError):
                rs.fuse(*args)

    def test_non_loopback_bind_refused(self):
        with self.assertRaises(ValueError):
            rs.create_server("0.0.0.0", 0)


if __name__ == "__main__":
    unittest.main()
