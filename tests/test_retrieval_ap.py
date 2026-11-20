import ast
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_ranking_methods():
    """Use the production methods without importing the model and corpus stack."""
    path = Path(__file__).resolve().parents[1] / 'evaluator/judgement/judger.py'
    module = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    judger = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'Judger')
    judger.body = [node for node in judger.body if isinstance(node, ast.FunctionDef) and node.name in {'_MAP', '_MRR'}]
    module.body = [judger]
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['Judger']()


def ap_at_k(relevant, ranking, cutoff):
    """Compute Appendix B, Equation 8 using exact fractions and prefix sets."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    precision = [
        Fraction(len(relevant.intersection(ranking[:rank])), rank)
        for rank, doc_id in enumerate(ranking[:cutoff], start=1) if doc_id in relevant
    ]
    return float(sum(precision) / len(relevant))


class RetrievalAPTest(unittest.TestCase):
    judge = load_ranking_methods()

    def scores(self, relevant, ranking, cutoff):
        nodes = [SimpleNamespace(id_=doc_id) for doc_id in relevant]
        results = [{'id': doc_id} for doc_id in ranking]
        return self.judge._MAP(nodes, results, cutoff), self.judge._MRR(nodes, results, cutoff)

    def test_paper_examples_and_cutoffs(self):
        for relevant, ranking, cutoff, ap, mrr in (
            ('AB', 'AC', 2, 0.5, 1.0),
            ('AB', 'AB', 2, 1.0, 1.0),
            ('AB', 'CD', 2, 0.0, 0.0),
            ('AB', 'CA', 2, 0.25, 0.5),
            ('A', 'A', 1, 1.0, 1.0),
            ('A', 'C', 1, 0.0, 0.0),
            ('', 'AC', 2, 0.0, 0.0),
            ('AB', 'ABC', 1, 0.5, 1.0),
            ('AB', 'ACB', 2, 0.5, 1.0),
            ('AB', 'AC', 5, 0.5, 1.0),
            ('AB', '', 5, 0.0, 0.0),
            ('AB', 'AB', 0, 0.0, 0.0),
        ):
            with self.subTest(relevant=relevant, ranking=ranking, cutoff=cutoff):
                self.assertEqual(ap_at_k(relevant, ranking, cutoff), ap)
                self.assertEqual(self.scores(relevant, ranking, cutoff), (ap, mrr))

    def test_repeated_ground_truth_id_counts_as_one_document(self):
        self.assertEqual(self.scores('AAB', 'AC', 2), (0.5, 1.0))

    def test_unique_rankings_match_oracle_and_stay_bounded(self):
        documents = 'ABCD'
        for truth_size in range(len(documents) + 1):
            for relevant in combinations(documents, truth_size):
                for rank_size in range(len(documents) + 1):
                    for ranking in permutations(documents, rank_size):
                        for cutoff in range(len(documents) + 2):
                            ap, _ = self.scores(relevant, ranking, cutoff)
                            self.assertTrue(0.0 <= ap <= 1.0)
                            self.assertAlmostEqual(ap, ap_at_k(relevant, ranking, cutoff),
                                                   msg=f'{relevant=}, {ranking=}, {cutoff=}')
