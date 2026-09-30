"""Автотесты ЛР2 «Модель Джелинского–Моранды».

Для каждого варианта проверяется сходимость метода Ньютона, K > 0 и B > n.
Для варианта 2 (команда №2) результаты сверяются со значениями из отчёта (рис. 1).
Запуск: pytest -v   (или python -m unittest -v)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jelinski_moranda import compute  # noqa: E402

VARIANTS = {
    1: [9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1, 9, 4, 1, 3, 3, 6, 1, 1, 11, 33, 7, 91, 2],
    2: [7, 10, 12, 6, 7, 3, 5, 9, 5, 7, 2, 6, 1, 8, 3, 1, 2, 3, 5, 1, 84, 30, 7, 3, 1],
    3: [5, 4, 11, 13, 6, 2, 7, 5, 8, 7, 1, 4, 2, 7, 6, 2, 3, 1, 4, 78, 25, 10, 7, 16, 3, 1, 2],
    4: [5, 8, 12, 7, 6, 4, 3, 7, 8, 5, 2, 9, 3, 6, 5, 2, 4, 3, 77, 2, 9, 8, 10, 1, 5, 3, 4, 2],
    5: [4, 13, 10, 5, 8, 1, 6, 7, 4, 9, 5, 2, 3, 8, 6, 3, 2, 3, 94, 5, 12, 8, 28, 3],
}


class TestAllVariants(unittest.TestCase):
    """Свойства решения, которые должны выполняться для любого варианта."""

    def test_newton_converges(self):
        for v, X in VARIANTS.items():
            with self.subTest(variant=v):
                self.assertTrue(compute(X)["converged"])

    def test_K_positive(self):
        for v, X in VARIANTS.items():
            with self.subTest(variant=v):
                self.assertGreater(compute(X)["K_hat"], 0)

    def test_B_greater_than_n(self):
        for v, X in VARIANTS.items():
            with self.subTest(variant=v):
                res = compute(X)
                self.assertGreater(res["B_hat"], res["n"])


class TestVariant2(unittest.TestCase):
    """Вариант команды №2: сверка с результатами из отчёта."""

    def setUp(self):
        self.res = compute(VARIANTS[2])

    def test_n(self):
        self.assertEqual(self.res["n"], 25)

    def test_B_hat(self):
        self.assertAlmostEqual(self.res["B_hat"], 35.016253, places=5)

    def test_K_hat(self):
        self.assertAlmostEqual(self.res["K_hat"], 5.336514e-03, delta=1e-8)

    def test_X_next(self):
        self.assertAlmostEqual(self.res["X_next"], 18.708416, places=5)

    def test_time_to_finish(self):
        self.assertAlmostEqual(self.res["time_to_finish"], 548.854174, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
