from __future__ import annotations

import unittest

from causal_unlearning.config import DataConfig, RunConfig, UnlearningConfig, parse_float_list


class ConfigTests(unittest.TestCase):
    def test_parse_float_list(self) -> None:
        self.assertEqual(parse_float_list("0.0, 0.5,1"), (0.0, 0.5, 1.0))

    def test_parse_float_list_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            parse_float_list("")

    def test_run_config_serializes_lambda_values(self) -> None:
        config = RunConfig(data=DataConfig(download=True))
        payload = config.to_dict()
        self.assertEqual(payload["lambda_ce_values"], [0.0, 0.1, 0.5, 1.0])
        self.assertTrue(payload["data"]["download"])

    def test_unlearning_config_runs_base_validation(self) -> None:
        config = UnlearningConfig(epochs=2, lr=5e-4, weight_decay=0.0, lambda_ce=0.5, lambda_locality=1e-3)
        self.assertEqual(config.epochs, 2)
        self.assertEqual(config.lambda_ce, 0.5)


if __name__ == "__main__":
    unittest.main()
