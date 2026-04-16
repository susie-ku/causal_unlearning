from __future__ import annotations

import unittest

from causal_unlearning.cli import build_parser


class CliParserTests(unittest.TestCase):
    def test_run_parser_accepts_lambda_sweep(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--lambda-ce-values", "0.0,0.2,1.0"])
        self.assertEqual(args.lambda_ce_values, "0.0,0.2,1.0")

    def test_train_parser_reads_world(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train", "--world", "observational"])
        self.assertEqual(args.world, "observational")


if __name__ == "__main__":
    unittest.main()

