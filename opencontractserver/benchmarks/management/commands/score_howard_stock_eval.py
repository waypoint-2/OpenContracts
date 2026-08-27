"""Score a saved Howard stock OpenContracts answer report."""

from __future__ import annotations

import argparse
from pathlib import Path

from django.core.management.base import BaseCommand

from opencontractserver.benchmarks.adapters.howard_contract_suite import (
    DEFAULT_FIXTURE_ROOT,
)
from opencontractserver.benchmarks.howard_scoring import (
    score_howard_stock_eval,
    write_howard_score_report,
)


class Command(BaseCommand):
    help = (
        "Score a Howard Expedia/TRX stock-agent answer JSON against the audited "
        "fixture ground truth."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "answer_report",
            type=Path,
            help="Path to stock_agent_answers_run*.json.",
        )
        parser.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_FIXTURE_ROOT,
            help="Howard fixture root containing questions.json and ground_truth.json.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Optional path for scored JSON. Defaults beside the answer report.",
        )

    def handle(self, *args, **options) -> None:
        answer_report = options["answer_report"]
        score_report = score_howard_stock_eval(
            answer_report,
            fixture_root=options["root"],
        )
        output_path = options["output"] or answer_report.with_name(
            f"{answer_report.stem}_score.json"
        )
        write_howard_score_report(score_report, output_path)

        aggregates = score_report["aggregates"]
        self.stdout.write(self.style.SUCCESS("Howard stock eval scored."))
        self.stdout.write(f"  Answer report: {answer_report}")
        self.stdout.write(f"  Score report:  {output_path}")
        self.stdout.write(
            "  Accuracy:      "
            f"{aggregates['correct_expected_total']}/"
            f"{aggregates['expected_ground_truth_total']} "
            f"({aggregates['overall_answer_accuracy']:.3f})"
        )
        self.stdout.write(
            "  Addressed:     "
            f"{aggregates['addressed_expected_total']}/"
            f"{aggregates['expected_ground_truth_total']} "
            f"({aggregates['ground_truth_address_rate']:.3f})"
        )
        self.stdout.write(
            "  Strong provenance: "
            f"{aggregates['questions_with_strong_provenance']}/"
            f"{score_report['question_count']} "
            f"({aggregates['strong_provenance_rate']:.3f})"
        )
        self.stdout.write(
            f"  PASS:          {aggregates['passes_single_run_thresholds']}"
        )

        self.stdout.write("")
        self.stdout.write("Question results:")
        for row in score_report["questions"]:
            status = "PASS" if row["all_expected_correct"] and row["provenance"]["strong"] else "FAIL"
            categories = ", ".join(row["failure_categories"]) or "-"
            self.stdout.write(
                f"  {row['question_id']}: {status} "
                f"correct={row['correct_expected_count']}/{row['expected_count']} "
                f"addressed={row['addressed_expected_count']}/{row['expected_count']} "
                f"provenance={row['provenance']['strong']} "
                f"categories={categories}"
            )
