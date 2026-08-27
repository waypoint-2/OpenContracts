"""Deterministic scoring for the Howard Expedia/TRX stock-agent evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opencontractserver.benchmarks.adapters.howard_contract_suite import (
    HowardContractSuite,
    load_howard_contract_suite,
)


@dataclass(frozen=True)
class HowardClaimRule:
    """Fixture-specific requirements for one audited ground-truth claim."""

    ground_truth_id: str
    address_any: tuple[str, ...]
    correct_all: tuple[str, ...]
    source_any: tuple[str, ...] = ()
    forbidden_any: tuple[str, ...] = ()


HOWARD_CLAIM_RULES: dict[str, HowardClaimRule] = {
    "GT_PARTIES_MSA": HowardClaimRule(
        ground_truth_id="GT_PARTIES_MSA",
        address_any=("master services agreement", "msa"),
        correct_all=("expedia, inc", "trx, inc", "washington", "georgia"),
        source_any=("master services agreement", "january 1, 2007"),
        forbidden_any=(
            "these documents bind trx, inc. and trx germany",
            "documents bind trx, inc. and trx germany",
            "msa binds trx germany",
        ),
    ),
    "GT_PARTIES_SOW": HowardClaimRule(
        ground_truth_id="GT_PARTIES_SOW",
        address_any=("statement of work", "sow"),
        correct_all=("expedia, inc", "trx, inc", "trx germany gmbh"),
        source_any=("dated and effective january 1, 2008", "trx germany gmbh"),
    ),
    "GT_PARTIES_AMENDMENT": HowardClaimRule(
        ground_truth_id="GT_PARTIES_AMENDMENT",
        address_any=("amendment no. 2", "amendment"),
        correct_all=("march 1, 2011", "expedia, inc", "trx, inc", "trx germany gmbh"),
        source_any=("dated and effective as of march 1, 2011", "trx germany gmbh"),
    ),
    "GT_EFFECTIVE_DATES": HowardClaimRule(
        ground_truth_id="GT_EFFECTIVE_DATES",
        address_any=("effective date", "effective dates"),
        correct_all=("january 1, 2007", "january 1, 2008", "march 1, 2011"),
        source_any=("january 1, 2007", "january 1, 2008", "march 1, 2011"),
    ),
    "GT_MSA_TERM_RENEWAL": HowardClaimRule(
        ground_truth_id="GT_MSA_TERM_RENEWAL",
        address_any=("initial term", "renewal term", "automatically renews"),
        correct_all=(
            "december 31, 2010",
            "article 11",
            "one-year||one (1) year||one year",
            "12 months||twelve (12) months||twelve months",
        ),
        source_any=("article 2", "december 31, 2010", "twelve"),
    ),
    "GT_MSA_SOW_PRECEDENCE": HowardClaimRule(
        ground_truth_id="GT_MSA_SOW_PRECEDENCE",
        address_any=("conflict", "inconsistency", "precedence", "take precedence"),
        correct_all=("statement of work", "take precedence", "services"),
        source_any=("take precedence",),
    ),
    "GT_SOW_PRIOR_SUPERSESSION": HowardClaimRule(
        ground_truth_id="GT_SOW_PRIOR_SUPERSESSION",
        address_any=("supersedes", "replaces", "prior sow"),
        correct_all=("january 1, 2007", "prior sow"),
        source_any=("previously entered into a statement of work, dated january 1, 2007",),
    ),
    "GT_SOW_TERMINATION_RIGHTS": HowardClaimRule(
        ground_truth_id="GT_SOW_TERMINATION_RIGHTS",
        address_any=("terminate", "termination", "good cause"),
        correct_all=("art. 627", "german civil code", "six", "prior written notice"),
        source_any=("art. 627 of the german civil code", "six (6) months"),
    ),
    "GT_MSA_PAYMENT_MECHANICS": HowardClaimRule(
        ground_truth_id="GT_MSA_PAYMENT_MECHANICS",
        address_any=("prepay", "invoice", "electronic funds transfer"),
        correct_all=("last business day", "15th day", "electronic funds transfer", "redacted"),
        source_any=("article 7", "last business day", "15th"),
    ),
    "GT_SOW_PRICING": HowardClaimRule(
        ground_truth_id="GT_SOW_PRICING",
        address_any=("pricing", "charges", "german price index"),
        correct_all=(
            "attachment 3",
            "section 7",
            "january 1, 2008",
            "german price index",
            "april",
            "redacted",
        ),
        source_any=("attachment 3", "pricing set forth in this statement of work", "german price index"),
    ),
    "GT_SOW_OPERATIONAL_DEADLINES": HowardClaimRule(
        ground_truth_id="GT_SOW_OPERATIONAL_DEADLINES",
        address_any=("deadline", "deploy", "deployment", "operational"),
        correct_all=("march 31, 2008", "july 1, 2008", "february 29, 2008"),
        source_any=("march 31, 2008", "july 1, 2008", "february 29, 2008"),
    ),
    "GT_SOW_LANGUAGE_PRECEDENCE": HowardClaimRule(
        ground_truth_id="GT_SOW_LANGUAGE_PRECEDENCE",
        address_any=("english", "german", "language"),
        correct_all=("english", "prevail"),
        source_any=("english version shall prevail",),
    ),
    "GT_AMENDMENT_TRANSITION_TIMELINE": HowardClaimRule(
        ground_truth_id="GT_AMENDMENT_TRANSITION_TIMELINE",
        address_any=("cooperate", "assist", "commence", "transition"),
        correct_all=("april 25, 2011", "october 31, 2011||31 october 2011"),
        source_any=("april 25, 2011", "31 october 2011||october 31, 2011"),
    ),
    "GT_AMENDMENT_REPLACES_SECTION_1_1": HowardClaimRule(
        ground_truth_id="GT_AMENDMENT_REPLACES_SECTION_1_1",
        address_any=("section 1.1", "scope", "deleted", "replaced"),
        correct_all=(
            "section 1.1",
            "october 31, 2011||31 october 2011",
            "deleted",
            "replaced",
        ),
        source_any=("effective 31 october 2011||effective october 31, 2011", "section 1.1"),
    ),
    "GT_AMENDMENT_CONTINUING_EFFECT": HowardClaimRule(
        ground_truth_id="GT_AMENDMENT_CONTINUING_EFFECT",
        address_any=("full force and effect", "govern and control", "continuing"),
        correct_all=("full force and effect", "agreement", "govern and control"),
        source_any=("full force and effect", "govern and control this amendment"),
    ),
    "GT_REDACTIONS_UNKNOWN": HowardClaimRule(
        ground_truth_id="GT_REDACTIONS_UNKNOWN",
        address_any=("redacted", "confidential treatment", "unknown"),
        correct_all=("prices", "volumes", "service locations", "services", "redacted"),
        source_any=("confidential treatment requested", "*"),
    ),
}


WEAK_PROVENANCE_PATTERNS = (
    "not explicitly found",
    "not explicitly named",
    "unspecified",
    "no specific document id",
    "no specific document",
    "not provided",
)


def score_howard_stock_eval(
    answer_report_path: Path | str,
    *,
    fixture_root: Path | str | None = None,
) -> dict[str, Any]:
    """Score a saved Howard stock-agent answer report against fixture truth."""

    suite = load_howard_contract_suite(fixture_root)
    answer_report = json.loads(Path(answer_report_path).read_text(encoding="utf-8"))
    result_by_id = {
        str(item["id"]): item
        for item in answer_report.get("results", answer_report.get("answers", ()))
    }
    ground_truth = {
        str(item["id"]): item for item in suite.ground_truth.get("claims", ())
    }

    rows = []
    for question in suite.questions:
        result = result_by_id.get(question.question_id)
        answer = str((result or {}).get("answer", ""))
        sources = list((result or {}).get("sources", ()) or ())
        expected_rows = []
        for gt_id in question.expected_ground_truth_ids:
            rule = HOWARD_CLAIM_RULES[gt_id]
            expected_rows.append(_score_claim(rule, answer=answer, sources=sources))

        provenance = _score_provenance(answer=answer, sources=sources)
        addressed_count = sum(1 for row in expected_rows if row["addressed"])
        correct_count = sum(1 for row in expected_rows if row["correct"])
        rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "expected_ground_truth_ids": list(question.expected_ground_truth_ids),
                "ground_truth_results": expected_rows,
                "addressed_expected_count": addressed_count,
                "correct_expected_count": correct_count,
                "expected_count": len(expected_rows),
                "all_expected_addressed": addressed_count == len(expected_rows),
                "all_expected_correct": correct_count == len(expected_rows),
                "citation_count": len(sources),
                "provenance": provenance,
                "failure_categories": _failure_categories(
                    expected_rows=expected_rows,
                    provenance=provenance,
                    answer=answer,
                ),
                "answer_excerpt": _squash(answer)[:500],
            }
        )

    thresholds = suite.ground_truth.get("acceptance_thresholds", {})
    aggregate = _aggregate(rows, thresholds=thresholds, ground_truth=ground_truth)
    return {
        "answer_report_path": str(answer_report_path),
        "fixture_root": str(suite.root),
        "corpus_id": answer_report.get("corpus_id"),
        "question_count": len(rows),
        "aggregates": aggregate,
        "questions": rows,
    }


def write_howard_score_report(report: dict[str, Any], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _score_claim(
    rule: HowardClaimRule,
    *,
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_answer = _norm(answer)
    normalized_sources = _norm(" ".join(_source_text(source) for source in sources))
    address_hits = _hits(normalized_answer, rule.address_any)
    correct_hits = _hits(normalized_answer, rule.correct_all)
    source_hits = _hits(normalized_sources, rule.source_any)
    answer_source_hits = _hits(normalized_answer, rule.source_any)
    forbidden_hits = _hits(normalized_answer, rule.forbidden_any)

    addressed = bool(address_hits)
    source_covered = not rule.source_any or bool(source_hits or answer_source_hits)
    correct = (
        addressed
        and len(correct_hits) == len(rule.correct_all)
        and not forbidden_hits
    )
    return {
        "ground_truth_id": rule.ground_truth_id,
        "addressed": addressed,
        "correct": correct,
        "source_covered": source_covered,
        "address_hits": address_hits,
        "missing_correct_terms": [
            term for term in rule.correct_all if not _term_present(normalized_answer, term)
        ],
        "source_hits": source_hits,
        "answer_source_hits": answer_source_hits,
        "forbidden_hits": forbidden_hits,
    }


def _score_provenance(*, answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_answer = _norm(answer)
    weak_phrases = _hits(normalized_answer, WEAK_PROVENANCE_PATTERNS)
    sources_with_text = sum(1 for source in sources if _source_text(source).strip())
    sources_with_location = sum(
        1
        for source in sources
        if source.get("annotation_id") is not None or source.get("page") is not None
    )
    return {
        "citation_count": len(sources),
        "sources_with_text": sources_with_text,
        "sources_with_location": sources_with_location,
        "weak_phrases": weak_phrases,
        "citations_resolve": bool(sources) and sources_with_text == len(sources),
        "strong": bool(sources)
        and sources_with_text == len(sources)
        and sources_with_location == len(sources)
        and not weak_phrases,
    }


def _failure_categories(
    *,
    expected_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    answer: str,
) -> list[str]:
    categories = set()
    if any(not row["source_covered"] for row in expected_rows):
        categories.add("retrieval_or_reranking")
    if any(row["addressed"] and not row["correct"] for row in expected_rows):
        categories.add("model_reasoning")
    if any(not row["addressed"] for row in expected_rows):
        categories.add("model_reasoning")
    if not provenance["strong"]:
        categories.add("citation_or_provenance")
    if "october 31, 2011" in _norm(answer) and any(
        row["ground_truth_id"].startswith("GT_AMENDMENT") and not row["correct"]
        for row in expected_rows
    ):
        categories.add("cross_document_effective_term_resolution")
    return sorted(categories)


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, Any],
    ground_truth: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_total = sum(row["expected_count"] for row in rows)
    addressed_total = sum(row["addressed_expected_count"] for row in rows)
    correct_total = sum(row["correct_expected_count"] for row in rows)
    unsupported_or_incorrect = expected_total - correct_total
    strong_provenance = sum(1 for row in rows if row["provenance"]["strong"])
    all_citations_resolve = all(row["provenance"]["citations_resolve"] for row in rows)
    critical_gt_ids = {
        gt_id
        for gt_id, claim in ground_truth.items()
        if claim.get("topic") in {"parties", "effective_dates", "payment", "order_of_precedence", "supersession"}
    }
    critical_rows = [
        gt_row
        for row in rows
        for gt_row in row["ground_truth_results"]
        if gt_row["ground_truth_id"] in critical_gt_ids
    ]
    critical_correct = all(row["correct"] for row in critical_rows)
    amendment_supersession = all(
        gt_row["correct"]
        for row in rows
        for gt_row in row["ground_truth_results"]
        if gt_row["ground_truth_id"] == "GT_AMENDMENT_REPLACES_SECTION_1_1"
    )
    explicit_uncertainty = all(
        gt_row["correct"]
        for row in rows
        for gt_row in row["ground_truth_results"]
        if gt_row["ground_truth_id"] == "GT_REDACTIONS_UNKNOWN"
    )
    overall_accuracy = correct_total / expected_total if expected_total else 0.0
    return {
        "expected_ground_truth_total": expected_total,
        "addressed_expected_total": addressed_total,
        "correct_expected_total": correct_total,
        "ground_truth_address_rate": addressed_total / expected_total
        if expected_total
        else 0.0,
        "overall_answer_accuracy": overall_accuracy,
        "unsupported_or_incorrect_expected_claims": unsupported_or_incorrect,
        "questions_with_strong_provenance": strong_provenance,
        "strong_provenance_rate": strong_provenance / len(rows) if rows else 0.0,
        "citation_resolution": 1.0 if all_citations_resolve else 0.0,
        "critical_terms_accuracy": 1.0 if critical_correct else 0.0,
        "amendment_supersession_correct": amendment_supersession,
        "explicit_uncertainty_for_redactions": explicit_uncertainty,
        "passes_single_run_thresholds": (
            all_citations_resolve
            and unsupported_or_incorrect <= thresholds.get("unsupported_substantive_claims", 0)
            and critical_correct
            and overall_accuracy >= thresholds.get("overall_answer_accuracy", 0.9)
            and amendment_supersession
            and explicit_uncertainty
        ),
    }


def _hits(normalized_text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if _term_present(normalized_text, term)]


def _term_present(normalized_text: str, term: str) -> bool:
    return any(_norm(option) in normalized_text for option in term.split("||"))


def _source_text(source: dict[str, Any]) -> str:
    block_context = source.get("block_context")
    metadata = source.get("metadata")
    if not isinstance(block_context, dict) and isinstance(metadata, dict):
        block_context = metadata.get("block_context")
    return " ".join(
        str(part or "")
        for part in (
            source.get("content"),
            source.get("rawText"),
            source.get("search_string"),
            metadata.get("content") if isinstance(metadata, dict) else "",
            metadata.get("search_string") if isinstance(metadata, dict) else "",
            block_context.get("block_text") if isinstance(block_context, dict) else "",
        )
    )


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _squash(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
