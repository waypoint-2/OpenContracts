"""Howard viability fixture helpers.

The Howard spike needs a benchmark shape that is deliberately more operational
than LegalBench-RAG: public source documents, local PDF fixtures, curated
ground truth, and fixed questions that exercise cross-document control terms.
This module only reads and validates that on-disk package. Database ingestion
and model execution live in the management command.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3] / "fixtures/benchmarks/howard_expedia_trx"
)


@dataclass(frozen=True)
class HowardFixtureDocument:
    key: str
    title: str
    filename: str
    source_url: str
    expected_material_terms: tuple[str, ...]


@dataclass(frozen=True)
class HowardQuestion:
    question_id: str
    question: str
    scope: str
    expected_ground_truth_ids: tuple[str, ...]
    requires_cross_document_reasoning: bool


@dataclass(frozen=True)
class HowardContractSuite:
    root: Path
    corpus_dir: Path
    documents: tuple[HowardFixtureDocument, ...]
    questions: tuple[HowardQuestion, ...]
    ground_truth: dict[str, Any]
    manifest: dict[str, Any]

    @property
    def missing_pdf_fixtures(self) -> tuple[Path, ...]:
        missing = []
        for document in self.documents:
            path = self.corpus_dir / document.filename
            if not path.is_file():
                missing.append(path)
        return tuple(missing)


def load_howard_contract_suite(root: Path | str | None = None) -> HowardContractSuite:
    """Load the Howard Expedia/TRX fixture package from disk."""
    fixture_root = Path(root or DEFAULT_FIXTURE_ROOT).expanduser().resolve()
    manifest = _read_json_mapping(fixture_root / "source_manifest.json")
    ground_truth = _read_json_mapping(fixture_root / "ground_truth.json")
    questions_data = _read_json_mapping(fixture_root / "questions.json")

    corpus_dir = fixture_root / "corpus"
    documents = tuple(
        HowardFixtureDocument(
            key=str(item["key"]),
            title=str(item["title"]),
            filename=str(item["filename"]),
            source_url=str(item["source_url"]),
            expected_material_terms=tuple(item.get("expected_material_terms", ())),
        )
        for item in manifest.get("documents", ())
    )
    if not documents:
        raise ValueError(f"{fixture_root / 'source_manifest.json'} has no documents")

    questions = tuple(
        HowardQuestion(
            question_id=str(item["id"]),
            question=str(item["question"]),
            scope=str(item["scope"]),
            expected_ground_truth_ids=tuple(item.get("expected_ground_truth_ids", ())),
            requires_cross_document_reasoning=bool(
                item.get("requires_cross_document_reasoning", False)
            ),
        )
        for item in questions_data.get("questions", ())
    )
    if not questions:
        raise ValueError(f"{fixture_root / 'questions.json'} has no questions")

    return HowardContractSuite(
        root=fixture_root,
        corpus_dir=corpus_dir,
        documents=documents,
        questions=questions,
        ground_truth=ground_truth,
        manifest=manifest,
    )


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Howard fixture file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Howard fixture file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Howard fixture file must contain a JSON object: {path}")
    return data
