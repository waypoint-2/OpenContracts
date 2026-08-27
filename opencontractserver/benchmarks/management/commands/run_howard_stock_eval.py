"""Run the Howard 15-question stock OpenContracts corpus-agent evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from opencontractserver.benchmarks.howard_scoring import (
    score_howard_stock_eval,
    write_howard_score_report,
)
from opencontractserver.benchmarks.adapters.howard_contract_suite import (
    DEFAULT_FIXTURE_ROOT,
    HowardQuestion,
    load_howard_contract_suite,
)
from opencontractserver.corpuses.models import Corpus
from opencontractserver.corpuses.services import CorpusDocumentService
from opencontractserver.llms import agents


class Command(BaseCommand):
    help = "Run and optionally score the Howard Expedia/TRX stock corpus-agent eval."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--corpus-id", type=int, required=True)
        parser.add_argument("--user", required=True)
        parser.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_FIXTURE_ROOT,
            help="Howard fixture root containing questions.json and source_manifest.json.",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=None,
            help="Answer report path. Defaults under ./benchmark_runs/.",
        )
        parser.add_argument(
            "--model",
            default=None,
            help="Optional model override accepted by the corpus agent.",
        )
        parser.add_argument(
            "--similarity-top-k",
            type=int,
            default=20,
            help="Per-tool retrieval budget for similarity search.",
        )
        parser.add_argument(
            "--sleep-between-questions",
            type=float,
            default=5.0,
            help="Seconds to sleep between questions to reduce provider rate limits.",
        )
        parser.add_argument(
            "--no-score",
            action="store_true",
            help="Skip deterministic scoring after writing the answer report.",
        )

    def handle(self, *args, **options) -> None:
        asyncio.run(self._handle_async(options))

    async def _handle_async(self, options: dict[str, Any]) -> None:
        suite = load_howard_contract_suite(options["root"])
        User = get_user_model()
        try:
            user = await User.objects.aget(username=options["user"])
        except User.DoesNotExist as exc:
            raise CommandError(f"User {options['user']!r} not found") from exc

        output_path = options["output"] or _default_answer_report_path(
            corpus_id=options["corpus_id"]
        )
        corpus_documents = await _load_corpus_document_descriptions(
            corpus_id=options["corpus_id"],
            user=user,
        )
        results: list[dict[str, Any]] = []

        for index, question in enumerate(suite.questions, start=1):
            self.stdout.write(
                f"QUESTION_START {index}/{len(suite.questions)} {question.question_id}"
            )
            agent_kwargs: dict[str, Any] = {
                "corpus": options["corpus_id"],
                "user_id": user.id,
                "streaming": False,
                "persist": False,
                "temperature": 0,
                "similarity_top_k": options["similarity_top_k"],
            }
            if options["model"]:
                agent_kwargs["model"] = options["model"]
            agent = await agents.for_corpus(**agent_kwargs)
            response = await agent.chat(
                _build_question_prompt(
                    question,
                    corpus_documents=corpus_documents,
                )
            )
            results.append(
                {
                    "index": index,
                    "id": question.question_id,
                    "question": question.question,
                    "scope": question.scope,
                    "expected_ground_truth_ids": list(
                        question.expected_ground_truth_ids
                    ),
                    "answer": response.content,
                    "sources": [_serialize_source(source) for source in response.sources],
                    "metadata": response.metadata,
                }
            )
            self.stdout.write(
                f"QUESTION_DONE {index}/{len(suite.questions)} "
                f"{question.question_id} sources={len(response.sources)} "
                f"chars={len(response.content)}"
            )
            if index < len(suite.questions) and options["sleep_between_questions"] > 0:
                time.sleep(options["sleep_between_questions"])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "corpus_id": options["corpus_id"],
                    "question_count": len(suite.questions),
                    "similarity_top_k": options["similarity_top_k"],
                    "started_at": timezone.now().isoformat(),
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.stdout.write(f"ANSWER_REPORT {output_path}")

        if not options["no_score"]:
            score_report = score_howard_stock_eval(
                output_path,
                fixture_root=options["root"],
            )
            score_path = output_path.with_name(f"{output_path.stem}_score.json")
            write_howard_score_report(score_report, score_path)
            self.stdout.write(f"SCORE_REPORT {score_path}")
            self.stdout.write(
                f"FINAL_PASS {score_report['aggregates']['passes_single_run_thresholds']}"
            )


def _build_question_prompt(
    question: HowardQuestion,
    *,
    corpus_documents: list[dict[str, Any]],
) -> str:
    documents = "\n".join(
        _format_document_line(document) for document in corpus_documents
    )
    cross_document_instruction = (
        "This is a cross-document question. Search or inspect each relevant "
        "document separately before answering, and explain which document "
        "controls when the documents interact."
        if question.requires_cross_document_reasoning
        else "This is a single-document question. Still verify the source document title, page, and annotation before answering."
    )
    topic_instruction = _topic_instruction(question)
    return (
        "Answer this Howard viability evaluation question using only stored "
        "OpenContracts corpus text and tool-returned citations.\n\n"
        "Corpus documents:\n"
        f"{documents}\n\n"
        "Required answer headings, exactly once each: Claim, Source document, "
        "Source location/page/annotation, Supporting text, Confidence, "
        "Conflicts, Unknowns.\n\n"
        "Rules:\n"
        "- Cite the exact source document title or fixture key for every "
        "substantive claim.\n"
        "- Include page and annotation IDs when the tools provide them.\n"
        "- If a retrieved source lacks a document title, use the corpus document "
        "list and cited text to identify it; do another exact-text search if "
        "needed.\n"
        "- For list, date, pricing, termination, or deadline questions, enumerate "
        "all matching items found in the retrieved text before finalizing.\n"
        "- If values are redacted with asterisks or confidential-treatment "
        "placeholders, state the unknowns explicitly and do not infer them.\n"
        "- Do not claim a document is unspecified merely because the source object "
        "omits a display title; resolve the title from the document context.\n\n"
        f"{cross_document_instruction}\n"
        f"{topic_instruction}\n\n"
        f"Question ID: {question.question_id}\n"
        f"Question: {question.question}"
    )


def _topic_instruction(question: HowardQuestion) -> str:
    text = question.question.lower()
    if "parties" in text or "trx entities" in text:
        return (
            "Party questions must distinguish the MSA parties from the SOW and "
            "Amendment parties; do not apply TRX Germany GmbH to the MSA unless "
            "the MSA text itself says so."
        )
    if "pricing" in text or "charge adjustments" in text:
        return (
            "Pricing questions must check pricing location, payment-term cross "
            "reference, pricing effective date, annual adjustment mechanics, "
            "English/German interpretation, and redactions."
        )
    if "deployment deadlines" in text:
        return (
            "Deadline questions must enumerate every deployment deadline in the "
            "relevant section, including technology, staffing, and software dates."
        )
    if "controlling relationship" in text or "control" in text:
        return (
            "Control questions must address SOW-vs-MSA precedence, prior-SOW "
            "supersession, Amendment No. 2 replacement of Section 1.1, and "
            "continuing effect except as amended."
        )
    return "Use exact-text searches for key clauses before finalizing the answer."


def _serialize_source(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


async def _load_corpus_document_descriptions(
    *,
    corpus_id: int,
    user,
) -> list[dict[str, Any]]:
    def _load() -> list[dict[str, Any]]:
        corpus = Corpus.objects.get(id=corpus_id)
        documents = CorpusDocumentService.get_corpus_documents(user=user, corpus=corpus)
        return [
            {
                "id": document.id,
                "title": document.title,
                "fixture_key": (document.custom_meta or {}).get("howard_fixture_key"),
            }
            for document in documents
        ]

    return await sync_to_async(_load)()


def _format_document_line(document: dict[str, Any]) -> str:
    key = document.get("fixture_key") or "unknown_fixture"
    return f"- document_id={document['id']} {key}: {document['title']}"


def _default_answer_report_path(*, corpus_id: int) -> Path:
    timestamp = timezone.now().strftime("%Y%m%d-%H%M%S")
    return Path("benchmark_runs") / f"{timestamp}_howard_stock_corpus_{corpus_id}" / "answers.json"
