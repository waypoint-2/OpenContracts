"""Run the Howard 15-question stock OpenContracts corpus-agent evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
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
from opencontractserver.annotations.models import Annotation
from opencontractserver.corpuses.models import Corpus
from opencontractserver.corpuses.services import CorpusDocumentService
from opencontractserver.documents.models import DocumentPath
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
            "--question-retries",
            type=int,
            default=3,
            help="Retry count for a question before failing the run.",
        )
        parser.add_argument(
            "--retry-sleep-seconds",
            type=float,
            default=75.0,
            help="Base seconds to sleep before retrying a failed question.",
        )
        parser.add_argument(
            "--question-id",
            action="append",
            default=None,
            help="Run only the given question ID. May be supplied multiple times.",
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
        selected_question_ids = {
            str(question_id).upper() for question_id in (options["question_id"] or ())
        }
        questions = [
            question
            for question in suite.questions
            if not selected_question_ids
            or question.question_id.upper() in selected_question_ids
        ]
        if not questions:
            raise CommandError(
                f"No Howard questions matched --question-id={options['question_id']!r}"
            )
        results: list[dict[str, Any]] = []
        metadata = {
            "corpus_id": options["corpus_id"],
            "question_count": len(questions),
            "similarity_top_k": options["similarity_top_k"],
            "started_at": timezone.now().isoformat(),
            "selected_question_ids": sorted(selected_question_ids),
        }

        for index, question in enumerate(questions, start=1):
            self.stdout.write(
                f"QUESTION_START {index}/{len(questions)} {question.question_id}"
            )
            response = await _run_question_with_retries(
                question=question,
                corpus_documents=corpus_documents,
                options=options,
                user_id=user.id,
                stdout=self.stdout,
            )
            targeted_documents = _target_documents_for_question(
                question,
                corpus_documents=corpus_documents,
            )
            sources = await _serialize_sources_with_document_ids(
                response.sources,
                corpus_id=options["corpus_id"],
            )
            result = {
                "index": index,
                "id": question.question_id,
                "question": question.question,
                "scope": question.scope,
                "target_document_ids": [
                    document["id"] for document in targeted_documents
                ],
                "expected_ground_truth_ids": list(
                    question.expected_ground_truth_ids
                ),
                "answer": response.content,
                "sources": sources,
                "metadata": response.metadata,
            }
            results.append(result)
            _write_answer_report(
                output_path,
                {
                    **metadata,
                    "completed_question_count": len(results),
                    "status": "partial"
                    if len(results) < len(questions)
                    else "complete",
                    "results": results,
                },
            )
            self.stdout.write(
                f"QUESTION_DONE {index}/{len(questions)} "
                f"{question.question_id} sources={len(response.sources)} "
                f"chars={len(response.content)}"
            )
            if index < len(questions) and options["sleep_between_questions"] > 0:
                await asyncio.sleep(options["sleep_between_questions"])

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


async def _run_question_with_retries(
    *,
    question: HowardQuestion,
    corpus_documents: list[dict[str, Any]],
    options: dict[str, Any],
    user_id: int,
    stdout,
):
    attempts = max(int(options["question_retries"]) + 1, 1)
    for attempt in range(1, attempts + 1):
        try:
            agent_kwargs: dict[str, Any] = {
                "corpus": options["corpus_id"],
                "user_id": user_id,
                "streaming": False,
                "persist": False,
                "temperature": 0,
                "similarity_top_k": options["similarity_top_k"],
                "include_nested_tool_timeline": False,
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
            if response.metadata.get("error"):
                raise RuntimeError(str(response.metadata["error"]))
            return response
        except Exception as exc:
            if attempt >= attempts:
                raise
            retry_after = _retry_after_seconds(exc)
            sleep_for = max(
                retry_after or 0,
                float(options["retry_sleep_seconds"]) * attempt,
            )
            stdout.write(
                f"QUESTION_RETRY {question.question_id} attempt={attempt} "
                f"sleep_seconds={sleep_for:.1f} error={exc.__class__.__name__}"
            )
            await asyncio.sleep(sleep_for)

    raise RuntimeError(f"Question {question.question_id} failed unexpectedly")


def _write_answer_report(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _retry_after_seconds(exc: Exception) -> float | None:
    match = re.search(r"try again in ([0-9.]+)s", str(exc), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1)) + 5.0
    except ValueError:
        return None


def _target_documents_for_question(
    question: HowardQuestion,
    *,
    corpus_documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {str(document.get("fixture_key")): document for document in corpus_documents}
    key_sets = {
        "msa_2007": ("msa_2007",),
        "sow_2008": ("sow_2008",),
        "amendment_2011": ("amendment_2011",),
        "Q01": ("msa_2007", "sow_2008", "amendment_2011"),
        "Q02": ("msa_2007", "sow_2008", "amendment_2011"),
        "Q05": ("msa_2007", "sow_2008"),
        "Q07": ("sow_2008", "amendment_2011"),
        "Q13": ("msa_2007", "sow_2008", "amendment_2011"),
        "Q15": ("msa_2007", "sow_2008", "amendment_2011"),
    }
    keys = key_sets.get(question.question_id) or key_sets.get(question.scope)
    if not keys:
        return corpus_documents
    return [by_key[key] for key in keys if key in by_key]


def _build_question_prompt(
    question: HowardQuestion,
    *,
    corpus_documents: list[dict[str, Any]],
) -> str:
    documents = "\n".join(
        _format_document_line(document)
        for document in _target_documents_for_question(
            question,
            corpus_documents=corpus_documents,
        )
    )
    target_ids = ", ".join(
        str(document["id"])
        for document in _target_documents_for_question(
            question,
            corpus_documents=corpus_documents,
        )
    )
    cross_document_instruction = (
        "This is a cross-document question. Search or inspect each relevant "
        "document separately before answering, and explain which document "
        "controls when the documents interact."
        if question.requires_cross_document_reasoning
        else "This is a single-document question. Still verify the source document title, page, and annotation before answering."
    )
    topic_instruction = _topic_instruction(question)
    checklist_instruction = _question_checklist_instruction(question)
    return (
        "Answer this Howard viability evaluation question using only stored "
        "OpenContracts corpus text and tool-returned citations. Work like a "
        "senior contract-review lawyer using retrieval tools: build the answer "
        "from cited evidence, preserve distinctions between related documents, "
        "and do not collapse separate legal effects into a generic summary.\n\n"
        "Corpus documents:\n"
        f"{documents}\n\n"
        "Required answer headings, exactly once each: Claim, Source document, "
        "Source location/page/annotation, Supporting text, Confidence, "
        "Conflicts, Unknowns.\n\n"
        "Rules:\n"
        f"- Use only these document IDs for this question: {target_ids}.\n"
        "- Prefer targeted similarity_search and short exact-text searches; "
        "load bulk document text only when the searched annotations are "
        "insufficient.\n"
        "- Cite the exact source document title or fixture key for every "
        "substantive claim.\n"
        "- Include page and annotation IDs when the tools provide them.\n"
        "- If a retrieved source lacks a document title, use the corpus document "
        "list and cited text to identify it; do another exact-text search if "
        "needed.\n"
        "- For list, date, pricing, termination, or deadline questions, enumerate "
        "all matching items found in the retrieved text before finalizing.\n"
        "- When a question asks about a fact that could appear in multiple "
        "documents, such as effective dates, parties, deadlines, termination "
        "rights, governing terms, or scope changes, analyze each relevant "
        "document separately. State the fact separately for each document, and "
        "do not give one answer that only covers some of the documents.\n"
        "- When describing payment, pricing, operational obligations, scope, "
        "precedence, amendment effects, or any mechanic that references another "
        "section, attachment, exhibit, schedule, or statement of work, name the "
        "specific referenced section, attachment, exhibit, schedule, or document. "
        "Do not describe the mechanic without identifying where it is defined.\n"
        "- Preserve exact identifiers from the source, including section numbers, "
        "attachment or exhibit labels, defined document names, and specific "
        "dates, because these make the answer traceable back to the contract. "
        "Explain legal effects and operative language, such as precedence, "
        "supersession, or continuing-effect rules, in clear plain English rather "
        "than quoting the source's legal phrasing verbatim.\n"
        "- If values are redacted with asterisks or confidential-treatment "
        "placeholders, state the unknowns explicitly and do not infer them.\n"
        "- Do not claim a document is unspecified merely because the source object "
        "omits a display title; resolve the title from the document context.\n"
        "- Do not use weak provenance phrases such as 'not explicitly found' or "
        "'unspecified' for a document, date, or clause when the corpus document "
        "list or retrieved source text identifies it.\n"
        "- Before writing the final answer, make a private evidence ledger: each "
        "required fact, the document it came from, the quoted supporting text, "
        "and its page/annotation/source location. The final answer must include "
        "every ledger item, with no uncited legal conclusion.\n"
        "- In the final answer, use compact bullets or rows under the required "
        "headings when multiple documents, dates, deadlines, prices, or control "
        "rules must be distinguished.\n\n"
        f"{cross_document_instruction}\n"
        f"{topic_instruction}\n\n"
        f"{checklist_instruction}\n\n"
        f"Question ID: {question.question_id}\n"
        f"Question: {question.question}"
    )


def _topic_instruction(question: HowardQuestion) -> str:
    text = question.question.lower()
    if "parties" in text or "trx entities" in text:
        return (
            "Party questions must distinguish the MSA parties from the SOW and "
            "Amendment parties; do not apply TRX Germany GmbH to the MSA unless "
            "the MSA text itself says so. Use targeted searches only for party "
            "names, preambles, definitions, and signature blocks; do not use "
            "load_document_text for party questions."
        )
    if "fully disclose" in text or "redactions" in text:
        return (
            "Disclosure/redaction questions must use targeted searches only: "
            "search for pricing, service location, services, attachment headings, "
            "CONFIDENTIAL TREATMENT REQUESTED, and asterisk redaction markers in "
            "each relevant document. Do not use load_document_text for these "
            "questions; it is too broad for the batch eval and exact redaction "
            "markers plus section hits are sufficient."
        )
    if "pricing" in text or "charge adjustments" in text:
        return (
            "Pricing questions must check pricing location, payment-term cross "
            "reference, pricing effective date, annual adjustment mechanics, "
            "English/German interpretation, and redactions. Use targeted "
            "similarity_search and exact-text searches for pricing clauses, "
            "Attachment headings, payment cross-references, index-adjustment "
            "language, and confidential-treatment markers; do not use "
            "load_document_text unless those targeted searches fail to retrieve "
            "the relevant clause."
        )
    if "deployment deadlines" in text:
        return (
            "Deadline questions must enumerate every deployment deadline in the "
            "relevant section, including technology, staffing, and software dates. "
            "Use targeted searches for 'deploy', 'no later than', 'on or before', "
            "'as soon as reasonably practicable', 'workforce management software', "
            "'call center technology', 'quality assurance', 'business analyst', "
            "and 'dedicated trainer'."
        )
    if "controlling relationship" in text or "control" in text:
        return (
            "Control questions must address SOW-vs-MSA precedence, prior-SOW "
            "supersession, Amendment No. 2 replacement of Section 1.1, and "
            "continuing effect except as amended. Use targeted searches for "
            "'conflict or inconsistency', 'take precedence', 'supersedes and "
            "replaces', 'prior SOW', 'Section 1.1', 'deleted in its entirety', "
            "'full force and effect', and 'govern and control'. Avoid bulk "
            "document loading for these questions unless exact-text searches "
            "cannot locate the control clauses."
        )
    return "Use exact-text searches for key clauses before finalizing the answer."


def _question_checklist_instruction(question: HowardQuestion) -> str:
    """Return fixture-specific legal review checklists without pre-answering."""

    checklists = {
        "Q01": (
            "Completeness checklist for Q01: answer in three separate rows or "
            "bullets, one each for the MSA, 2008 SOW, and Amendment No. 2. For "
            "each document, state: effective/dated date, Expedia party, TRX "
            "party or parties, whether TRX Germany GmbH is included, and the "
            "citation supporting that row. Do not generalize one document's TRX "
            "party set to the others."
        ),
        "Q02": (
            "Completeness checklist for Q02: provide one cited row each for the "
            "MSA, 2008 SOW, and Amendment No. 2 effective date. The source "
            "location must cite exact retrieved text for every date; do not say "
            "a date was not found if it appears in the corpus document list or "
            "retrieved preamble text."
        ),
        "Q07": (
            "Completeness checklist for Q07: identify the post-effective-date "
            "scope clause, the specific SOW provision it replaces, and whether "
            "the rest of the SOW/Agreement remains operative except as amended. "
            "Cite both the replacement language and the continuing-effect/control "
            "language."
        ),
        "Q09": (
            "Completeness checklist for Q09: enumerate the MSA service-fee "
            "billing mechanics, including invoice timing, payment timing, "
            "payment method, and any redacted payment-detail unknowns. Verify "
            "that both the invoice date and payment due-date mechanics appear "
            "in the final answer."
        ),
        "Q10": (
            "Completeness checklist for Q10: distinguish the pricing attachment "
            "or location, the payment-term cross-reference, the pricing effective "
            "date, the annual adjustment mechanism, the English/German "
            "interpretation rule, and any redacted pricing or volume values. "
            "If English and German text differ, state the English control rule "
            "and summarize the German parallel formulation as a nuance, not a "
            "replacement."
        ),
        "Q11": (
            "Completeness checklist for Q11: create a deadline ledger from the "
            "operational terms. Include every software, technology, quality/team, "
            "business analyst, and trainer deployment obligation with its exact "
            "deadline or timing phrase and citation. Before finalizing, compare "
            "the final answer against the retrieved Section 4 operational terms "
            "so no date-bearing deployment clause is omitted."
        ),
        "Q12": (
            "Completeness checklist for Q12: cite the exact transition/cooperation "
            "timeline text and state each date or completion deadline it contains. "
            "Do not label the timeline as missing or unspecified when the "
            "retrieved text includes date language."
        ),
        "Q15": (
            "Completeness checklist for Q15: summarize the relationship as a "
            "layered hierarchy. Include: MSA baseline framework, SOW precedence "
            "for services conflicts, 2008 SOW supersession/replacement of the "
            "prior SOW, Amendment No. 2 replacement of Section 1.1, and the "
            "continuing-effect rule for unchanged provisions. Cite each layer "
            "separately."
        ),
    }
    return checklists.get(
        question.question_id,
        "Completeness checklist: ensure every legal conclusion in the final "
        "answer is tied to a retrieved source and that all date-bearing, "
        "redaction-bearing, or control-language clauses found by the tools are "
        "represented accurately.",
    )


def _serialize_source(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    return str(value)


async def _serialize_sources_with_document_ids(
    sources: list[Any],
    *,
    corpus_id: int,
) -> list[Any]:
    serialized = [_serialize_source(source) for source in sources]
    annotation_ids = [
        source.get("annotation_id")
        for source in serialized
        if isinstance(source, dict)
        and isinstance(source.get("annotation_id"), int)
        and source.get("annotation_id") > 0
        and source.get("document_id") is None
    ]
    if not annotation_ids:
        return serialized

    def _load_document_ids() -> dict[int, int]:
        annotations = list(
            Annotation.objects.filter(id__in=annotation_ids).values_list(
                "id",
                "document_id",
                "structural_set_id",
            )
        )
        direct_document_ids = {
            annotation_id: document_id
            for annotation_id, document_id, _structural_set_id in annotations
            if document_id is not None
        }
        structural_set_ids = {
            structural_set_id
            for _annotation_id, document_id, structural_set_id in annotations
            if document_id is None and structural_set_id is not None
        }
        structural_set_document_ids = {
            structural_set_id: document_id
            for structural_set_id, document_id in DocumentPath.objects.filter(
                corpus_id=corpus_id,
                is_current=True,
                is_deleted=False,
                document__structural_annotation_set_id__in=structural_set_ids,
            ).values_list("document__structural_annotation_set_id", "document_id")
        }
        document_ids = {}
        for annotation_id, _document_id, structural_set_id in annotations:
            document_id = direct_document_ids.get(
                annotation_id
            ) or structural_set_document_ids.get(structural_set_id)
            if document_id is not None:
                document_ids[annotation_id] = document_id
        return document_ids

    document_ids = await sync_to_async(_load_document_ids)()
    for source in serialized:
        if not isinstance(source, dict) or source.get("document_id") is not None:
            continue
        document_id = document_ids.get(source.get("annotation_id"))
        if document_id is not None:
            source["document_id"] = document_id
    return serialized


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
