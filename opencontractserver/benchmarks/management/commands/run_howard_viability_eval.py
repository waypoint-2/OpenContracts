"""Run the Howard OpenContracts viability ingest/readiness check."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from opencontractserver.annotations.models import Annotation, Embedding, Relationship
from opencontractserver.benchmarks.adapters.howard_contract_suite import (
    DEFAULT_FIXTURE_ROOT,
    HowardFixtureDocument,
    load_howard_contract_suite,
)
from opencontractserver.corpuses.models import Corpus
from opencontractserver.corpuses.services.corpus_documents import CorpusDocumentService
from opencontractserver.documents.document_service import DocumentService
from opencontractserver.documents.models import Document, DocumentProcessingStatus
from opencontractserver.types.enums import PermissionTypes
from opencontractserver.utils.permissioning import set_permissions_for_obj_to_user


DEFAULT_EMBEDDER_PATH = (
    "opencontractserver.pipeline.embedders.sent_transformer_microservice."
    "MicroserviceEmbedder"
)


class Command(BaseCommand):
    help = (
        "Load the Howard Expedia/TRX public-contract fixture package through the "
        "normal upload path and verify parse + embedding readiness."
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--root",
            type=Path,
            default=DEFAULT_FIXTURE_ROOT,
            help="Fixture root containing source_manifest.json, questions.json, ground_truth.json, and corpus/*.pdf.",
        )
        parser.add_argument(
            "--user",
            required=True,
            help="Username of the Django user that will own the evaluation corpus.",
        )
        parser.add_argument(
            "--corpus-title",
            default=None,
            help="Optional corpus title. Defaults to a timestamped Howard title.",
        )
        parser.add_argument(
            "--timeout-seconds",
            type=int,
            default=600,
            help="Seconds to wait for each document to finish parsing.",
        )
        parser.add_argument(
            "--embedding-timeout-seconds",
            type=int,
            default=300,
            help="Seconds to wait for each completed document's embedding.",
        )
        parser.add_argument(
            "--embedder-path",
            default=DEFAULT_EMBEDDER_PATH,
            help="Embedder path to verify with Document.get_embedding().",
        )
        parser.add_argument(
            "--dimension",
            type=int,
            default=384,
            help="Embedding dimension to verify.",
        )
        parser.add_argument(
            "--report-path",
            type=Path,
            default=None,
            help="Optional JSON report path. Defaults under ./benchmark_runs/.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate fixture package only; do not create a corpus or upload documents.",
        )

    def handle(self, *args, **options) -> None:
        suite = load_howard_contract_suite(options["root"])
        missing = suite.missing_pdf_fixtures
        if missing:
            missing_text = "\n".join(f"  - {path}" for path in missing)
            raise CommandError(
                "Howard fixture PDFs are not staged. Add the SEC-rendered PDFs "
                f"before running ingestion:\n{missing_text}"
            )

        self.stdout.write(
            self.style.NOTICE(
                f"Loaded Howard fixture package: {len(suite.documents)} documents, "
                f"{len(suite.questions)} questions"
            )
        )

        if options["dry_run"]:
            self.stdout.write(self.style.SUCCESS("Dry run complete; no data created."))
            return

        username = options["user"]
        User = get_user_model()
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist as exc:
            raise CommandError(f"User {username!r} not found") from exc

        run_id = uuid.uuid4().hex[:12]
        timestamp = timezone.now().strftime("%Y%m%d-%H%M%S")
        corpus_title = (
            options["corpus_title"] or f"Howard Expedia/TRX viability {timestamp}"
        )
        corpus = Corpus.objects.create(
            title=corpus_title,
            description=(
                "Howard viability evaluation corpus loaded from the public "
                "Expedia/TRX SEC contract package."
            ),
            creator=user,
        )
        set_permissions_for_obj_to_user(user, corpus, [PermissionTypes.CRUD])

        report: dict[str, Any] = {
            "run_id": run_id,
            "started_at": timezone.now().isoformat(),
            "fixture_root": str(suite.root),
            "corpus_id": corpus.id,
            "corpus_title": corpus.title,
            "embedder_path": options["embedder_path"],
            "dimension": options["dimension"],
            "documents": [],
            "question_count": len(suite.questions),
            "ground_truth_claim_count": len(suite.ground_truth.get("claims", [])),
        }

        for fixture_document in suite.documents:
            doc_report = self._ingest_one_document(
                user=user,
                corpus=corpus,
                run_id=run_id,
                fixture_document=fixture_document,
                corpus_dir=suite.corpus_dir,
                timeout_seconds=options["timeout_seconds"],
                embedding_timeout_seconds=options["embedding_timeout_seconds"],
                embedder_path=options["embedder_path"],
                dimension=options["dimension"],
            )
            report["documents"].append(doc_report)

        report["finished_at"] = timezone.now().isoformat()
        report["final_pass"] = all(
            doc.get("processing_status") == DocumentProcessingStatus.COMPLETED
            and doc.get("document_embedding_present")
            and doc.get("text_length", 0) > 0
            and doc.get("annotation_count", 0) > 0
            for doc in report["documents"]
        )

        report_path = options["report_path"] or _default_report_path(timestamp, run_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

        self.stdout.write(self.style.SUCCESS("Howard viability ingest check complete."))
        self.stdout.write(f"  Corpus ID:   {corpus.id}")
        self.stdout.write(f"  Report path: {report_path}")
        self.stdout.write(f"  FINAL_PASS:  {report['final_pass']}")

    def _ingest_one_document(
        self,
        *,
        user,
        corpus: Corpus,
        run_id: str,
        fixture_document: HowardFixtureDocument,
        corpus_dir: Path,
        timeout_seconds: int,
        embedding_timeout_seconds: int,
        embedder_path: str,
        dimension: int,
    ) -> dict[str, Any]:
        source_path = corpus_dir / fixture_document.filename
        upload_filename = f"{run_id}-{fixture_document.filename}"
        self.stdout.write(f"Uploading {upload_filename}")

        source_doc, error = DocumentService.create_document(
            user=user,
            file_bytes=source_path.read_bytes(),
            filename=upload_filename,
            title=fixture_document.title,
            description=f"Howard fixture source: {fixture_document.source_url}",
            custom_meta={
                "howard_fixture_key": fixture_document.key,
                "howard_source_url": fixture_document.source_url,
                "howard_run_id": run_id,
            },
            is_public=False,
        )
        if source_doc is None:
            raise CommandError(
                f"Upload failed for {fixture_document.key}: {error or 'unknown error'}"
            )

        processing_status = _wait_for_processing(
            source_doc.id, timeout_seconds=timeout_seconds
        )
        source_doc.refresh_from_db()

        embedding_present = False
        if processing_status == DocumentProcessingStatus.COMPLETED:
            embedding_present = _wait_for_document_embedding(
                source_doc.id,
                embedder_path=embedder_path,
                dimension=dimension,
                timeout_seconds=embedding_timeout_seconds,
            )
            source_doc.refresh_from_db()

        corpus_doc = None
        status = "not_added"
        if processing_status == DocumentProcessingStatus.COMPLETED:
            corpus_doc, status, error = CorpusDocumentService.add_document_to_corpus(
                user=user,
                document=source_doc,
                corpus=corpus,
            )
            if corpus_doc is None:
                raise CommandError(
                    f"Corpus add failed for {fixture_document.key}: {error or status}"
                )

        analysis_doc = source_doc
        text_length = _document_text_length(analysis_doc)
        annotation_count = Annotation.objects.filter(document=analysis_doc).count()
        annotation_embedding_count = Embedding.objects.filter(
            annotation__document=analysis_doc,
            embedder_path=embedder_path,
            **{f"vector_{dimension}__isnull": False},
        ).count()
        relationship_count = Relationship.objects.filter(document=analysis_doc).count()
        relationship_embedding_count = Embedding.objects.filter(
            relationship__document=analysis_doc,
            embedder_path=embedder_path,
            **{f"vector_{dimension}__isnull": False},
        ).count()

        return {
            "fixture_key": fixture_document.key,
            "title": fixture_document.title,
            "source_url": fixture_document.source_url,
            "document_id": analysis_doc.id,
            "corpus_document_id": corpus_doc.id if corpus_doc else None,
            "upload_status": status,
            "processing_status": processing_status,
            "processing_error": analysis_doc.processing_error,
            "page_count": analysis_doc.page_count,
            "text_length": text_length,
            "annotation_count": annotation_count,
            "annotation_embedding_count": annotation_embedding_count,
            "relationship_count": relationship_count,
            "relationship_embedding_count": relationship_embedding_count,
            "document_embedding_present": embedding_present,
        }


def _wait_for_processing(document_id: int, *, timeout_seconds: int) -> str:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        status = (
            Document.objects.filter(id=document_id)
            .values_list("processing_status", flat=True)
            .first()
        )
        if status in (DocumentProcessingStatus.COMPLETED, DocumentProcessingStatus.FAILED):
            return status
        time.sleep(1)
    raise CommandError(
        f"Document {document_id} did not finish processing within {timeout_seconds}s"
    )


def _wait_for_document_embedding(
    document_id: int,
    *,
    embedder_path: str,
    dimension: int,
    timeout_seconds: int,
) -> bool:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        document = Document.objects.get(id=document_id)
        if document.get_embedding(embedder_path, dimension) is not None:
            return True
        time.sleep(1)
    return False


def _document_text_length(document: Document) -> int:
    if not document.txt_extract_file:
        return 0
    with document.txt_extract_file.open("rb") as handle:
        data = handle.read()
    return len(data.decode("utf-8", errors="replace"))


def _default_report_path(timestamp: str, run_id: str) -> Path:
    return Path("benchmark_runs") / f"{timestamp}_howard_expedia_trx_{run_id}" / "ingest_report.json"
