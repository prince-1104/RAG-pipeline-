from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from typing import BinaryIO
from starlette.concurrency import run_in_threadpool

from app.settings import get_settings


class ObjectStore:
    async def put_pdf(self, tenant_id: str, corpus_id: str, filename: str, content: bytes) -> str:
        raise NotImplementedError

    async def put_pdf_with_key(self, board: str, class_level: int, subject: str, doc_id: str, content: bytes) -> str:
        """
        Upload PDF to R2 with specific key structure: board=<board>/class=<class>/subject=<subject>/pdfs/<doc_id>.pdf
        Returns R2 URI (r2://bucket/key).
        """
        raise NotImplementedError

    async def put_pdf_with_key_stream(
        self,
        board: str,
        class_level: int,
        subject: str,
        doc_id: str,
        fileobj: BinaryIO,
        content_type: str | None = None,
    ) -> str:
        """
        Stream upload PDF to object store with the standard key structure.
        """
        raise NotImplementedError

    async def get_bytes(self, uri: str) -> bytes:
        raise NotImplementedError

    def build_local_pdf_uri(self, class_level: int, subject: str, filename: str) -> str:
        """
        Build a stable URI for a local CBSE PDF.
        Only implemented by LocalObjectStore.
        """
        raise NotImplementedError


def _is_within(base: Path, target: Path) -> bool:
    base = base.resolve()
    target = target.resolve()
    try:
        target.relative_to(base)
        return True
    except Exception:  # noqa: BLE001
        return False


def _read_pdf_bytes(path: Path) -> bytes:
    with path.open("rb") as f:
        head = f.read(8)
        if not head.startswith(b"%PDF"):
            raise ValueError(f"not_a_pdf:{path.name}")
        rest = f.read()
        return head + rest


@dataclass
class LocalObjectStore(ObjectStore):
    pdf_base_path: str
    uploads_base_path: str

    async def put_pdf_with_key(self, board: str, class_level: int, subject: str, doc_id: str, content: bytes) -> str:
        """Not supported for local storage - use R2 backend."""
        raise NotImplementedError("put_pdf_with_key not supported for local storage backend")

    async def put_pdf_with_key_stream(
        self,
        board: str,
        class_level: int,
        subject: str,
        doc_id: str,
        fileobj: BinaryIO,
        content_type: str | None = None,
    ) -> str:
        """Not supported for local storage - use R2 backend."""
        raise NotImplementedError("put_pdf_with_key_stream not supported for local storage backend")

    async def put_pdf(self, tenant_id: str, corpus_id: str, filename: str, content: bytes) -> str:
        """
        Uploads are supported for future mode, but not required for CBSE local PDFs.
        Writes only under UPLOADS_BASE_PATH (never into PDF_BASE_PATH).
        """
        base = self._resolve_base_path(self.uploads_base_path)
        dest_dir = base / tenant_id / corpus_id / "pdfs"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = (dest_dir / filename).resolve()
        if not _is_within(base, dest):
            raise ValueError("path_traversal_detected")
        with dest.open("wb") as f:
            f.write(content)
        # Stable logical URI (not absolute file path)
        rel = dest.relative_to(base).as_posix()
        return f"localupload://{rel}"

    def build_local_pdf_uri(self, class_level: int, subject: str, filename: str) -> str:
        """
        Build a stable URI for a local CBSE PDF.
        Format: localpdf://cbse/{class}/{subject}/{filename}
        """
        # Sanitize filename to prevent path traversal
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise ValueError("invalid_filename_contains_path_separators")
        safe_subject = subject.replace("/", "_").replace("\\", "_")
        return f"localpdf://cbse/{class_level}/{safe_subject}/{filename}"

    def _resolve_base_path(self, base_path: str) -> Path:
        """Resolve base path: if relative, resolve relative to repo root."""
        base = Path(base_path)
        if base.is_absolute():
            return base.resolve()
        # Relative to repo root (3 levels up from app/storage/object_store.py)
        repo_root = Path(__file__).parent.parent.parent
        return (repo_root / base).resolve()

    async def get_bytes(self, uri: str) -> bytes:
        parsed = urlparse(uri)
        scheme = parsed.scheme
        if scheme == "localpdf":
            base = self._resolve_base_path(self.pdf_base_path)
            rel = (parsed.netloc + parsed.path).lstrip("/")
            target = (base / rel).resolve()
            if not _is_within(base, target):
                raise ValueError("pdf_outside_base_path")
            if not target.exists():
                raise FileNotFoundError(f"pdf_missing:{rel}")
            return _read_pdf_bytes(target)

        if scheme == "localupload":
            base = self._resolve_base_path(self.uploads_base_path)
            rel = (parsed.netloc + parsed.path).lstrip("/")
            target = (base / rel).resolve()
            if not _is_within(base, target):
                raise ValueError("upload_outside_base_path")
            if not target.exists():
                raise FileNotFoundError(f"pdf_missing:{rel}")
            return _read_pdf_bytes(target)

        if scheme == "file":
            # Accept file:// URIs only if they are under PDF_BASE_PATH or UPLOADS_BASE_PATH.
            target = Path(parsed.path).resolve()
            pdf_base = self._resolve_base_path(self.pdf_base_path)
            upl_base = self._resolve_base_path(self.uploads_base_path)
            if not (_is_within(pdf_base, target) or _is_within(upl_base, target)):
                raise ValueError("file_uri_outside_allowed_dirs")
            return _read_pdf_bytes(target)

        raise ValueError("unsupported_uri_scheme")


@dataclass
class R2ObjectStore(ObjectStore):
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    endpoint_url: str

    def __post_init__(self):
        """Initialize S3 client for R2 (S3-compatible)."""
        self._s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",  # R2 uses "auto" as region
        )

    async def put_pdf(self, tenant_id: str, corpus_id: str, filename: str, content: bytes) -> str:
        """Legacy method - not used for R2 ingestion."""
        raise NotImplementedError("use put_pdf_with_key for R2")

    async def put_pdf_with_key(self, board: str, class_level: int, subject: str, doc_id: str, content: bytes) -> str:
        """
        Upload PDF to R2 with key structure: board=<board>/class=<class>/subject=<subject>/pdfs/<doc_id>.pdf
        Returns R2 URI: r2://<bucket>/board=<board>/class=<class>/subject=<subject>/pdfs/<doc_id>.pdf
        """
        from app.observability.logging import log
        
        # Validate PDF header
        if not content.startswith(b"%PDF"):
            raise ValueError("not_a_pdf")

        # Build key with required structure
        # Sanitize inputs to prevent path traversal
        safe_board = board.replace("/", "_").replace("\\", "_")
        safe_subject = subject.replace("/", "_").replace("\\", "_")
        safe_doc_id = doc_id.replace("/", "_").replace("\\", "_").replace(".", "_")
        
        key = f"board={safe_board}/class={class_level}/subject={safe_subject}/pdfs/{safe_doc_id}.pdf"

        log.info(
            "r2_upload_starting",
            bucket=self.bucket_name,
            key=key,
            content_size=len(content),
            doc_id=doc_id,
        )

        try:
            self._s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=content,
                ContentType="application/pdf",
            )
            storage_uri = f"r2://{self.bucket_name}/{key}"
            log.info(
                "r2_upload_success",
                bucket=self.bucket_name,
                key=key,
                storage_uri=storage_uri,
                doc_id=doc_id,
            )
            return storage_uri
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            log.error(
                "r2_upload_client_error",
                bucket=self.bucket_name,
                key=key,
                error_code=error_code,
                error_message=error_msg,
                doc_id=doc_id,
            )
            raise RuntimeError(f"r2_upload_failed: {error_code} - {error_msg}") from e
        except BotoCoreError as e:
            log.error(
                "r2_upload_boto_error",
                bucket=self.bucket_name,
                key=key,
                error=str(e),
                error_type=type(e).__name__,
                doc_id=doc_id,
            )
            raise RuntimeError(f"r2_client_error: {e}") from e

    async def put_pdf_with_key_stream(
        self,
        board: str,
        class_level: int,
        subject: str,
        doc_id: str,
        fileobj: BinaryIO,
        content_type: str | None = None,
    ) -> str:
        """
        Stream upload PDF to R2 with key structure: board=<board>/class=<class>/subject=<subject>/pdfs/<doc_id>.pdf
        Returns R2 URI: r2://<bucket>/board=<board>/class=<class>/subject=<subject>/pdfs/<doc_id>.pdf
        """
        from app.observability.logging import log

        # Build key with required structure
        safe_board = board.replace("/", "_").replace("\\", "_")
        safe_subject = subject.replace("/", "_").replace("\\", "_")
        safe_doc_id = doc_id.replace("/", "_").replace("\\", "_").replace(".", "_")
        key = f"board={safe_board}/class={class_level}/subject={safe_subject}/pdfs/{safe_doc_id}.pdf"

        log.info(
            "r2_stream_upload_starting",
            bucket=self.bucket_name,
            key=key,
            doc_id=doc_id,
        )

        try:
            extra_args = {"ContentType": content_type or "application/pdf"}
            # upload_fileobj streams without loading into memory
            await run_in_threadpool(
                self._s3_client.upload_fileobj,
                Fileobj=fileobj,
                Bucket=self.bucket_name,
                Key=key,
                ExtraArgs=extra_args,
            )
            storage_uri = f"r2://{self.bucket_name}/{key}"
            log.info(
                "r2_stream_upload_success",
                bucket=self.bucket_name,
                key=key,
                storage_uri=storage_uri,
                doc_id=doc_id,
            )
            return storage_uri
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            log.error(
                "r2_stream_upload_client_error",
                bucket=self.bucket_name,
                key=key,
                error_code=error_code,
                error_message=error_msg,
                doc_id=doc_id,
            )
            raise RuntimeError(f"r2_upload_failed: {error_code} - {error_msg}") from e
        except BotoCoreError as e:
            log.error(
                "r2_stream_upload_boto_error",
                bucket=self.bucket_name,
                key=key,
                error=str(e),
                error_type=type(e).__name__,
                doc_id=doc_id,
            )
            raise RuntimeError(f"r2_client_error: {e}") from e

    async def get_bytes(self, uri: str) -> bytes:
        """Retrieve bytes from R2."""
        parsed = urlparse(uri)
        if parsed.scheme != "r2":
            raise ValueError(f"unsupported_uri_scheme_for_r2: {parsed.scheme}")

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"r2_key_not_found: {key}") from e
            raise RuntimeError(f"r2_get_failed: {e}") from e
        except BotoCoreError as e:
            raise RuntimeError(f"r2_client_error: {e}") from e

    def build_local_pdf_uri(self, class_level: int, subject: str, filename: str) -> str:
        """Not supported for R2."""
        raise NotImplementedError("build_local_pdf_uri not supported for R2")


def get_object_store() -> ObjectStore:
    from app.observability.logging import log
    
    s = get_settings()
    if s.storage_backend == "local":
        return LocalObjectStore(pdf_base_path=s.pdf_base_path, uploads_base_path=s.uploads_base_path)
    elif s.storage_backend == "r2":
        missing = []
        if not s.r2_account_id:
            missing.append("R2_ACCOUNT_ID")
        if not s.r2_access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not s.r2_secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        if not s.r2_bucket_name:
            missing.append("R2_BUCKET_NAME")
        if not s.r2_endpoint_url:
            missing.append("R2_ENDPOINT_URL")
        
        if missing:
            error_msg = f"r2_credentials_missing: {', '.join(missing)}"
            log.error("r2_configuration_error", missing_vars=missing)
            raise RuntimeError(error_msg)
        
        log.info(
            "r2_object_store_initialized",
            bucket=s.r2_bucket_name,
            endpoint=s.r2_endpoint_url,
            account_id_set=bool(s.r2_account_id),
        )
        
        return R2ObjectStore(
            account_id=s.r2_account_id,
            access_key_id=s.r2_access_key_id,
            secret_access_key=s.r2_secret_access_key,
            bucket_name=s.r2_bucket_name,
            endpoint_url=s.r2_endpoint_url,
        )
    else:
        raise RuntimeError(f"unsupported_storage_backend:{s.storage_backend}")


