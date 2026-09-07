"""Validated cached-prefix records for Vision-OPD training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from verl.protocol import DataProto


TOKEN_IDS_SOURCE = "base_tokenizer_reencoded_openai_response_text"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id_from_extra_info(extra_info: Any) -> str:
    if not isinstance(extra_info, dict):
        raise ValueError("extra_info must be a mapping")
    provenance = extra_info.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("extra_info.provenance must be a mapping")
    sample_id = str(provenance.get("sample_id", "")).strip()
    if not sample_id:
        raise ValueError("extra_info.provenance.sample_id is empty")
    return sample_id


def prompt_and_image_from_messages(messages: Any) -> tuple[str, Path]:
    if not isinstance(messages, (list, np.ndarray)) or len(messages) != 1:
        raise ValueError("cached prefix requires exactly one user message")
    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("cached prefix requires one user-role message")
    content = message.get("content")
    if not isinstance(content, (list, np.ndarray)):
        raise ValueError("cached prefix requires processed multimodal message content")
    text_parts: list[str] = []
    image_paths: list[Path] = []
    for item in content:
        if not isinstance(item, dict):
            raise ValueError("message content item must be a mapping")
        if item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
        elif item.get("type") == "image":
            raw_path = item.get("path") or item.get("image")
            if isinstance(raw_path, (str, Path)):
                image_paths.append(Path(raw_path).resolve())
    if len(image_paths) != 1:
        raise ValueError("cached prefix requires exactly one Student image path")
    return "".join(text_parts).strip(), image_paths[0]


@dataclass(frozen=True)
class CachedPrefixRecord:
    sample_id: str
    prompt_sha256: str
    image_path: Path
    response_token_ids: tuple[int, ...]
    finish_reason: str
    generation_config_sha256: str
    model_path: str


class CachedPrefixStore:
    """Immutable sample-ID index with prompt/image/token validation."""

    def __init__(self, records: dict[str, CachedPrefixRecord], *, path: Path, sha256: str):
        self._records = records
        self.path = path
        self.sha256 = sha256

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_records: int,
        max_response_length: int,
        expected_token_ids_source: str = TOKEN_IDS_SOURCE,
        expected_generation_config_sha256: str | None = None,
        expected_model_path: str | None = None,
    ) -> "CachedPrefixStore":
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"cached prefix Parquet not found: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError("cached prefix Parquet SHA256 mismatch")
        table = pq.read_table(path)
        required = {
            "sample_id", "split", "prompt_sha256", "image_path", "response_token_ids",
            "response_length", "finish_reason", "inference_error", "token_ids_source",
            "generation_config_sha256", "model_path",
        }
        missing = sorted(required.difference(table.column_names))
        if missing:
            raise ValueError(f"cached prefix Parquet missing columns: {missing}")
        if table.num_rows != expected_records:
            raise ValueError(f"expected {expected_records} cached records, found {table.num_rows}")
        records: dict[str, CachedPrefixRecord] = {}
        for row_index, row in enumerate(table.select(sorted(required)).to_pylist()):
            sample_id = str(row["sample_id"]).strip()
            token_ids = tuple(int(token) for token in (row["response_token_ids"] or []))
            if not sample_id or sample_id in records:
                raise ValueError(f"cached row {row_index}: empty or duplicate sample_id")
            if row["split"] != "train":
                raise ValueError(f"cached row {row_index}: split must be train")
            if row["inference_error"] is not None:
                raise ValueError(f"cached row {row_index}: inference_error is not null")
            if row["token_ids_source"] != expected_token_ids_source:
                raise ValueError(f"cached row {row_index}: token_ids_source mismatch")
            if not token_ids or len(token_ids) > max_response_length:
                raise ValueError(f"cached row {row_index}: invalid response token length")
            if int(row["response_length"]) != len(token_ids):
                raise ValueError(f"cached row {row_index}: response_length mismatch")
            if expected_generation_config_sha256 and row["generation_config_sha256"] != expected_generation_config_sha256:
                raise ValueError(f"cached row {row_index}: generation config mismatch")
            if expected_model_path and Path(row["model_path"]).resolve() != Path(expected_model_path).resolve():
                raise ValueError(f"cached row {row_index}: Base model path mismatch")
            records[sample_id] = CachedPrefixRecord(
                sample_id=sample_id,
                prompt_sha256=str(row["prompt_sha256"]),
                image_path=Path(row["image_path"]).resolve(),
                response_token_ids=token_ids,
                finish_reason=str(row["finish_reason"]),
                generation_config_sha256=str(row["generation_config_sha256"]),
                model_path=str(row["model_path"]),
            )
        return cls(records, path=path, sha256=actual_sha256)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def sample_ids(self) -> frozenset[str]:
        return frozenset(self._records)

    def bind(self, batch: DataProto) -> dict[str, float]:
        extra_infos = batch.non_tensor_batch.get("extra_info")
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if extra_infos is None or raw_prompts is None:
            raise ValueError("cached prefix batch requires extra_info and raw_prompt")
        response_ids = np.empty(len(batch), dtype=object)
        sample_ids = np.empty(len(batch), dtype=object)
        finish_reasons = np.empty(len(batch), dtype=object)
        for index, (extra_info, raw_prompt) in enumerate(zip(extra_infos, raw_prompts, strict=True)):
            sample_id = sample_id_from_extra_info(extra_info)
            if sample_id not in self._records:
                raise KeyError(f"cached prefix missing sample_id: {sample_id}")
            record = self._records[sample_id]
            prompt_text, image_path = prompt_and_image_from_messages(raw_prompt)
            prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            if prompt_sha256 != record.prompt_sha256:
                raise ValueError(f"cached prefix prompt mismatch for {sample_id}")
            if image_path != record.image_path:
                raise ValueError(f"cached prefix Student image mismatch for {sample_id}")
            response_ids[index] = list(record.response_token_ids)
            sample_ids[index] = sample_id
            finish_reasons[index] = record.finish_reason
        batch.non_tensor_batch["cached_response_ids"] = response_ids
        batch.non_tensor_batch["cached_sample_id"] = sample_ids
        batch.non_tensor_batch["cached_finish_reason"] = finish_reasons
        return {
            "cached_prefix/resolved_records": float(len(batch)),
            "cached_prefix/online_generation_calls": 0.0,
        }
