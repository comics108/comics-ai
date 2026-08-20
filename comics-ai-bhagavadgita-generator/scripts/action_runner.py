"""Provider-neutral immutable action runner with authorization and idempotent caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from production_models import Lineage


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ModelAction:
    id: str
    action_type: str
    input_version_ids: tuple[str, ...]
    input_checksums: tuple[str, ...]
    constraints: dict[str, Any]
    expected_output_contract: str
    provider_id: str
    model: str
    provider_version: str
    configuration: dict[str, Any]
    prompt_hash: str | None
    authorization_id: str
    max_cost_usd: float
    reference_source_ids: tuple[str, ...]
    idempotency_fingerprint: str

    def fingerprint_payload(self) -> dict:
        payload = asdict(self)
        payload.pop("idempotency_fingerprint")
        return payload

    def validate(self) -> None:
        if not all((self.id, self.action_type, self.expected_output_contract, self.provider_id,
                    self.model, self.provider_version, self.authorization_id)):
            raise ValueError("action identity/provider/output/authorization fields are required")
        if self.max_cost_usd < 0 or not self.input_checksums:
            raise ValueError("action budget and input checksums are invalid")
        if canonical_sha256(self.fingerprint_payload()) != self.idempotency_fingerprint:
            raise ValueError("action idempotency fingerprint does not match immutable request")


def create_action(**fields) -> ModelAction:
    provisional = ModelAction(**fields, idempotency_fingerprint="")
    return ModelAction(**fields, idempotency_fingerprint=canonical_sha256(provisional.fingerprint_payload()))


@dataclass(frozen=True)
class Authorization:
    id: str
    provider_id: str
    action_ids: tuple[str, ...]
    expires_at: str
    allow_paid_call: bool
    max_cost_usd: float
    allow_reference_upload: bool
    allowed_source_ids: tuple[str, ...]
    issuer: str


@dataclass(frozen=True)
class ActionPlan:
    provider_id: str
    supported: bool
    external: bool
    paid: bool
    estimated_cost_usd: float
    upload_source_ids: tuple[str, ...]
    prerequisites: tuple[str, ...]
    expected_outputs: int
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class CandidateDraft:
    files: tuple[Path, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Candidate:
    id: str
    action_id: str
    index: int
    files: tuple[str, ...]
    file_checksums: tuple[str, ...]
    metadata: dict[str, Any]
    review_state: Literal["proposed"]
    lineage: Lineage


@dataclass(frozen=True)
class RunResult:
    candidates: tuple[Candidate, ...]
    cached: bool


class ActionProvider(Protocol):
    provider_id: str

    def plan(self, action: ModelAction) -> ActionPlan: ...
    def execute(self, action: ModelAction, authorization: Authorization, staging: Path) -> list[CandidateDraft]: ...


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate_authorization(
    action: ModelAction, plan: ActionPlan, authorization: Authorization, now: datetime
) -> None:
    if authorization.id != action.authorization_id or authorization.provider_id != action.provider_id:
        raise PermissionError("authorization is not bound to this action/provider")
    if action.id not in authorization.action_ids:
        raise PermissionError("authorization does not include this action")
    if _parse_time(authorization.expires_at) <= now.astimezone(timezone.utc):
        raise PermissionError("authorization has expired")
    if not plan.supported:
        raise ValueError(f"provider does not support action: {plan.unsupported_reason}")
    if plan.paid:
        if not authorization.allow_paid_call:
            raise PermissionError("paid provider call is not authorized")
        allowed_cost = min(action.max_cost_usd, authorization.max_cost_usd)
        if plan.estimated_cost_usd > allowed_cost:
            raise PermissionError("provider estimate exceeds authorized budget")
    uploads = set(plan.upload_source_ids)
    if uploads:
        if not authorization.allow_reference_upload:
            raise PermissionError("reference upload is not authorized")
        if not uploads.issubset(authorization.allowed_source_ids):
            raise PermissionError("reference upload source is outside authorization")


def _candidate_from_payload(payload: dict) -> Candidate:
    lineage_payload = payload["lineage"]
    lineage = Lineage(**{
        **lineage_payload,
        "input_checksums": tuple(lineage_payload["input_checksums"]),
        "reviewer_decision_ids": tuple(lineage_payload["reviewer_decision_ids"]),
    })
    return Candidate(**{
        **payload, "files": tuple(payload["files"]),
        "file_checksums": tuple(payload["file_checksums"]), "lineage": lineage,
    })


class ActionRunner:
    def __init__(self, root: Path, *, code_revision: str, timestamp: str, now: datetime):
        self.root = root
        self.code_revision = code_revision
        self.timestamp = timestamp
        self.now = now

    def run(
        self, action: ModelAction, provider: ActionProvider, authorization: Authorization
    ) -> RunResult:
        action.validate()
        if provider.provider_id != action.provider_id:
            raise ValueError("provider ID does not match action")
        action_root = self.root / action.idempotency_fingerprint
        result_path = action_root / "result.json"
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            return RunResult(tuple(_candidate_from_payload(item) for item in payload["candidates"]), True)
        plan = provider.plan(action)
        validate_authorization(action, plan, authorization, self.now)
        action_root.mkdir(parents=True, exist_ok=True)
        request_path = action_root / "request.json"
        request_payload = {"schema_version": 1, "action": asdict(action), "plan": asdict(plan),
                           "authorization": asdict(authorization)}
        if request_path.exists():
            existing_request = json.loads(request_path.read_text(encoding="utf-8"))
            if canonical_sha256(existing_request) != canonical_sha256(request_payload):
                raise ValueError("immutable action request changed between retries")
        else:
            request_path.write_text(
                json.dumps(request_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        staging = action_root / ".staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        drafts = provider.execute(action, authorization, staging)
        if not drafts:
            raise ValueError("provider returned no candidates")
        candidates = []
        published_root = action_root / "candidates"
        published_root.mkdir(exist_ok=True)
        staging_resolved = staging.resolve()
        for index, draft in enumerate(drafts, start=1):
            if not draft.files:
                raise ValueError("candidate has no output files")
            target_root = published_root / f"candidate-{index:03}"
            target_root.mkdir()
            published_files, checksums = [], []
            for source in draft.files:
                resolved = source.resolve()
                if not resolved.is_file() or staging_resolved not in resolved.parents:
                    raise ValueError("provider candidate file escaped action staging")
                target = target_root / source.name
                os.replace(source, target)
                published_files.append(str(target))
                checksums.append(file_sha256(target))
            lineage = Lineage(
                input_checksums=action.input_checksums, action_id=action.id,
                code_revision=self.code_revision, model_checkpoint=f"{action.model}@{action.provider_version}",
                configuration_hash=canonical_sha256(action.configuration),
                environment={"provider": action.provider_id, "external": plan.external},
                timestamp=self.timestamp,
                cost_usage={"estimated_usd": plan.estimated_cost_usd,
                            "authorized_max_usd": min(action.max_cost_usd, authorization.max_cost_usd)},
                reviewer_decision_ids=(),
            )
            candidates.append(Candidate(
                id=f"{action.id}:candidate:{index}:v1", action_id=action.id, index=index,
                files=tuple(published_files), file_checksums=tuple(checksums), metadata=draft.metadata,
                review_state="proposed", lineage=lineage,
            ))
        shutil.rmtree(staging)
        with result_path.open("x", encoding="utf-8") as stream:
            json.dump({"schema_version": 1, "candidates": [asdict(item) for item in candidates]},
                      stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
        return RunResult(tuple(candidates), False)


class LocalVisualPlanProvider:
    provider_id = "local-visual-plan-v1"

    def plan(self, action: ModelAction) -> ActionPlan:
        return ActionPlan(self.provider_id, action.action_type == "generation_plan", False, False,
                          0., (), (), 2,
                          None if action.action_type == "generation_plan" else "unsupported action type")

    def execute(self, action: ModelAction, authorization: Authorization, staging: Path) -> list[CandidateDraft]:
        drafts = []
        for index, strategy in enumerate(("source-first-composition", "symbolic-minimal-composition"), start=1):
            path = staging / f"plan-{index}.json"
            path.write_text(json.dumps({
                "schema_version": 1, "action_id": action.id, "strategy": strategy,
                "constraints": action.constraints, "status": "plan_only_not_production_art",
            }, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            drafts.append(CandidateDraft((path,), {"strategy": strategy, "output_kind": "visual_plan"}))
        return drafts


class ExternalImageClient(Protocol):
    def generate(
        self, *, model: str, configuration: dict[str, Any], idempotency_key: str,
        reference_source_ids: tuple[str, ...], max_cost_usd: float,
    ) -> list[dict[str, Any]]: ...


class GptImage2Provider:
    """Authorization-aware adapter; it holds no credential and is disabled without a client."""

    provider_id = "gpt-image-2"

    def __init__(self, client: ExternalImageClient | None = None):
        self.client = client

    def plan(self, action: ModelAction) -> ActionPlan:
        supported = action.action_type in {"generation", "edit", "repair", "outpaint"}
        count = int(action.configuration.get("candidate_count", 4))
        estimated = float(action.configuration.get("estimated_cost_usd", action.max_cost_usd))
        return ActionPlan(
            self.provider_id, supported, True, True, estimated,
            action.reference_source_ids, (), count,
            None if supported else "unsupported gpt-image-2 action type",
        )

    def execute(
        self, action: ModelAction, authorization: Authorization, staging: Path
    ) -> list[CandidateDraft]:
        if self.client is None:
            raise RuntimeError("gpt-image-2 client is disabled; no credential-bearing client supplied")
        outputs = self.client.generate(
            model=action.model, configuration=action.configuration,
            idempotency_key=action.idempotency_fingerprint,
            reference_source_ids=action.reference_source_ids,
            max_cost_usd=min(action.max_cost_usd, authorization.max_cost_usd),
        )
        drafts = []
        for index, output in enumerate(outputs, start=1):
            extension = str(output.get("extension", "png")).lower()
            if extension not in {"png", "jpg", "webp"}:
                raise ValueError("external image client returned unsupported extension")
            data = output.get("bytes")
            if not isinstance(data, bytes) or not data:
                raise ValueError("external image client returned empty image bytes")
            path = staging / f"external-candidate-{index:03}.{extension}"
            path.write_bytes(data)
            drafts.append(CandidateDraft((path,), dict(output.get("metadata", {}))))
        return drafts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage", type=Path)
    parser.add_argument("beats", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    beats = json.loads(args.beats.read_text(encoding="utf-8"))
    beats_by_id = {beat["id"]: beat for chapter in beats["chapters"] for beat in chapter["beats"]}
    provider = LocalVisualPlanProvider()
    runner = ActionRunner(args.root, code_revision="workspace-2026-08-11", timestamp="2026-08-11T00:00:00Z",
                          now=datetime(2026, 8, 11, tzinfo=timezone.utc))
    records = []
    for chapter in coverage["chapters"]:
        for item in chapter["coverage"]:
            if item["state"] != "generation_required":
                continue
            beat = beats_by_id[item["beat_id"]]
            action_id = item["proposed_action_ids"][0]
            checksum = canonical_sha256(beat)
            authorization_id = f"auto-local:{action_id}:v1"
            fields = dict(
                id=action_id, action_type="generation_plan", input_version_ids=(beat["id"],),
                input_checksums=(checksum,), constraints={
                    "source_sloka_ids": beat["source_sloka_ids"], "no_baked_text": True,
                    "scroll_type": "vertical", "canonical_identity_required_before_art": True,
                }, expected_output_contract="visual_plan_candidate", provider_id=provider.provider_id,
                model="deterministic-local-plan", provider_version="1", configuration={"candidate_count": 2},
                prompt_hash=canonical_sha256(beat["synopsis"]), authorization_id=authorization_id,
                max_cost_usd=0., reference_source_ids=(),
            )
            action = create_action(**fields)
            authorization = Authorization(
                authorization_id, provider.provider_id, (action.id,), "2027-08-11T00:00:00Z",
                False, 0., False, (), "auto:local-actions-v1",
            )
            result = runner.run(action, provider, authorization)
            records.append({"action_id": action.id, "fingerprint": action.idempotency_fingerprint,
                            "candidate_ids": [candidate.id for candidate in result.candidates],
                            "cached": result.cached})
    summary = {"schema_version": 1, "coverage_sha256": file_sha256(args.coverage),
               "beats_sha256": file_sha256(args.beats), "action_count": len(records),
               "candidate_count": sum(len(item["candidate_ids"]) for item in records),
               "external_calls": 0, "paid_cost_usd": 0., "actions": records}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"actions": summary["action_count"], "candidates": summary["candidate_count"],
                      "external_calls": 0, "paid_cost_usd": 0.}))


if __name__ == "__main__":
    main()
