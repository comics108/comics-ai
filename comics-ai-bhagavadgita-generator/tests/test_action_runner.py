import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from action_runner import (ActionPlan, ActionRunner, Authorization, CandidateDraft,
                           GptImage2Provider, create_action)


def _action(provider="fake", cost=0., references=()):
    return create_action(
        id="action-1", action_type="generation_plan", input_version_ids=("asset:v1",),
        input_checksums=("a" * 64,), constraints={}, expected_output_contract="candidate",
        provider_id=provider, model="model", provider_version="1", configuration={"n": 2},
        prompt_hash="b" * 64, authorization_id="auth-1", max_cost_usd=cost,
        reference_source_ids=references,
    )


def _authorization(provider="fake", paid=False, cost=0., upload=False, sources=()):
    return Authorization("auth-1", provider, ("action-1",), "2027-01-01T00:00:00Z",
                         paid, cost, upload, sources, "test")


class FakeProvider:
    provider_id = "fake"

    def __init__(self, plan=None):
        self.calls = 0
        self._plan = plan or ActionPlan("fake", True, False, False, 0., (), (), 2)

    def plan(self, action):
        return self._plan

    def execute(self, action, authorization, staging):
        self.calls += 1
        drafts = []
        for index in range(2):
            path = staging / f"candidate-{index}.txt"
            path.write_text(f"candidate {index}", encoding="utf-8")
            drafts.append(CandidateDraft((path,), {"index": index}))
        return drafts


def _runner(tmp_path):
    return ActionRunner(tmp_path, code_revision="test", timestamp="2026-08-11T00:00:00Z",
                        now=datetime(2026, 8, 11, tzinfo=timezone.utc))


def test_successful_retry_is_cached_and_never_duplicates_candidates(tmp_path):
    provider = FakeProvider()
    runner = _runner(tmp_path)
    first = runner.run(_action(), provider, _authorization())
    second = runner.run(_action(), provider, _authorization())
    assert first.cached is False and second.cached is True
    assert provider.calls == 1
    assert len(first.candidates) == len(second.candidates) == 2
    assert all(candidate.review_state == "proposed" for candidate in first.candidates)
    assert all(candidate.lineage.input_checksums == ("a" * 64,) for candidate in first.candidates)


def test_paid_or_upload_execution_is_denied_before_provider_call(tmp_path):
    paid_plan = ActionPlan("fake", True, True, True, 2., (), (), 1)
    provider = FakeProvider(paid_plan)
    with pytest.raises(PermissionError, match="paid"):
        _runner(tmp_path).run(_action(cost=5.), provider, _authorization(paid=False, cost=5.))
    assert provider.calls == 0

    upload_plan = ActionPlan("fake", True, True, False, 0., ("source-secret",), (), 1)
    provider = FakeProvider(upload_plan)
    with pytest.raises(PermissionError, match="upload"):
        _runner(tmp_path / "upload").run(
            _action(references=("source-secret",)), provider, _authorization(upload=False)
        )
    assert provider.calls == 0


def test_tampered_fingerprint_and_expired_authorization_fail_closed(tmp_path):
    action = _action()
    tampered = action.__class__(**{**action.__dict__, "constraints": {"changed": True}})
    with pytest.raises(ValueError, match="fingerprint"):
        _runner(tmp_path).run(tampered, FakeProvider(), _authorization())
    expired = Authorization("auth-1", "fake", ("action-1",), "2020-01-01T00:00:00Z",
                            False, 0., False, (), "test")
    with pytest.raises(PermissionError, match="expired"):
        _runner(tmp_path / "expired").run(action, FakeProvider(), expired)


def test_paid_provider_retry_reuses_remote_idempotency_key(tmp_path):
    class IdempotentPaidProvider(FakeProvider):
        def __init__(self):
            super().__init__(ActionPlan("fake", True, True, True, 1., (), (), 1))
            self.remote_results = {}
            self.fail_after_first_remote_call = True

        def execute(self, action, authorization, staging):
            if action.idempotency_fingerprint not in self.remote_results:
                self.calls += 1
                self.remote_results[action.idempotency_fingerprint] = b"remote-result"
                if self.fail_after_first_remote_call:
                    self.fail_after_first_remote_call = False
                    raise RuntimeError("transport failed after remote completion")
            path = staging / "candidate.bin"
            path.write_bytes(self.remote_results[action.idempotency_fingerprint])
            return [CandidateDraft((path,), {"remote_cache": True})]

    provider = IdempotentPaidProvider()
    runner = _runner(tmp_path)
    action = _action(cost=2.)
    authorization = _authorization(paid=True, cost=2.)
    with pytest.raises(RuntimeError, match="transport"):
        runner.run(action, provider, authorization)
    result = runner.run(action, provider, authorization)
    assert result.cached is False
    assert provider.calls == 1
    assert result.candidates[0].metadata["remote_cache"] is True


def test_gpt_image_adapter_remains_disabled_before_paid_upload_authorization(tmp_path):
    class SpyClient:
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            return [{"bytes": b"image", "extension": "png", "metadata": {}}]

    client = SpyClient()
    provider = GptImage2Provider(client)
    action = create_action(
        id="action-1", action_type="edit", input_version_ids=("asset:v1",),
        input_checksums=("a" * 64,), constraints={}, expected_output_contract="rgba_candidate",
        provider_id="gpt-image-2", model="gpt-image-2", provider_version="1",
        configuration={"candidate_count": 4, "estimated_cost_usd": 1.5},
        prompt_hash="b" * 64, authorization_id="auth-1", max_cost_usd=2.,
        reference_source_ids=("source-private",),
    )
    authorization = Authorization(
        "auth-1", "gpt-image-2", ("action-1",), "2027-01-01T00:00:00Z",
        False, 2., False, (), "test",
    )
    with pytest.raises(PermissionError, match="paid"):
        _runner(tmp_path).run(action, provider, authorization)
    assert client.calls == 0
