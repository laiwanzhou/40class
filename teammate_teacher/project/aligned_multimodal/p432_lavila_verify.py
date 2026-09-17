"""Durable verifier for the P432/P158 LaViLa single-member probability bank.

This module is intentionally a verifier only: it never constructs a model on
an accelerator and never fits or selects a checkpoint.  All values are checked
against the receipt, the frozen cache lineage, and the callback artifacts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .p432_lavila_cache import CHECKPOINT_SHA, LaViLaCache
from .p432_lavila_provider import BASE_SEED, EXCLUDED_USERS
from .p432_lavila_training import FIXED_RECIPE, TokenHead
from .p432_probe_contract import validate_spotcheck, sha
from .stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance


def state_schema(tokens: int = 48):
    """Return the exact CPU state schema without allocating model weights."""
    if tokens != 48:
        raise ProtocolError("invalid LaViLa token count")
    with torch.device("meta"):
        model = TokenHead(input_dim=768, hidden_dim=192, heads=6, layers=2,
                          dropout=.20, view_dropout=.15, num_tokens=48)
    return {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}


def verify_diagnostics(d, rows, seed):
    if not isinstance(d, dict):
        raise ProtocolError("missing LaViLa diagnostics")
    planned = 35 * math.ceil(rows / 128)
    vals = np.asarray([d.get("planned_steps"), d.get("optimizer_steps"), d.get("amp_skips")], dtype=float)
    if not np.isfinite(vals).all() or np.any(vals != np.floor(vals)):
        raise ProtocolError("noninteger LaViLa update counters")
    total, good, skipped = vals
    lr = 3e-4 * .05 + (3e-4 - 3e-4 * .05) * (1 + np.cos(np.pi * good / planned)) / 2
    if (total != planned or good + skipped != planned or good < .9 * planned or skipped < 0
            or d.get("epochs") != 35 or d.get("seed") != seed or d.get("token_count") != 48
            or d.get("input_dim") != 768 or d.get("recipe") != FIXED_RECIPE
            or not np.isfinite(d.get("final_lr", np.nan))
            or not np.isclose(d["final_lr"], lr, rtol=1e-8, atol=1e-12)):
        raise ProtocolError("LaViLa training recipe/progress mismatch")
    telemetry = d.get("telemetry")
    if not isinstance(telemetry, list) or [x.get("epoch") for x in telemetry] != [1, 10, 20, 30, 35]:
        raise ProtocolError("LaViLa epoch telemetry incomplete")
    prior = 0
    for x in telemetry:
        counts = np.asarray([x.get("optimizer_steps"), x.get("amp_skips")], dtype=float)
        if (not np.isfinite(counts).all() or np.any(counts != np.floor(counts))
                or np.any(counts < 0) or counts.sum() != x["epoch"] * math.ceil(rows / 128)
                or counts[0] < prior or not np.isfinite(x.get("train_loss", np.nan))):
            raise ProtocolError("invalid LaViLa epoch telemetry")
        prior = counts[0]
    if telemetry[-1]["optimizer_steps"] != good or telemetry[-1]["amp_skips"] != skipped:
        raise ProtocolError("LaViLa final counter mismatch")


def _fresh_probe(cache_info, evidence, users):
    paths = cache_info.get("input_sha256", {})
    if not isinstance(paths, dict) or len(paths) != 7:
        raise ProtocolError("LaViLa cache provenance inventory differs")
    for p, digest in paths.items():
        if sha(p) != digest:
            raise ProtocolError("LaViLa cache input changed")
    by_name = {Path(p).name: Path(p) for p in paths}
    if not {"frame_tokens.npy", "summary.json", "rows.csv", "manifest.csv"}.issubset(by_name):
        raise ProtocolError("LaViLa cache constructor paths missing")
    cache = LaViLaCache(by_name["frame_tokens.npy"].parent, by_name["rows.csv"], by_name["manifest.csv"])
    if cache.provenance != cache_info:
        raise ProtocolError("LaViLa cache provenance differs")
    actual = validate_spotcheck(cache, evidence["probe_dir"], users)
    if actual != evidence:
        raise ProtocolError("LaViLa probe evidence changed")
    return cache


def verify_context(folder, expected):
    folder = Path(folder)
    r = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    required = ("context", "outer_fold", "source_ids", "source_users", "target_ids", "target_users",
                "source_label_sha256", "source_class_counts", "cache_provenance", "probe_evidence")
    for key in required:
        if r[key] != expected[key]:
            raise ProtocolError(f"LaViLa authoritative context differs: {key}")
    if (r.get("expert_name") != "p158_lavila_frame_token" or r.get("aggregation") != "softmax(single_float32_logits)"
            or r.get("target_labels_received") is not False or r.get("checkpoint_selection") is not False
            or r.get("injected_test_fitter") is not False or r.get("member_callback_used") is not True
            or r.get("provenance_checked") is not True):
        raise ProtocolError("LaViLa recipe/fit mode invalid")
    source, target = r["source_ids"], r["target_ids"]
    su, tu = set(r["source_users"]), set(r["target_users"])
    if (not source or not target or len(source)!=len(r["source_users"]) or len(target)!=len(r["target_users"])
            or len(set(source)) != len(source) or len(set(target)) != len(target) or set(source) & set(target)
            or su & tu or (su | tu) & set(EXCLUDED_USERS)):
        raise ProtocolError("LaViLa identity/subject exclusion failed")
    cache_info = r["cache_provenance"]
    if (cache_info.get("type") != "frozen_external" or cache_info.get("model") != "LaViLa_TimeSformer_B"
            or cache_info.get("checkpoint_sha256") != CHECKPOINT_SHA or cache_info.get("labels_consumed") is not False):
        raise ProtocolError("LaViLa cache ancestry missing")
    evidence = r["probe_evidence"]
    _fresh_probe(cache_info, evidence, dict(zip(source + target, r["source_users"] + r["target_users"])))

    fit = r.get("fits")
    if isinstance(r["outer_fold"],bool) or r["outer_fold"] not in (0,1,2):
        raise ProtocolError("invalid LaViLa outer fold")
    seed = BASE_SEED + 1000 * int(r["outer_fold"])
    if not isinstance(fit, list) or len(fit) != 1:
        raise ProtocolError("LaViLa member count wrong")
    f = fit[0]
    if r.get("recipe")!=FIXED_RECIPE or r.get("seed")!=seed or r.get("diagnostics")!=f.get("diagnostics"):
        raise ProtocolError("LaViLa top-level recipe/diagnostics differ")
    node = r["context"] + f".seed{seed}"
    if (f.get("node") != node or f.get("seed") != seed or f.get("source_ids") != source
            or f.get("source_users") != r["source_users"] or f.get("target_ids") != target
            or f.get("target_users") != r["target_users"]):
        raise ProtocolError("LaViLa member recipe/IDs wrong")
    verify_diagnostics(f["diagnostics"], len(source), seed)
    member = folder / "members" / f"seed{seed}"
    mr = json.loads((member / "receipt.json").read_text(encoding="utf-8"))
    if any(mr.get(k) != v for k, v in f.items()):
        raise ProtocolError("LaViLa member/context receipts disagree")
    if sha(member / "checkpoint.pt") != mr.get("checkpoint_sha256") or sha(member / "outputs.npz") != mr.get("logits_sha256"):
        raise ProtocolError("LaViLa member artifact hash mismatch")
    state = torch.load(member / "checkpoint.pt", map_location="cpu", weights_only=True)
    schema = state_schema()
    if (not state or set(state) != set(schema)
            or any((tuple(state[k].shape), state[k].dtype) != v or not state[k].is_cpu or not torch.isfinite(state[k]).all()
                   for k, v in schema.items())):
        raise ProtocolError("checkpoint is not the exact LaViLa TokenHead state schema")
    with np.load(member / "outputs.npz", allow_pickle=False) as mz, np.load(folder / "bank.npz", allow_pickle=False) as z:
        if set(mz.files) != {"logits"} or set(z.files) != {"probabilities", "logits", "sample_ids", "users", "expert_names"}:
            raise ProtocolError("LaViLa bank/member array inventory differs")
        logits = np.asarray(mz["logits"])
        if logits.shape != (len(target), 40) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ProtocolError("invalid LaViLa member logits")
        if z["logits"].dtype!=np.float32 or z["probabilities"].dtype!=np.float32 or not np.array_equal(z["logits"], logits) or not np.array_equal(z["sample_ids"], target) or not np.array_equal(z["users"], r["target_users"]):
            raise ProtocolError("LaViLa bank/member identity differs")
        if z["expert_names"].astype(str).tolist() != ["p158_lavila_frame_token"] or z["probabilities"].shape != (len(target), 1, 40):
            raise ProtocolError("LaViLa bank schema differs")
        p = (np.exp(logits - logits.max(1, keepdims=True)).astype(np.float32))
        p /= p.sum(1, keepdims=True, dtype=np.float32)
        if p.dtype != np.float32 or not np.array_equal(p[:, None, :], z["probabilities"]):
            raise ProtocolError("not original float32 logits softmax")
    external = r["context"] + ".external_lavila"
    wanted = {external: ArtifactNode(external, provenance="frozen_external"),
              node: ArtifactNode(node, (external,), frozenset(su), "supervised", True),
              r["context"] + ".probability": ArtifactNode(r["context"] + ".probability", (node,))}
    actual = {k: ArtifactNode(k, tuple(v["parents"]), frozenset(v["supervised_train_subjects"]), v["provenance"], v["has_task_labels"], v.get("oof_labels_only", False)) for k, v in r["artifact_dag"].items()}
    if actual != wanted or any(k!=v.get("node_id") for k,v in r["artifact_dag"].items()) or r.get("prediction_node") != r["context"] + ".probability":
        raise ProtocolError("LaViLa provenance topology differs")
    for u in tu:
        assert_prediction_provenance(u, [r["context"] + ".probability"], actual)
    return r
