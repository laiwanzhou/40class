"""Durable verifier for the source-only P433 physical-token bank."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .p433_physical_cache import FAMILY_NAMES, PhysicalTokenCache, _sha as sha
from .p433_physical_provider import BASE_SEEDS, EXCLUDED_USERS, EXPERT_NAME
from .p433_physical_training import FIXED_RECIPE, TokenHead
from .stable_routing_protocol import ArtifactNode, ProtocolError, assert_prediction_provenance


def state_schema():
    with torch.device("meta"):
        model = TokenHead(input_dim=768, hidden_dim=192, heads=6, layers=2,
                          dropout=.20, view_dropout=.15, num_tokens=18)
    return {k: (tuple(v.shape), v.dtype) for k, v in model.state_dict().items()}


def verify_diagnostics(d, rows, seed):
    if not isinstance(d, dict): raise ProtocolError("missing P433 diagnostics")
    planned = 35 * math.ceil(rows / 128)
    values = np.asarray([d.get("planned_steps"), d.get("optimizer_steps"), d.get("amp_skips"), d.get("scheduler_steps")], dtype=float)
    if not np.isfinite(values).all() or np.any(values != np.floor(values)):
        raise ProtocolError("noninteger P433 update counters")
    total, good, skipped, scheduler_steps = values
    lr = 3e-4 * .05 + (3e-4 - 3e-4 * .05) * (1 + np.cos(np.pi * scheduler_steps / planned)) / 2
    if (total != planned or scheduler_steps != planned or good + skipped != planned or good < .9 * planned
            or skipped < 0 or d.get("epochs") != 35 or d.get("seed") != seed
            or d.get("token_count") != 18 or d.get("input_dim") != 768 or d.get("recipe") != FIXED_RECIPE
            or not np.isfinite(d.get("final_lr", np.nan)) or not np.isclose(d["final_lr"], lr, rtol=1e-8, atol=1e-12)):
        raise ProtocolError("P433 training recipe/progress mismatch")
    telemetry = d.get("telemetry")
    if not isinstance(telemetry, list) or [x.get("epoch") for x in telemetry] != [1, 10, 20, 30, 35]:
        raise ProtocolError("P433 epoch telemetry incomplete")
    prior = 0
    for x in telemetry:
        c = np.asarray([x.get("optimizer_steps"), x.get("amp_skips"), x.get("scheduler_steps")], dtype=float)
        expected = x["epoch"] * math.ceil(rows / 128)
        if (not np.isfinite(c).all() or np.any(c != np.floor(c)) or np.any(c < 0)
                or c[0] + c[1] != expected or c[2] != expected or c[0] < prior
                or not np.isfinite(x.get("train_loss", np.nan))):
            raise ProtocolError("invalid P433 epoch telemetry")
        prior = c[0]
    if (telemetry[-1]["optimizer_steps"] != good or telemetry[-1]["amp_skips"] != skipped
            or telemetry[-1]["scheduler_steps"] != scheduler_steps):
        raise ProtocolError("P433 final counter mismatch")


def _fresh_cache(provenance):
    if (not isinstance(provenance, dict) or provenance.get("type") != "frozen_external"
            or provenance.get("source") != "P238_original_physical_tokens"
            or provenance.get("families") != list(FAMILY_NAMES) or provenance.get("token_shape") != [18, 768]
            or provenance.get("labels_loaded") is not False or provenance.get("features_normalized") is not False
            or provenance.get("availability_masks_used") is not False):
        raise ProtocolError("P433 cache provenance invalid")
    paths = provenance.get("paths")
    if not isinstance(paths, dict) or list(paths) != list(FAMILY_NAMES):
        raise ProtocolError("P433 cache paths differ")
    manifest = Path(__file__).resolve().parent / "data" / "manifest.csv"
    cache = PhysicalTokenCache([paths[n] for n in FAMILY_NAMES], manifest=manifest)
    if cache.provenance != provenance:
        raise ProtocolError("P433 cache provenance changed")
    for key, digest in provenance.get("input_sha256", {}).items():
        target = manifest if key == "canonical_manifest" else Path(paths[key[:-9]]) if key.endswith("_features") else Path(paths[key[:-9]]).with_name("cache_summary.json")
        if sha(target) != digest: raise ProtocolError("P433 frozen cache input changed")
    return cache


def verify_context(folder, expected):
    folder = Path(folder)
    r = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
    keys = ("context", "outer_fold", "source_ids", "source_users", "target_ids", "target_users",
            "source_label_sha256", "source_class_counts", "cache_provenance")
    for key in keys:
        if r[key] != expected[key]: raise ProtocolError(f"P433 authoritative context differs: {key}")
    if isinstance(r['outer_fold'],bool) or r['outer_fold'] not in (0,1,2) or r.get('recipe')!=FIXED_RECIPE:
        raise ProtocolError('P433 fold/recipe differs')
    if (r.get("expert_name") != EXPERT_NAME or r.get("seed_bases") != list(BASE_SEEDS)
            or r.get("seeds") != [b + 1000 * r["outer_fold"] for b in BASE_SEEDS]
            or r.get("member_count") != 3 or r.get("aggregation") != "mean_float32_logits_then_softmax_float64"
            or r.get("target_labels_received") is not False or r.get("checkpoint_selection") is not False
            or r.get("injected_test_fitter") is not False or r.get("member_callback_used") is not True
            or r.get("provenance_checked") is not True):
        raise ProtocolError("P433 receipt scope/recipe invalid")
    source, target = r["source_ids"], r["target_ids"]; su, tu = set(r["source_users"]), set(r["target_users"])
    if (len(set(source)) != len(source) or len(set(target)) != len(target) or set(source) & set(target)
            or su & tu or (su | tu) & set(EXCLUDED_USERS)):
        raise ProtocolError("P433 source/target group isolation failed")
    cache=_fresh_cache(r["cache_provenance"])
    bound=dict(zip(cache.master_ids,cache.master_users))
    if (not source or not target or len(source)!=len(r['source_users']) or len(target)!=len(r['target_users'])
        or any(i not in bound or bound[i]!=u for i,u in zip(source+target,r['source_users']+r['target_users']))):
        raise ProtocolError('P433 canonical identity differs')
    fits = r.get("fits"); seeds = [b + 1000 * int(r["outer_fold"]) for b in BASE_SEEDS]
    if not isinstance(fits, list) or len(fits) != 3: raise ProtocolError("P433 member count wrong")
    nodes = {}; external = r["context"] + ".external_physical"; nodes[external] = ArtifactNode(external, provenance="frozen_external")
    for j, (f, seed) in enumerate(zip(fits, seeds)):
        node = r["context"] + f".seed{seed}"
        if (f.get("node") != node or f.get("seed") != seed or f.get("source_ids") != source
                or f.get("source_users") != r["source_users"] or f.get("target_ids") != target
                or f.get("target_users") != r["target_users"]): raise ProtocolError("P433 member recipe/IDs wrong")
        verify_diagnostics(f["diagnostics"], len(source), seed)
        member = folder / "members" / f"seed{seed}"
        mr = json.loads((member / "receipt.json").read_text(encoding="utf-8"))
        if any(mr.get(k) != v for k, v in f.items()): raise ProtocolError("P433 member receipt differs")
        if sha(member / "checkpoint.pt") != mr.get("checkpoint_sha256") or sha(member / "outputs.npz") != mr.get("logits_sha256"):
            raise ProtocolError("P433 member artifact hash mismatch")
        state = torch.load(member / "checkpoint.pt", map_location="cpu", weights_only=True); schema = state_schema()
        if (not state or set(state) != set(schema) or any((tuple(state[k].shape), state[k].dtype) != v or not state[k].is_cpu or not torch.isfinite(state[k]).all() for k, v in schema.items())):
            raise ProtocolError("checkpoint is not exact P433 TokenHead state schema")
        nodes[node] = ArtifactNode(node, (external,), frozenset(su), "supervised", True)
    probability_node = r["context"] + ".probability"; nodes[probability_node] = ArtifactNode(probability_node, tuple(r["context"] + f".seed{s}" for s in seeds))
    with np.load(folder / "bank.npz", allow_pickle=False) as z:
        if set(z.files) != {"probabilities", "member_logits", "mean_logits", "sample_ids", "users", "expert_names"}:
            raise ProtocolError("P433 bank array inventory differs")
        member_arrays = []
        for s in seeds:
            with np.load(folder / "members" / f"seed{s}" / "outputs.npz", allow_pickle=False) as output:
                if set(output.files) != {"logits"} or output["logits"].dtype != np.float32:
                    raise ProtocolError("P433 member logits inventory/dtype differs")
                member_arrays.append(output["logits"])
        member_logits = np.stack(member_arrays)
        if member_logits.shape!=(3,len(target),40) or not np.isfinite(member_logits).all():raise ProtocolError('P433 member logits invalid')
        mean = member_logits.mean(axis=0, dtype=np.float32)
        if not np.isfinite(mean).all():raise ProtocolError('P433 aggregate overflow')
        score = mean.astype(np.float64); score -= score.max(1, keepdims=True)
        p64 = np.exp(score); p64 /= p64.sum(1, keepdims=True); p = p64.astype(np.float32)
        if (any(z[k].dtype!=np.float32 for k in ('member_logits','mean_logits','probabilities')) or not np.array_equal(z["member_logits"], member_logits)
                or not np.array_equal(z["mean_logits"], mean) or not np.array_equal(z["probabilities"], p[:, None, :])
                or not np.array_equal(z["sample_ids"], target) or not np.array_equal(z["users"], r["target_users"])
                or z["expert_names"].astype(str).tolist() != [EXPERT_NAME]): raise ProtocolError("P433 bank values differ")
    actual = {k: ArtifactNode(k, tuple(v["parents"]), frozenset(v["supervised_train_subjects"]), v["provenance"], v["has_task_labels"], v.get("oof_labels_only", False)) for k, v in r["artifact_dag"].items()}
    if actual != nodes or any(k!=v.get('node_id') for k,v in r['artifact_dag'].items()) or r.get("prediction_node") != probability_node: raise ProtocolError("P433 provenance DAG differs")
    for user in tu: assert_prediction_provenance(user, [probability_node], actual)
    return r
