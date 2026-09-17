"""Portable, label-free P315/P307 routing geometry.

This module deliberately copies only the numerical kernels from p137
``group_features``, p134's frozen ``GlobalRepeatConfig``, p89's session
clustering, and p88's dynamic probability alignment.  It does not import any
experiment script or read an OOF/Test artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import datetime as _datetime
import numpy as np


@dataclass(frozen=True)
class RecordingMetadata:
    sample_ids: np.ndarray
    users: np.ndarray
    dates: np.ndarray
    starts: np.ndarray


@dataclass(frozen=True)
class TransitionModel:
    start_log_probability: np.ndarray
    end_log_probability: np.ndarray
    bigram_log_probability: np.ndarray
    trigram_log_probability: np.ndarray


CONFIG = dict(rank=2, gap=300.0, similarity=.75, overlap=.20,
              length=.80, weight=.50, alignment_gap=.20, max_group=3)


def _align(a: np.ndarray, b: np.ndarray, gap: float) -> tuple[list[tuple[int, int]], float]:
    sim = np.sqrt(np.clip(a[:, None, :], 0, None) * np.clip(b[None, :, :], 0, None)).sum(2)
    r, c = sim.shape
    score = np.full((r + 1, c + 1), -np.inf); trace = np.zeros((r + 1, c + 1), np.int8)
    score[0, 0] = 0
    for i in range(1, r + 1): score[i, 0] = score[i-1, 0] - gap; trace[i, 0] = 1
    for j in range(1, c + 1): score[0, j] = score[0, j-1] - gap; trace[0, j] = 2
    for i in range(1, r + 1):
        for j in range(1, c + 1):
            vals = (score[i-1, j-1] + sim[i-1, j-1], score[i-1, j] - gap, score[i, j-1] - gap)
            trace[i, j] = np.argmax(vals); score[i, j] = max(vals)
    pairs = []; i, j = r, c
    while i or j:
        t = int(trace[i, j])
        if i and j and t == 0: pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif i and (not j or t == 1): i -= 1
        else: j -= 1
    pairs.reverse()
    return pairs, float(np.mean([sim[i, j] for i, j in pairs])) if pairs else 0.0


def _valid_date(value: Any) -> str | None:
    """Return a canonical ISO date, rejecting missing and malformed metadata."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    # Metadata is date-only in the routing contract.  Accept an ISO datetime
    # as a convenience, but never use its time portion for ordering.
    try:
        return _datetime.date.fromisoformat(text).isoformat()
    except (TypeError, ValueError):
        try:
            return _datetime.datetime.fromisoformat(text).date().isoformat()
        except (TypeError, ValueError):
            return None


def _row_date(metadata: RecordingMetadata, row: int) -> str | None:
    return _valid_date(metadata.dates[row])


def _timestamp_tie_mask(metadata: RecordingMetadata) -> np.ndarray:
    """Mark all valid rows sharing an exact (canonical date, start) key."""
    keys: dict[tuple[str, float], list[int]] = {}
    for i, start in enumerate(metadata.starts):
        date = _row_date(metadata, i)
        if date is not None and np.isfinite(start):
            keys.setdefault((date, float(start)), []).append(i)
    mask = np.zeros(len(metadata.starts), dtype=bool)
    for rows in keys.values():
        if len(rows) > 1:
            mask[rows] = True
    return mask


def build_safe_sessions(metadata: RecordingMetadata, gap30: float = 30.0):
    """Build deterministic anonymous sessions from valid date/time metadata.

    Rows with missing/malformed dates are retained in the feature output but
    omitted from sessions.  Exact timestamp ties are emitted as isolated
    singleton sessions; they never form a sequence or acquire peers.
    """
    n = len(metadata.dates); groups: dict[str, list[int]] = {}
    for i, s in enumerate(metadata.starts):
        date = _row_date(metadata, i)
        if date is not None and np.isfinite(s): groups.setdefault(date, []).append(i)
    tie_mask = _timestamp_tie_mask(metadata)
    sessions: list[np.ndarray] = []; tied = 0
    for date in sorted(groups):
        rows = groups[date]
        rows.sort(key=lambda i: float(metadata.starts[i]))
        i = 0
        while i < len(rows):
            row = rows[i]
            if tie_mask[row]:
                j = i + 1
                while j < len(rows) and metadata.starts[rows[j]] == metadata.starts[row]:
                    j += 1
                tied += j - i
                sessions.extend([np.asarray([x], dtype=np.int64) for x in rows[i:j]])
                i = j
                continue
            cur = [row]; prev = float(metadata.starts[row]); j = i + 1
            # A tie run is a hard boundary: do not let a preceding unique
            # row consume tied rows (or rows after them) into its sequence.
            while j < len(rows) and not tie_mask[rows[j]]:
                current = float(metadata.starts[rows[j]])
                if current - prev > gap30:
                    break
                cur.append(rows[j]); prev = current; j += 1
            sessions.append(np.asarray(cur, dtype=np.int64)); i = j
    audit = {"rows": n, "sessions": len(sessions), "invalid_rows": n - sum(len(x) for x in sessions), "timestamp_tied_rows": tied,
             "users_used": False, "missing_rows_retained": True}
    return sessions, audit


def _clusters(sessions, probability, decoded, metadata, tie_mask=None):
    if tie_mask is None:
        tie_mask = _timestamp_tie_mask(metadata)
    candidates = []
    for i in range(len(sessions)):
        for j in range(i + 1, min(len(sessions), i + CONFIG["rank"] + 1)):
            a, b = sessions[i], sessions[j]
            # Tied timestamps are fundamentally unordered.  Keep their rows
            # available to callers, but do not let a singleton join a group.
            if np.any(tie_mask[a]) or np.any(tie_mask[b]): continue
            if _row_date(metadata, int(a[0])) != _row_date(metadata, int(b[0])): continue
            if abs(float(np.min(metadata.starts[b])) - float(np.min(metadata.starts[a]))) > CONFIG["gap"]: continue
            ratio = min(len(a), len(b)) / max(len(a), len(b))
            if ratio < CONFIG["length"] or abs(len(a)-len(b)) > 3: continue
            _, sim = _align(probability[a], probability[b], CONFIG["alignment_gap"])
            sa, sb = set(decoded[a].tolist()), set(decoded[b].tolist())
            overlap = len(sa & sb) / max(len(sa | sb), 1)
            if sim < CONFIG["similarity"] or overlap < CONFIG["overlap"]: continue
            candidates.append((sim + overlap + .08 * ratio - .01 * (j-i), i, j))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2])); groups=[]; membership={}
    for _, i, j in candidates:
        li, lj = membership.get(i), membership.get(j)
        if li is None and lj is None: membership[i]=membership[j]=len(groups); groups.append([i,j])
        elif li is not None and lj is None and len(groups[li]) < CONFIG["max_group"]: groups[li].append(j); membership[j]=li
        elif li is None and lj is not None and len(groups[lj]) < CONFIG["max_group"]: groups[lj].append(i); membership[i]=lj
    return [[sessions[k] for k in g] for g in groups]


def build_group_features(bank: np.ndarray, metadata: RecordingMetadata, include_peers: bool = True,
                         grouping_bank: np.ndarray | None = None):
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 3 or bank.shape[2] != 40: raise ValueError("bank must be [n,E,40]")
    geometry = bank if grouping_bank is None else np.asarray(grouping_bank, dtype=np.float64)
    if geometry.ndim != 3 or geometry.shape[0] != len(bank) or geometry.shape[1] < 1 or geometry.shape[2] != 40:
        raise ValueError("grouping_bank must be [n,G,40] with the same rows")
    if not np.isfinite(geometry).all() or np.any(geometry < 0):
        raise ValueError("grouping bank must be finite and nonnegative")
    grouping_probability = geometry.mean(1)
    n, e, _ = bank.shape; own = np.sqrt(np.clip(bank, 0, None)).reshape(n, -1)
    base = grouping_probability.argmax(1); one = np.eye(40, dtype=np.float32)[base]
    aggregate = own.copy(); aggregate_base = one.copy(); peer_count = np.zeros(n, np.float32); audit={"users_used":False,"include_peers":include_peers,"groups":0,"peer_rows":0}
    if include_peers:
        sessions, sa = build_safe_sessions(metadata, 30.0)
        groups = _clusters(sessions, grouping_probability, base, metadata,
                           _timestamp_tie_mask(metadata)); audit.update(sa, groups=len(groups))
        for group in groups:
            ref = max(group, key=len); aligned={int(x):[int(x)] for x in np.concatenate(group)}
            for other in group:
                if other is ref: continue
                pairs,_ = _align(grouping_probability[ref], grouping_probability[other], CONFIG["alignment_gap"])
                for i,j in pairs: aligned[int(ref[i])].append(int(other[j])); aligned[int(other[j])].append(int(ref[i]))
            for row, peers in aligned.items():
                if len(peers)>1: aggregate[row]=own[peers].mean(0); aggregate_base[row]=one[peers].mean(0); peer_count[row]=len(peers)-1
        audit["peer_rows"] = int(np.sum(peer_count > 0))
    return np.concatenate((own, aggregate, aggregate_base, peer_count[:,None]), 1).astype(np.float32), audit


def fit_source_transition(train_y, train_meta: RecordingMetadata, num_classes: int = 40, trigram_backoff: float = 1.0):
    sessions,_ = build_safe_sessions(train_meta, 30.0); y=np.asarray(train_y, dtype=np.int64); a=.25
    start=np.full(num_classes,a); end=np.full(num_classes,a); bi=np.full((num_classes,num_classes),a); tri=np.zeros((num_classes,num_classes,num_classes))
    for s in sessions:
        q=y[s]
        if len(q): start[q[0]]+=1; end[q[-1]]+=1
        if len(q)>1: np.add.at(bi,(q[:-1],q[1:]),1)
        if len(q)>2: np.add.at(tri,(q[:-2],q[1:-1],q[2:]),1)
    bp=bi/bi.sum(1,keepdims=True); total=tri.sum(2,keepdims=True); tm=np.divide(tri,total,out=np.zeros_like(tri),where=total>0); tp=(total/(total+trigram_backoff))*tm+(trigram_backoff/(total+trigram_backoff))*bp[None]
    return TransitionModel(np.log(start/start.sum()),np.log(end/end.sum()),np.log(np.maximum(bp,1e-300)),np.log(np.maximum(tp,1e-300)))


def decode_target(probability, target_meta: RecordingMetadata, transition: TransitionModel):
    p=np.asarray(probability,dtype=np.float64); out=p.argmax(1).astype(np.int64); sessions,audit=build_safe_sessions(target_meta,30.0); adjusted=.65*np.clip(p,1e-12,None); adjusted += .35*np.eye(40)[out]; adjusted/=adjusted.sum(1,keepdims=True)
    for s in sessions:
        if len(s)>40: continue
        # portable greedy backoff: transition-aware, with unique-label constraint.
        used=set()
        for t,row in enumerate(s):
            score=np.log(adjusted[row]);
            if t: score += .45*(transition.bigram_log_probability[out[s[t-1]]] if t==1 else transition.trigram_log_probability[out[s[t-2]],out[s[t-1]]])
            score[list(used)] = -np.inf
            out[row]=int(np.argmax(score)); used.add(int(out[row]))
    audit.update({"transition_weight":.45,"emission_weight":.65,"own_weight":.35,"fallback_sessions_over_40":int(sum(len(s)>40 for s in sessions))})
    return out, audit
