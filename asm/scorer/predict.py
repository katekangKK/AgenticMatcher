"""Pretrained scoring with stable source aggregation and preserved candidate order."""
from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np

from asm.scorer.features import COLUMNS, eligibility, extract, project


SOURCE_AGGREGATION_VERSION = "source_shifted_fsum_v2"


def source_statistics(values, top_position):
    """Population statistics, preserving real variation without epsilon clipping.

    Sum offsets from the column minimum with fsum. Exactly constant columns
    have an exact zero variance; small genuine variation survives. Compensated
    sums also avoid candidate row-order dependent accumulation.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Source statistics require a nonempty finite matrix")
    mean, std = [], []
    for column in values.T:
        anchor = float(column.min())
        offsets = [float(v) - anchor for v in column]
        center = math.fsum(offsets) / len(offsets)
        mean.append(anchor + center)
        std.append(math.sqrt(math.fsum((v - center) ** 2 for v in offsets) / len(offsets)))
    return {"top": values[top_position], "mean": np.asarray(mean),
            "max": values.max(axis=0), "std": np.asarray(std)}


class Scorer:
    def __init__(self, root=None):
        root = Path(root) if root is not None else Path(__file__).parent.parent / "models" / "latest"
        self.manifest = json.loads((root / "bundle.json").read_text())
        if list(COLUMNS) != self.manifest["candidate_features"]:
            raise ValueError("Scorer feature schema does not match this runtime.")
        for name in ("source.joblib", "candidate.joblib"):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != self.manifest["files"][name]:
                raise ValueError("Model checksum mismatch: " + name)
        # joblib/pickle models must come from a trusted source.
        self.source = joblib.load(root / "source.joblib")["model"]
        self.candidate = joblib.load(root / "candidate.joblib")["model"]
        if self.source.n_features_in_ != 328 or self.candidate.n_features_in_ != 82:
            raise ValueError("Unexpected model feature dimensions.")

    def score(self, snapshot, final_candidates):
        original = [(name, float(value)) for name, value in final_candidates]
        names = [name for name, _ in original]
        if any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names)):
            raise ValueError("Final target names must be nonempty and unique.")
        if any(not np.isfinite(v) or not 0 <= v <= 1 for _, v in original):
            raise ValueError("Final scores must be finite and in [0, 1].")
        def unchanged(reason):
            return {"applied": False, "reason": reason, "candidates": original}
        if not original:
            return unchanged("no_match")
        try:
            allowed, reason = eligibility(snapshot, names)
            if not allowed:
                return unchanged(reason)
            rows = extract(snapshot)
            pool = [row["candidate_id"] for row in rows]
            x = np.asarray([[row["features"][f] for f in COLUMNS] for row in rows], dtype=float)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite features")
            top = next((i for i, row in enumerate(rows) if row["features"]["core_rank"] == 1), 0)
            stats = source_statistics(x, top)
            vector = np.concatenate([stats[a] for a in ("top", "mean", "max", "std")])
            positions = {name: i for i, name in enumerate(pool)}
            indices = [positions[name] for name in names]
        except (ValueError, TypeError, KeyError, IndexError, AssertionError, OverflowError, AttributeError):
            return unchanged("invalid_pre_submit_snapshot")
        gamma = float(self.source.predict_proba(vector.reshape(1, -1))[0, 1])
        beta = self.candidate.predict_proba(x)[:, 1]
        values = project([gamma * beta[i] for i in indices])
        return {"applied": True, "reason": "eligible",
                "candidates": [(name, float(v)) for name, v in zip(names, values)]}
