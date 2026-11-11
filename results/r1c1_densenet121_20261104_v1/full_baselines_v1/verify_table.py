"""Independent standard-library AUROC/AP reconstruction for all table entries."""
import bisect
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path


def metrics(rows, method):
    negative = sorted(r[method] for r in rows if not r["inconsistent"])
    positive = [r[method] for r in rows if r["inconsistent"]]
    wins = sum(bisect.bisect_left(negative, value)
               + .5 * (bisect.bisect_right(negative, value) - bisect.bisect_left(negative, value))
               for value in positive)
    auroc = wins / (len(positive) * len(negative))
    ordered = sorted(rows, key=lambda r: r[method], reverse=True)
    true_positive, total, ap = 0, 0, 0.
    for _, group in itertools.groupby(ordered, key=lambda r: r[method]):
        group = list(group)
        new_positive = sum(r["inconsistent"] for r in group)
        true_positive += new_positive
        total += len(group)
        ap += (new_positive / len(positive)) * (true_positive / total)
    return auroc, ap


def main():
    root = Path(__file__).resolve().parent
    summaries = json.loads((root / "summary.json").read_text())
    manifest = json.loads((root / "DOWNLOAD_HASHES.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    csv_rows = list(csv.DictReader((root / "table_rq1_4_with_ci.csv").open(encoding="utf-8")))
    assert len(csv_rows) == 40
    maximum_error = 0.
    for summary in summaries:
        rows = [json.loads(line) for line in (root / (summary["run"] + "_scores.jsonl")).read_text().splitlines()]
        assert len(rows) == summary["valid_pairs"]
        assert len({(r["source_id"], r["background"]) for r in rows}) == len(rows)
        for method, estimates in summary["ranking"].items():
            assert all(math.isfinite(r[method]) for r in rows)
            for metric, observed in zip(("auroc", "ap"), metrics(rows, method)):
                error = abs(observed - estimates[metric]); maximum_error = max(error, maximum_error)
                assert error < 2e-13, (summary["dataset"], method, metric, error)
                csv_row = next(r for r in csv_rows if r["dataset"] == summary["dataset"]
                               and r["measure"] == method and r["metric"] == metric)
                assert float(csv_row["estimate"]) == estimates[metric]
        for metric in ("auroc", "ap"):
            assert summary["ranking"]["H-RCAM"][metric] == max(r[metric] for r in summary["ranking"].values())
    result = {"state": "passed", "entries_checked": 40,
        "independent_methods": ["AUROC via sorted negative scores and half-credit ties",
                                "AP via cumulative threshold groups, including ties"],
        "maximum_numerical_error": maximum_error, "csv_entries_equal_summary": True,
        "download_hashes_verified": True, "H_RCAM_has_largest_point_estimates_all_four_columns": True}
    (root / "TABLE_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
