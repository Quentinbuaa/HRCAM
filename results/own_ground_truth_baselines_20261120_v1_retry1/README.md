# Final Table 9: attribution-target sensitivity

This directory contains the derived data for Table 9 (RQ3.3) of the final
Entropy proof. The two Grad-CAM conventions are own-predicted class and fixed
ground-truth class. Each convention uses the same valid pairs within each
dataset: 2979 pairs from 993 Oxford-IIIT Pet sources and 2889 pairs from 963
CUB-200-2011 sources. Each source contributes black, gray and white
background-replacement pairs.

The JSONL files give each pair's source identifier, replacement background,
prediction-inconsistency outcome, and scores for all 13 measures under both
conventions. The JSON summary contains AUROC, average precision, confidence
intervals, paired differences and input hashes. The saved intervals use 1000
source-cluster bootstrap draws (seed 20261011).

The archived three-convention diagnostic used 2887 CUB pairs because two
fixed-source maps were invalid. The final two-convention comparison uses the
valid own-predicted and fixed-ground-truth records for those two pairs.

From the repository root, run

```bash
python verify_revision_release.py
```

to recompute the point estimates from the released scores.
