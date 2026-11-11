# R1C6: bounded attribution-method/layer sensitivity check

The author authorized this experiment if it is straightforward to implement.
Inspection found that one shared forward/backward pass can provide two CAM
methods at two layers, without training, dependencies, or new data. Execute
this bounded study. Placement/RQ numbering and manuscript edits remain for
later author review.

## Prespecified scope

Use original `pets_v1` and `cub_v1` fitted ResNet-18 classifiers, exact original
holdout IDs/order/batches of 16 (including final short batches), supplied masks,
ImageNet normalization, and black/gray/white follow-ups. No architecture run,
training, recalibration, layer search, smoothing/fusion, or favorable-result
selection. Retain every condition, including weaker results and invalid maps.

Four fixed configurations form a 2x2 comparison:

1. Grad-CAM at `layer4` (original, 7x7 activated spatial output).
2. Grad-CAM at `layer3[0]` (earlier 14x14 block output).
3. LayerCAM at `layer4`.
4. LayerCAM at `layer3[0]`.

The earlier layer is `layer3[-2]` in the two-block ResNet-18 stage, exactly
`layer3[0]`. This follows the LayerCAM authors' advice to avoid selecting a
block output immediately followed by a strided convolution, which can yield
grid artifacts. It is selected before any outcomes, not by its performance.
The unchanged original layer4 remains the main study setting.

For activation A and target-logit gradient G:
- Grad-CAM native map = ReLU(sum_channels(mean_spatial(G) * A)).
- LayerCAM native map = ReLU(sum_channels(ReLU(G) * A)).

Use logit targets, no eigen/augmentation smoothing, no cross-layer fusion.
LayerCAM's local positive-gradient weighting differs from Grad-CAM's global
channel weighting. Upsample each native map to 224x224 with bilinear
interpolation (align_corners=False), then subtract its minimum and divide by
its maximum clamped to 1e-12, exactly as in the original pipeline.

Save full-precision native maps (small 7x7/14x14 arrays), not duplicate 224x224
arrays. Reconstruct the full-resolution maps deterministically for scores and
audit. Own-class is the primary comparison; retain fixed-source targeting as
a diagnostic. Original own/fixed-source Grad-CAM-layer4 maps and scores must
reproduce the archived values in the original batch contexts.

## Analysis

For each method/layer/target, apply original map-validity rules and original
mask eligibility (.05-.95 foreground fraction). Keep all records and disclose
invalid maps. Outcomes/predictions are shared across configurations and checked
against the original records.

Report per-configuration all-valid counts/AUROC/AP, plus comparisons on common
eligible pairs valid across all four configurations, separately by target
convention. The original Table 3 is unchanged. Include source-cluster CIs
(1000 resamples, seed 20261105) for H-RCAM AUROC/AP and paired differences
against Grad-CAM-layer4, preserving all probes of each sampled source together.
Compute JSD/NCD point diagnostics as well. Report Spearman rank agreement and
absolute score differences on common pairs, since similar discrimination does
not imply identical scores or rankings. These descriptive correlations do not
replace the separate R2C4 component-redundancy study.

This is evidence about sensitivity within two methods/two layers on Pet/CUB,
not invariance across arbitrary explanation methods/layers. Do not choose the
method/layer with the highest observed AUROC as a new default after evaluation.

## Audit and execution

Independently replay native maps using NumPy reference weighting and identical
batch context; verify layer4 gradients against the GAP+linear-head closed
form. Reconstruct full maps and verify valid score components with SciPy.
Independently reconstruct table metrics and core intervals with pure-Python
rank counts. Check original files and manuscript hashes are unchanged.

Use remote-5080 through Tailscale 100.74.145.98. Every remote Python process
starts with `taskset -c 24-31`; BLAS/PyTorch threads four. New isolated output
only, frozen protocol/code before extraction. No environment modifications.

Primary references inspected before implementation:
- Jiang et al., *LayerCAM: Exploring Hierarchical Class Activation Maps for
  Localization*, IEEE TIP 2021: https://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf
- Authors' implementation and layer-choice guidance:
  https://github.com/PengtaoJiang/LayerCAM-jittor
- PyTorch reference formula:
  https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/layer_cam.py

This authorization does not start R2C4 L1/L2/SSIM, R1C3 predictive validation,
or the detailed R1C2 root-cause study. They remain separate items.
