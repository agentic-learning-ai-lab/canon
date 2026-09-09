# MetaWorld ablation eval-camera configs

Alternate rollout-camera sets used only for the paper's **appendix ablation analyses** (not the
headline numbers). The main evaluation uses the camera pool in the released dataset's
`metaworld/policy/camera_configs.json` (6 in-distribution + 4 OOD azimuths {0,170,125,310}). Each
file here follows the same schema as that config: `list[{id, azimuth, elevation, distance, lookat, ...}]`.

## Files

### `interp_extrap_redesign.json` — interpolation/extrapolation OOD split
4 extrapolation cameras sit in the azimuth gaps not covered by any SSL-pretraining camera (front
[45°,135°], back [225°,315°]), at the gaps' interior quartile azimuths (67.5°/112.5°/247.5°/292.5°),
with elevation/distance spanning the pretraining pool's own range. 4 interpolation cameras linearly
interpolate (azimuth, elevation, distance together) between two training-cluster corner cameras
(136°→219°) at 20/40/60/80%. 4 tasks × 8 cameras × 30 trials/goal, eval-only on existing checkpoints.

### `interp_extrap_aggressive.json` — aggressive extrapolation stress test
Companion to the above: pushes elevation genuinely outside the pretraining pool's range (-56.3° to
-21.7°) at the same gap-center-adjacent azimuths, and fixes distance at 1.75m (beyond the pretraining
max of 1.49m) rather than also varying it — closer distances at these azimuths put the robot arm's
own structure in the frame regardless of elevation. 4 cameras × 4 tasks × 30 trials/goal, eval-only.
Labeled explicitly as a stress test, not the main fair-OOD result.

### `dist_ood.json` — SO(3)-7D vs SO(3)-6D distance-robustness ablation
Same 10 azimuths as the main eval, but 4 of them (az 8/35/136/180) are pushed to camera distances
outside the SSL pretraining range [1.10, 1.49] m (0.85 / 0.95 / 1.65 / 1.85 m); the other 6 stay
in-range. Isolates whether the SO(3)-7D variant's extra scale degree of freedom (`log(d_tgt/d_src)`,
not full SE(3) translation) buys any robustness when camera distance goes out of distribution, versus
the 6D method which has no distance term. Eval-only, reloading the same fixed cam-6-trained BC policy
for both encoders so rollout cameras are independent of BC training; 4 tasks × 300 trials.

| encoder | ID-distance (6 cams) | OOD-distance (4 cams) | overall (10 cams) |
|---------|:--:|:--:|:--:|
| 6D | 0.556 | 0.602 | 0.574 |
| 7D | 0.610 | 0.631 | 0.618 |
| 7D − 6D | +0.054 | +0.029 | +0.044 |

The distance term buys no distance-specific robustness: the OOD-distance gap (+0.029) is smaller
than the ID-distance gap (+0.054), and both are within noise of each other. A 0.85→1.85 m (~2×) range
is within CNN scale-tolerance and these tasks are distance-tolerant regardless of encoder.

### `dist_vary.json` — in-distribution distance-varying precursor
Same 10 azimuths, distances spread within the SSL range 1.10–1.49 m rather than pushed OOD; the
milder version of the distance test above. Distance variation barely moves either encoder's success
rate within this range. Superseded by `dist_ood.json`; kept for completeness.

### `in_support_sslpool.json` — on-support vs. off-support ground-truth-rotation analysis
The 20 SSL-training-pool azimuths, evaluated on the encoder's rotation support (the exact
camera-to-canonical rotations seen during SSL forward-dynamics pretraining). Used to show that the
ground-truth geometric warp matches or slightly exceeds the learned predicted view-conditioning code
when evaluated on-support, yet falls below it at off-support deployment cameras — evidence the
predicted rotation functions as a self-calibrated conditioning code, not raw metric geometry. The
main eval's deployment cameras are all off-support, by contrast.

### `interp_extrap_aggressive_v2.json` — aggressive stress test, distance also varied
Wider variant of the stress test above: instead of fixing distance at 1.75m and varying only
azimuth/elevation, this set also varies distance per camera, spanning 0.8–3.0 m (both closer and
farther than the pretraining range of 1.10–1.49 m).
