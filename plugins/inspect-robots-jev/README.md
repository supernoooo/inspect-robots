# inspect-robots-jev

The plugin provides `jev-direct` and `jev-hybrid` entry points for the
real `yam_arms` embodiment and the optional `isaacsim-2yam` simulation profile.
Both use the Batch 02 vision service, Batch 03/04
geometry and safe candidates, and a pinned Jev Choice request to select a short
trajectory. `jev-hybrid` adds one checked MolmoAct2 YAM prefix. The action
order is left j0–j5, left gripper, right j0–j5, right gripper.
Grippers use `0=closed, 1=open`, matching the embodiment contract.

`AgentCandidateValidator` is the separate Batch 03 local bridge from a
`ProposalBatch` to `ValidatedCandidateSet(available, hold, filtered,
observation_reason)`. Each available candidate carries only a checked prefix
of at most three absolute joint targets and a JSON-safe `jev_summary()`.
The validator reads the actual rig's frozen YAM config and declaration. Both
validation modes check the full Agent expansion against the configured bounds
and step limits before discarding the suffix. Measured mode additionally
requires calibrated geometry and checks the full trajectory with the MJCF
motion geometry and YAM collision model. Its summary separates model
intent from computed joint, gripper, and end-effector changes and records model
and calibration fingerprints. Unknown objects are outside these local collision
checks, as are finger contacts during gripper closure. A bad image/time yields an independent one-step hold; an invalid joint
measurement raises `InputError`. Constructing or calling the validator does not
invoke JEV, `eval`, `reset`, or a motor driver.

## Batch 04 jev-agent policy

`jev-agent` binds to a real YAM declaration in `pairing_preflight`, then uses
the current observation to request three GPT-6 Astra Responses proposals. It
filters every proposal through `AgentCandidateValidator` and offers only the
verified short prefixes plus `hold` to JEV. `selector=agent_preferred` is an
offline comparison mode using the same filter. The policy returns at most three
joint targets and replans after the controller consumes them. The Agent's
preferred ID is never sent to JEV.
The Agent also reports a task phase, qualitative gripper-to-object relation,
object state, and the expected visible change for each candidate. These are
unverified image assessments. It also reports normalized target, gripper, and
placement points plus candidate `du/dv` predictions. Jev receives only these
bounded 2-D fields, derived image errors, and the previous observed execution
result; it never receives RGB, absolute joint targets, or trajectories. An
unmoved object is expected during approach: progress comes from reduced
gripper-to-object image error. The policy retains twelve observed transitions
per camera/arm and, after three samples, fits a regularized local joint-to-image
mapping for direction consistency checks. It never performs calibration probes.
Three consecutive Jev holds or three observed movement rounds without visual progress request
operator review instead of another automatic choice.

The default `validation_mode=yam` uses the normal Agent-style rollout without
calibration or MJCF, or configure `calibration_path` and `mjcf_path` for
`validation_mode=measured`. Set `max_dispatch_age_s` for either mode. Without
that decision-time threshold,
the policy returns a one-step hold. `inference_budget_s`, `agent_timeout_s`,
and `jev_timeout_s` bound and report service time. A decision that outlives its
budget also holds. Invalid joint state raises `InputError`
and dispatches nothing. `done` requires two consecutive fresh observations and
`give_up` requires three; earlier reports hold and
reobserve. A confirmed result uses the rollout's stop request and never claims
task success. `audit_records` and the returned chunk metadata
include the decision ID, proposed/filtered/selected IDs, and selected prefix.
They describe a dispatch request, not motor execution; the latter belongs to
the next observation and controller record. No real robot motion is accepted by
these offline tests.

The normal `inspect-robots run` CLI also accepts `--policy jev-agent` with the
same `--instruction`, `--embodiment yam_arms`, `--max-steps`, `--scorer`,
`--rerun`, `--store-frames`, and `--log-dir` options used for `agent`.
`model=openai/gpt-6-astra`, `wire=responses`, `effort=medium`,
`max_llm_calls`, and `max_speed_frac` can be passed through `-P`. The validator
expands proposals using the paired YAM action space, including its declared
gripper step limit. Images are always supplied to the proposal model, so
`images=always` is implicit. `image_horizon` is not a `jev-agent` option: its
proposal API sends the current observation only. Measured mode additionally
needs the calibration and MJCF paths, matching measured collision base poses
and table height in the YAM configuration. Those inputs serve the candidate
geometry checks; a successful
plain `agent` run does not populate them.

`validation_mode=measured` requires the measured calibration and MJCF and checks
the full proposed trajectory and interpolated collisions before Jev selects a
movement prefix. `validation_mode=yam` needs neither file: it checks the
expanded candidates against YAM joint bounds and step limits, then lets Jev
select a short prefix for the normal Inspect Robots rollout. The selected
actions pass through the rollout's configured approver chain, just as with
`agent`; the full trajectory and interpolated collision checks remain
`not_checked` in the audit. The YAM collision approver is present only when
enabled and available in the embodiment configuration. Both modes require an
explicit `max_dispatch_age_s`. By default, `freshness_mode=frame_sequence` uses YAM's
three camera publication IDs and a new joint sample. The first observation
sets a baseline and holds; the next observation must have newer IDs from the
same camera generation. Capture timestamps remain finite audit fields but do
not impose an age or skew gate. The dispatch limit starts at decision start.
`freshness_mode=capture_age` retains the earlier time-based behavior for
offline compatibility.

For real YAM, install the patched YAM source checkout from Batch 02 in editable
mode alongside this package and Inspect Robots. Its local version is
`0.36.1.dev0+...` (based on tag `v0.36.0`); the unpatched `0.36.0` wheel lacks
capture timestamps. Isaac packages and simulation entry points are not
required. Set `cam_height` and `cam_width` on the JEV policy to the rig's
actual output dimensions. The old `360x640` default belongs to the simulation
profile and fails strict pairing against the currently configured `224x224`
real rig. The policy requires three `uint8` RGB
images, finite in-range 14-D `joint_pos`, a nonempty
instruction, three `image_times`, `state_time`, and by default
`Observation.extra.camera_frame_ids`. `Observation.extra.approvals` and
`env_step` provide controller feedback for the preceding decision; other extra fields,
historical simulator truth, rewards, and operator verdicts never enter model
decisions. A recoverable
input fault produces one absolute hold target copied from the valid 14-D
current state; a missing or invalid state raises `InputError` and blocks motion.
The real pairing gate also checks the action order, bounds, gripper encoding,
absolute joint mode, camera names and output dimensions before `reset`.

The YAM capture thread or process stamps each frame at local receive/publication
time and transfers its RGB, stamp, and frame ID in one snapshot. The embodiment
stamps the validated encoder read immediately. A legacy injected `ImageMap`
has no image timestamps and makes JEV hold. Missing or nonfinite timestamps
still hold because audit pairing requires them. Their absolute age and skew
do not gate the default frame-sequence workflow. Missing, repeated, or
restarted camera IDs also hold. These timestamps describe assembly time,
not camera exposure time. The optional legacy `capture_age` mode applies
`max_image_age_s` and `max_skew_s` when needed for older offline fixtures.

```bash
python -m pytest -q plugins/inspect-robots-jev/tests/test_contract.py \
  plugins/inspect-robots-isaacsim-2yam/tests/test_isaacsim_2yam.py
```

This offline test does not require Isaac Sim or a model service.

## Batch 05 jev-agent execution audit

Each `jev-agent` call has an episode-scoped `decision_id`. A sidecar at
`jev-audit/<episode_id>/<decision_id>.json` records the task, model and rig
fingerprints, input joint readback and image times, every proposal and local
filter result, selected short prefix, JEV selection/probabilities, Agent and
JEV usage, and segment/total inference time. `dispatch_status` describes only
the policy's request. Every candidate remains `proposed` in the record;
`selected_for_dispatch` identifies the one prefix returned to the controller.

On the next fresh, valid observation, the previous row gains `observed_state`
(joint readback, delta, image times) and `approval` from the rollout's
`Observation.extra.approvals`. Mismatched decision IDs or step numbers mark
`audit_incomplete`. `execution.status=observed_after_dispatch` means a later
measurement arrived; it does not claim motors followed the prefix, an object
moved, or the task succeeded. If the trial ends first, the status is
`unverified_no_next_observation`. Holds, errors, and `done`/`give_up` have their
own reasons and never become task success. Only the operator verdict in the
trial record determines completion. An intervening stale hold does not discard
an older dispatch still awaiting a fresh observation.

The next Agent request receives a concise caller report with the prior choice,
approval flags, measured joint change, and uncertainty. When YAM reports a
clamp or rewrite, that report explicitly says the requested prefix may differ
from the controller action. At trial end, `controller_trace.reviewed_prefix`
compares any actions recorded after approval with the requested prefix;
recording a reviewed action is still not proof of physical execution.

The trial's `policy_transcript` contains bounded rows, and
`trial_metadata.jev_audit` indexes sidecars, missing/invalid files, missing
frame references, write failures, unverified decisions, and total trial time.
Rows contain frame paths and times, never raw RGB. Configured API keys are
redacted. Frame logging must be enabled for complete image references.

```bash
.venv/bin/python -m pytest -q plugins/inspect-robots-jev/tests/test_jev_agent_audit.py \
  plugins/inspect-robots-jev/tests/test_jev_agent_policy.py \
  plugins/inspect-robots-agent/tests/test_policy_e2e.py
```

## Batch 05 direct branch

`jev-direct` requires explicit `calibration_path` and `mjcf_path` files, each
validated at construction. Its stable configuration names are `vision_url`,
`calibration_path`, `mjcf_path`, `jev_model`, and `max_image_age_s`.
`jev_model` defaults to the pinned `jev-1.13.0`; aliases are rejected. The
vision endpoint defaults to `http://127.0.0.1:8765/v1/detect`. The Jev endpoint
can be overridden with `jev_url` (default
`https://api.typesafe.ai/v1/systemone`) and bounded with `jev_timeout_s`.
Only `TYPESAFE_API_KEY` is read for authentication. No key or raw observation
is included in the Choice JSON or the policy audit. `audit_records` and
`transcript()` expose JSON-safe detections, candidate/filter decisions,
probabilities (or an explicit missing marker), the selected trajectory, model,
hold reason, and latency. Every motion chunk ends with a new observation.

```bash
python -m pytest -q plugins/inspect-robots-jev/tests/test_jev_choice.py \
  plugins/inspect-robots-jev/tests/test_direct.py
```

These tests use fake HTTP services; they do not call the online Jev API.

## Generic Choice for checked action proposals

`JevChoiceClient.choose_generic(instruction=..., observation_context=...,
candidates=...)` accepts a nonempty text context and a sequence of
`ChoiceOption(id, summary)` values. IDs must be unique and there may be at most
255 options. Each summary may be text, a finite number, or a small nested map
of text and numeric values. The caller must form summaries only after its
safety filter. The generic path rejects binary/image data, arrays, unrestricted
coordinate fields, credential fields, and Agent `preferred_id` fields. It
permits only the normalized `u/v`, predicted `du/dv`, confidence, and
derived-error keys used by the RGB loop. It sends no Agent preference. Keep raw
targets and trajectories outside this request. For a single checked action,
include an explicit `hold` option.

The result gives `selected_id`, optional `probabilities`, pinned `model`,
`latency_s`, and the service's optional `usage` map. Invalid candidates,
unknown IDs, invalid probabilities, timeouts, and authentication failures raise
`ChoiceError` with a safe code. The caller decides whether to hold. The
original `choose(instruction=..., candidates=CandidateSet)` path still sends
the same stage and `jev_summary()` criteria for `jev-direct` and `jev-hybrid`.
An injected `transport(request, *, timeout)` can be used for offline tests.

## Batch 06 hybrid branch

`jev-hybrid` accepts the same vision, calibration, MJCF, Jev, and image-age
parameters as `jev-direct`, plus `molmo_url` (default
`http://127.0.0.1:8202`) and `molmo_prefix_steps` (default `3`, bounded by
the shared motion planner's six-step horizon). `molmo_timeout_s` bounds the
HTTP call. It uses MolmoAct2's YAM `json_numpy` wire format: three RGB arrays,
a 14-D float32 `state`, the instruction, timestamp, and ten denoising steps.
One POST inference is made for each fresh, usable observation. A duplicate or
stale observation makes no POST.

Before its first inference, the client reads `GET /act` and requires the YAM
repository and norm tag, 3 cameras, a 14-D state, and a pinned checkpoint plus
resolved hexadecimal revision. A server exposing only the upstream health
fields without checkpoint/revision yields `molmo_invalid_identity` and holds;
the server must expose those fields as described in the `isaacsim-2yam` setup.
The YAM service's camera and action order follows its fixed `top, left, right`
and `[left j0..j5, gripper, right j0..j5, gripper]` contract. If it advertises
`action_order`, that field is checked too.

The complete response is checked for shape, finite values, joint limits, and
gripper encoding. Only the configured short prefix enters the shared per-step
collision checker and Jev candidate set. Jev may choose this prefix or one of
the current EEF corrections. The unused Molmo suffix is discarded; another
decision requires new images and joint state. Invalid responses, unsafe
prefixes, vision failures, and Choice failures produce a one-step hold with a
specific reason. The audit stores checkpoint/revision, candidate source and
span, Choice probabilities, both inference latencies, and the selected short
trajectory. It does not store authentication headers or raw images.

```bash
python -m pytest -q plugins/inspect-robots-jev/tests/test_molmo_client.py \
  plugins/inspect-robots-jev/tests/test_hybrid.py
```

These tests use local fake `/act`, vision, and Choice transports. Live model,
calibration, Isaac, and collision fidelity are not established by this batch.

## Batch 06 offline jev-agent replay

`scripts/replay_jev_agent.py` reads a manifest of three saved RGB files per
round (PNG/JPEG RGB or uint8 H×W×3 `.npy`), their individual monotonic capture
times, a 14-value joint readback and its time, the task instruction, and
optional approval reports. `replay_at` is the saved decision time; the runner
rebases each capture time onto an injectable monotonic clock while preserving
its age. The `rig` object is a `YamConfig` declaration. Calibration and MJCF
files supply local geometry. The CLI starts with only YAM's pure-data modules
loaded and never creates a YAM embodiment, opens CAN, resets, checks health,
or sends a chunk. Use a fresh Python process for the CLI.
PNG/JPEG decoding requires Pillow; `.npy` replay uses NumPy alone.

The runnable [synthetic manifest](examples/replay-manifest.example.json) has
one valid selection and a second fresh observation that replans. Create its
saved RGB files and run the deterministic fixture mode:

```bash
.venv/bin/python plugins/inspect-robots-jev/scripts/create_replay_example.py /tmp/jev-replay-demo
.venv/bin/python plugins/inspect-robots-jev/scripts/replay_jev_agent.py \
  /tmp/jev-replay-demo/manifest.json --out /tmp/jev-replay-demo/log
```

The fixture is synthetic; for recorded YAM data, replace its RGB paths,
calibration, model, rig values, timestamps, instructions, and measured joint
readbacks with the saved capture. `agent` candidates and `jev_id` provide
repeatable fixture responses. Omit those fixture fields and add
`--live-services` to call the configured Agent and JEV services and record
their latency and token usage. That mode still produces only offline proposals.

Example CLI output (two decisions, shortened to the decision fields):

```json
{"proposed_only":true,"decisions":[{"proposed_only":true,"reason":null,"agent_selection":"move","jev_selection":"move"},{"proposed_only":true,"reason":null,"agent_selection":"next","jev_selection":"next"}]}
```

Full Batch 05-compatible JSON sidecars appear at
`log/jev-audit/<episode_id>/<decision_id>.json` and can be read with
`inspect_robots_jev.audit.load_episode`; `log/replay-index.json` supplies the
episode ID, decision count, and explicit offline marker. Every row and selectable candidate is marked
`proposed_only=true`; `selected_for_dispatch` remains null and
`dispatch_status=proposed_only` denotes a hypothetical prefix. A later joint
readback is a comparison to that prefix, never evidence of motor execution.
Audit JSON contains image references, not image bytes, and configured provider
secrets are redacted. The replay does not claim field shadow validation.

## Batch 07 live shadow API

`inspect_robots_jev.shadow.ShadowRunner` accepts an externally owned
`ReadOnlySource` with separate `read_cameras()` and `read_joints()` methods.
`CameraRead` supplies exactly `top_cam`, `left_cam`, and `right_cam` RGB arrays,
per-frame monotonic capture times and frame IDs; `JointRead` supplies a 14-value
absolute joint readback and its monotonic time. All timestamps must use the same
clock domain as the runner. Construct the validator from the actual measured
rig configuration, calibration and MJCF, then provide an `AgentProposer` and
`JevChoiceClient`. Call `run_round(instruction, scene_change="unchanged")` only
when the operator has checked the scene; `changed` and `unknown` record holds.
Each round writes `jev-audit/` JSON and three `shadow-frames/` NumPy files.
`summary()` returns p95/max capture, Agent, filter, JEV, total and observation
age, plus the over-age hold rate and scene-change counts.

The field CLI `scripts/shadow_jev_agent.py` loads a JSON manifest containing
`rig` (YamConfig fields), `calibration`, `mjcf`, `instruction`, signed
`max_image_age_s`, `max_skew_s`, and `max_dispatch_age_s`. Its
`--source-factory MODULE:FUNCTION` receives the parsed `YamConfig` and must
return the audited read source; `--rounds` bounds the session. It prompts for
the operator's scene status before each round and writes
`shadow-summary.json`. It also writes `shadow-session.json` with installed
package versions, requested model IDs, source factory name, and hashes of the
manifest, rig config, calibration and MJCF. These are software fingerprints;
the field report still needs proof of the actual runtime rig config and device
firmware. Loading the source factory is a site-controlled device
operation and requires the separate command audit in the field report. The CLI
never calls its `close()` method; the source owner must document and audit
shutdown independently. No command for an unaudited hardware source is
provided here.

The runner has no embodiment, controller, `reset`, `close`, or motor-command
path. It does not connect devices. The source owner must establish and audit a
motionless connection/sampling/shutdown path before field use. YAM's default
driver may calibrate a gripper on connection; `reset()` homes and connected
`close()` may park. A `collision_guardrail=true` config declaration alone is
insufficient: verify the installed YAM approver is contributed with the actual
rig geometry and rejects injected unsafe targets without reaching a motor.
The [Batch 07 field report template](../../docs/development/jev-agent/batch/07-field-report-template.md)
is the field acceptance record. Current state is **待现场验证**.

## Batch 08 attended motion software gate

`inspect_robots_jev.staged_motion.StagedMotionGate` uses the existing
`JevAgentPolicy` decision and short prefix. The Agent proposal and local
candidate validation provide phase and visual-progress summaries; JEV chooses only among locally checked
candidate IDs and `hold`. The staged path additionally dry-runs each locally
safe prefix through a private copy of the same YAM approval chain before JEV
sees it; the live approval state changes only for the selected action. This
separate attended gate still requires `validation_mode=measured` and passing
full-trajectory and interpolated-collision checks for motion stages. For
stages 2–4 call
`prepare_from_policy(policy, observation, approver)`, inspect the new audit row, obtain
the operator and emergency-stop watcher's explicit `permit`, and execute with
the existing Inspect Robots
`DefaultController` and the actual YAM approver chain. `prepare_hold` creates
the independent current-pose hold for stage 1. The supplied embodiment remains
owned by the caller; this module has no device construction, reset, close, or
CAN interface. The lock path is an advisory cross-process guard; field staff
must also verify that only one process controls the YAM devices. Construct the
gate in a `with` block or call `close()` after the session; it holds the lock
across operator-review gaps and releases only that file lock on exit.
The gate retains the trial-local approver state across successive `execute`
calls so the YAM delta and collision checks use the last approved pose.
Supply a core `FrameStore` and unique `trial_id` when constructing the gate to
save all input and post-step camera frames; `save_record(path)` then links each
reviewed action, joint readback and frame file by decision ID.

The gate checks the Batch 07 signed record reference, rig/Agent/JEV fingerprints,
fresh three-camera plus 14-joint input, exact selected prefix and decision ID,
segment length, micro joint delta, decision elapsed time, actual YAM collision
approver presence, and post-step joint/image feedback. A modified or vetoed
approval stops before motor dispatch. Each stage needs reviewed evidence and
operator/watch signatures to advance; stage 2 checks both arms and both
grippers, stage 3 requires two safe candidates plus `hold`, and stage 4 needs
new decisions and Agent feedback IDs. An operator may call `stop(reason)` on
wrong direction or another field anomaly; `recover` requires review and a new
joint/camera observation, then clears approver state before a new decision.
Recovery invalidates every unsigned segment in the affected stage; start that
stage again from a fresh proposal and recheck the actual rig/approver state.
Inspect Robots/YAM logs and original frame files remain necessary for
field proof; the gate's in-memory evidence alone does not certify motion.

The [Batch 08 field record](BATCH08-FIELD-RECORD.md) has separate signed forms
for all four stages; the [offline record](BATCH08-OFFLINE-RECORD.md) reports the
fake-embodiment gate. Batch 07 is still **待现场验证**, so these tests do not authorize
real motion or claim that any stage has physically passed.
Call `save_record(path)` after each segment or stop to persist software evidence;
attach the original Inspect Robots action log, YAM approval events, and camera
frames to the signed field form. Missing raw evidence cannot be repaired by the
software record.

```bash
.venv/bin/python -m pytest -q plugins/inspect-robots-jev/tests/test_jev_agent_staged_motion.py \
  plugins/inspect-robots-jev/tests/test_jev_agent_policy.py
```

## Batch 07 shadow/replay regression

```bash
.venv/bin/python -m pytest -q plugins/inspect-robots-jev/tests/test_jev_agent_shadow.py \
  plugins/inspect-robots-jev/tests/test_jev_agent_replay.py
```

| Fault | Replay reason |
| --- | --- |
| Missing or invalid RGB | `missing_rgb` / `invalid_rgb` |
| Invalid, skewed, stale or future capture time | `invalid_time` / `time_skew` / `stale_time` |
| Reversed saved decision time | `replay_time_not_new` |
| Repeated timestamps / identical frames | `observation_not_new` / `duplicate_frame` |
| Agent or JEV service failure | `agent_*` / `jev_*` |
| Unknown JEV candidate ID | `jev_unknown_candidate_id` |
| Every candidate collides or is filtered | `no_safe_candidate` |
| Next joint readback differs from proposed target | `next_state_mismatch` |
| Agent requests `done` or `give_up` | stop with that reason |

```bash
.venv/bin/python -m pytest -q plugins/inspect-robots-jev/tests/test_jev_agent_replay.py \
  plugins/inspect-robots-jev/tests/test_jev_agent_audit.py
```

Local result: `37 passed` with the synthetic saved observations and fake
services. Recorded physical YAM observations were not present in this repo.

## Batch 07 offline audit and diagnosis

Each decision has an episode-scoped `decision_id` in `audit_records` and in the
trial's `policy_transcript`. The record includes allowed observation timestamps
and joint state, vision detections with boxes, mask references and scores,
localization quality, every current candidate and filtering reason, Choice
probabilities or an explicit missing marker, the selected 14-D chunk, model
versions, segment latencies, and the next observed joint state when available.
The execution status compares that next state to the planned final joint target;
it does not assert physical task success. A duplicate or stale observation does
not resolve a pending execution.

Audit rows are detached JSON capped at 32 KiB each. At most 48 rows are retained
in the inline transcript; `audit_omitted_prior` counts older rows omitted from
that inline view. During eval, every decision is also written atomically to its
own 32 KiB limited JSON sidecar at `jev-audit/<episode_id>/<decision_id>.json`.
The trial's `trial_metadata.jev_audit` index records the episode ID, decision
count, size limit, and any missing or failed writes. A later observation updates
the matching decision file with its joint-target result, even after it has
left the inline window. Large detection lists are shortened with
`detection_count` and `detections_omitted`, while candidate and filtering
decisions are kept. The API key is redacted, and no RGB or mask pixels are
embedded in audit JSON. If an audit write fails, the index reports it; the
trial should not be treated as fully traceable.

During an Inspect Robots eval run, `on_trial_start` writes compressed binary
mask sidecars under `jev-masks/`. At trial end, `on_trial_end` adds references to
the rollout's existing `frames/` files by matching the recorded observation
timestamp. The mask reference also carries the vision request ID and detection
index. Direct calls to `act()` without a trial log retain the request reference
but have no frame or mask file path.

```bash
python plugins/inspect-robots-jev/scripts/diagnose_trajectory.py path/to/eval-log.json
python plugins/inspect-robots-jev/scripts/diagnose_trajectory.py path/to/eval-log.json --decision-id ID
python -m pytest -q plugins/inspect-robots-jev/tests
python -m pytest -q plugins/inspect-robots-jev/vision_service/tests
```

The diagnostic command follows the trial index to read all sidecar decisions,
including those older than the inline window; `--decision-id` reads one bounded
record directly. It also accepts a standalone JSON transcript list. It
reports separate evidence for suspected vision error, unreachable or filtered
targets, Jev choice requiring review, and execution not reaching the planned
joint target. A Jev selection's actual correctness requires human or evaluator
review. No simulator truth, reward, or success flag is used for these labels.

## Batch 03 geometry handoff

`Calibration.load(path)` reads a strict version 1 JSON file. The committed
[`synthetic_calibration_v1.json`](tests/fixtures/synthetic_calibration_v1.json)
shows the schema; its numbers are test values, not a measured YAM calibration.
Intrinsics use pixel coordinates `(u right, v down)` and an optical frame with
`+X right, +Y down, +Z forward`. Each transform gives the optical frame in its
parent as `translation_m` and unit `rotation_wxyz`. `top_cam` is fixed under
`base`; `left_cam` and `right_cam` are fixed under their respective EEFs. The
base convention is `+X forward, +Y left, +Z up`, in metres. The table plane is
`normal_base · point = offset_m`; the normal points upward. The box opening has
known size, rim height above the table, vertical clearance, and lateral safety
margin. The caller must supply measured intrinsics and extrinsics, including
the conversion from camera mount convention to the optical convention.

`YamKinematics(path)` loads a user-provided, flattened static MJCF with a fixed
`bimanual_base` and `left_link_6` / `right_link_6` EEF bodies. Arm wire indices
`0..5` map to `left_joint1..6`, `7..12` to `right_joint1..6`; wire indices `6`
and `13` map to each side's two finger hinges at `-0.0475 * normalized_gripper`
metres/radians according to the embodiment contract. `forward(side, q)` and
`jacobian(side, q)` return results in the robot base frame. `inverse(...)`
returns `IKResult` with `joint_pos=None` on failure, plus a reason and residuals.
The MJCF SHA-256 is included in kinematic and localization results.

`localize_targets(decoded_input, outcomes, calibration, kinematics)` accepts
only the Batch 01 decoded whitelist and Batch 02 `VisionOutcome` data. It
returns one `Localization` per category, including a failure for missing or
unusable targets. `candidate_position()` returns `None` on every quality
failure. Red block and yellow ball positions are **table contact candidates**
from mask median pixels; object center height is not inferred. The box result
is an opening center above the known rim, with remaining safe half extents.
Mask width and pixel quantization give a conservative approximate position
error, not a calibrated statistical covariance. Occlusion, low confidence,
oblique rays, bad timestamps, missing FK, excessive uncertainty, duplicate
instances, and multi-view disagreement prevent a usable position. `diagnostic()`
on localization and IK results produces JSON-safe provenance and failure data.
No runtime truth pose, object state, or simulator camera pose is consulted.

```bash
python -m pytest -q plugins/inspect-robots-jev/tests/test_geometry.py
```

## Batch 08: field setup and delivery

Run these commands from the repository root. All commands in this section are
**pending user execution** on the Isaac machine; the offline tests below do not
prove live recognition or placement. Replace `/path/to/...` with real absolute
paths. The [Isaac 2YAM setup](../inspect-robots-isaacsim-2yam/README.md) and
[vision service setup](vision_service/README.md) give the adapter details.

In the Python environment that already imports Isaac Lab and Isaac Sim, install
the CLI, Rerun recording support, the embodiment, this policy plugin, and the
existing MolmoAct2 policy. `ffmpeg` must also be on `PATH` for MP4 export.

```bash
uv pip install --python /path/to/isaaclab/bin/python \
  -e '.[rerun]' -e plugins/inspect-robots-isaacsim-2yam \
  -e plugins/inspect-robots-jev -e /path/to/inspect-robots-yam
/path/to/isaaclab/bin/inspect-robots list policies
/path/to/isaaclab/bin/inspect-robots list tasks
```

Confirm that `molmoact2`, `jev-direct`, `jev-hybrid`, and
`isaacsim-2yam-put-everything-in-box` appear. Install the visual service in a
separate Python environment; its dependency on the embodiment is only for the
YAM contract and does not load Isaac Lab:

```bash
uv venv /path/to/jev-vision-venv
uv pip install --python /path/to/jev-vision-venv/bin/python \
  -e . -e plugins/inspect-robots-isaacsim-2yam \
  -e plugins/inspect-robots-jev -e plugins/inspect-robots-jev/vision_service
```

The service uses [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
(`IDEA-Research/grounding-dino-tiny`, snapshot
`a2bb814dd30d776dcf7e30523b00659f4f141c71`) for boxes and
[SAM 2.1 Hiera tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny)
(`facebook/sam2.1-hiera-tiny`, snapshot
`de431c4043854a71d8101e17995dfe596bf101a5`) for masks. Those immutable
revisions include Transformers `model.safetensors` weights. On first start,
`snapshot_download` fetches JSON, safetensors, and TXT files into the Hugging
Face cache; use `--local-files-only` on later starts to require cached files.
The service prints and returns the resolved revisions. No weights are bundled
here, and none were downloaded for this handoff.

```bash
/path/to/jev-vision-venv/bin/inspect-robots-jev-vision \
  --host 127.0.0.1 --port 8765 \
  --detector-revision a2bb814dd30d776dcf7e30523b00659f4f141c71 \
  --segmenter-revision de431c4043854a71d8101e17995dfe596bf101a5
```

Provide the flattened YAM MJCF and all referenced meshes from the MolmoAct2
asset checkout; follow the [asset download procedure](../inspect-robots-isaacsim-2yam/README.md#install)
on the field machine. Use its
`sim_eval/assets/yam/yam_mujoco/bimanual_yam_linear_flattened.xml` as both
`-E asset_path` and `-P mjcf_path`. Record `sha256sum` of the MJCF. Supply a
**measured** version 1 three-camera calibration JSON at
`/absolute/path/to/yam_calibration_v1.json`: intrinsics must be `360x640`,
`top_cam` under base and wrist cameras under their EEFs, with measured optical
transforms, table plane, and box opening. The
[schema fixture](tests/fixtures/synthetic_calibration_v1.json) contains test
numbers only and must not be used as field calibration. Record its file version
and SHA-256. Policy construction validates both files before rollout.

Set the key only in the Isaac process environment; never write it in a config,
command argument, README, or log. The Jev Choice model is fixed to
`jev-1.13.0`; the default endpoint is `https://api.typesafe.ai/v1/systemone`.

```bash
read -r -s -p 'TYPESAFE_API_KEY: ' TYPESAFE_API_KEY
export TYPESAFE_API_KEY
```

First prove the simulator cameras and reset in the Isaac environment, saving
`reset`, `step`, and `reset_again` frames. Check the real top/left/right PNGs
with the vision service and inspect overlays, masks, scores and categories.
If recognition is unstable, save the troublesome frames, check the overlay and
revise calibration or the service prompt; offline tests do not establish live
accuracy.

```bash
/path/to/isaaclab/bin/python \
  plugins/inspect-robots-isaacsim-2yam/scripts/boot_proof.py \
  --asset-path /absolute/path/to/bimanual_yam_linear_flattened.xml \
  --output-dir /absolute/path/to/boot-proof
/path/to/jev-vision-venv/bin/python \
  plugins/inspect-robots-jev/scripts/check_vision.py \
  /absolute/path/to/boot-proof/reset_top_cam.png \
  --camera top_cam --output-dir /absolute/path/to/vision-top
```

Repeat `check_vision.py` for `reset_left_cam.png --camera left_cam` and
`reset_right_cam.png --camera right_cam`. To check only the overlay pipeline
without live model inference, add `--fake`; its detections are artificial.
The boot proof, model service, calibration, and live overlays are all
**pending user execution**.

Start the YAM MolmoAct2 server from its own checkout for the baseline and
hybrid branches. Its `GET /act` must report
`allenai/MolmoAct2-BimanualYAM`, norm tag `yam_dual_molmoact2`, `checkpoint`,
and a resolved 40–64 character hexadecimal `revision`. Record that revision.
The `jev-direct` branch does not use MolmoAct2 and can run without this server.

```bash
cd /path/to/molmoact2
uv run python examples/yam/host_server_yam.py \
  --host 127.0.0.1 --port 8202 --dtype bfloat16
```

### Fixed-seed comparison (pending user execution)

The following three runs have identical registered task, 10 scenes, constructor
seed 7, evaluation seed 7, `360x640` camera settings, and the task's own 400
step horizon. Do not pass `--max-steps` or `--scorer` to a registered task.
Keep the same asset, calibration, service revisions, Isaac build, safety
settings and hardware for all three. Only the baseline and hybrid need the
Molmo server. The direct policy chooses among a finite local candidate set;
report that policy difference beside any outcome comparison.

```bash
/path/to/isaaclab/bin/inspect-robots run \
  --task isaacsim-2yam-put-everything-in-box -T episodes=10 -T seed=7 --seed 7 \
  --policy molmoact2 --embodiment isaacsim-2yam \
  -P server_url=http://127.0.0.1:8202 -P cam_height=360 -P cam_width=640 \
  -E asset_path=/absolute/path/to/bimanual_yam_linear_flattened.xml \
  -E cam_height=360 -E cam_width=640 \
  --save-video --rerun-save --no-rerun \
  --log-dir logs/jev-compare/s07-n10/molmoact2

/path/to/isaaclab/bin/inspect-robots run \
  --task isaacsim-2yam-put-everything-in-box -T episodes=10 -T seed=7 --seed 7 \
  --policy jev-hybrid --embodiment isaacsim-2yam \
  -P molmo_url=http://127.0.0.1:8202 -P molmo_prefix_steps=3 \
  -P vision_url=http://127.0.0.1:8765/v1/detect \
  -P calibration_path=/absolute/path/to/yam_calibration_v1.json \
  -P mjcf_path=/absolute/path/to/bimanual_yam_linear_flattened.xml \
  -P jev_model=jev-1.13.0 -P cam_height=360 -P cam_width=640 \
  -E asset_path=/absolute/path/to/bimanual_yam_linear_flattened.xml \
  -E cam_height=360 -E cam_width=640 \
  --save-video --rerun-save --no-rerun \
  --log-dir logs/jev-compare/s07-n10/jev-hybrid

/path/to/isaaclab/bin/inspect-robots run \
  --task isaacsim-2yam-put-everything-in-box -T episodes=10 -T seed=7 --seed 7 \
  --policy jev-direct --embodiment isaacsim-2yam \
  -P vision_url=http://127.0.0.1:8765/v1/detect \
  -P calibration_path=/absolute/path/to/yam_calibration_v1.json \
  -P mjcf_path=/absolute/path/to/bimanual_yam_linear_flattened.xml \
  -P jev_model=jev-1.13.0 -P cam_height=360 -P cam_width=640 \
  -E asset_path=/absolute/path/to/bimanual_yam_linear_flattened.xml \
  -E cam_height=360 -E cam_width=640 \
  --save-video --rerun-save --no-rerun \
  --log-dir logs/jev-compare/s07-n10/jev-direct
```

Each `--log-dir` is a root. Inspect Robots creates
`YYYYMMDD_runNNNN/` inside it. Preserve its eval `.json`, one `.rrd`,
`actions/*.jsonl`, `frames/*.npy`, `videos/*.mp4` (three cameras per trial),
and for Jev `jev-audit/` and `jev-masks/`. Check warnings: a missing Rerun SDK,
queue drops, audit write failures, or ffmpeg failure makes the corresponding
artifact incomplete. Never merge successive run directories. In a single
fixed-seed trial, inspect actual three-camera images, detections and masks,
localization quality, kept/filtered candidates, Choice probabilities, selected
short 14-D trajectory, next observation, and any hold reason. Use
[`diagnose_trajectory.py`](scripts/diagnose_trajectory.py) on each Jev eval
JSON for causal leads; a suspected wrong Choice still needs human review.

The offline evaluator script reads the Rerun `trial/<scene_id>/e<epoch>/reward`
series and takes the **highest step's value**, not the maximum reward. The
Isaac reward is 0, 0.5, or 1 for zero, one, or two objects in the box.
`success_at_end` comes independently from the JSON scorer; a missing reward is
reported as `missing`, never inferred from success. Provide the three concrete
eval JSON paths from their run directories:

```bash
/path/to/isaaclab/bin/python \
  plugins/inspect-robots-jev/scripts/summarize_eval.py \
  --molmoact2-log logs/jev-compare/s07-n10/molmoact2/YYYYMMDD_runNNNN/EVAL.json \
  --jev-hybrid-log logs/jev-compare/s07-n10/jev-hybrid/YYYYMMDD_runNNNN/EVAL.json \
  --jev-direct-log logs/jev-compare/s07-n10/jev-direct/YYYYMMDD_runNNNN/EVAL.json \
  --task-seed 7 --eval-seed 7 \
  --output logs/jev-compare/s07-n10/aligned.csv
```

Replace `YYYYMMDD_runNNNN/EVAL.json` with each actual path. The script requires
one `.rrd` beside each JSON; it rejects unmatched scene/epoch sets and wrong
evaluation seeds. Trial seeds use Inspect Robots' CRC32 derivation from outer
seed, task scene seed and epoch. The task constructor seed is not stored in the
eval JSON, so keep the command record to verify `-T seed=7`. `decisions` and
`holds` come from complete Jev audit sidecars; baseline values are blank.
`mean_policy_latency_s`, `mean_molmo_latency_s` and `mean_jev_latency_s` come
from the Jev audit. The JSON's run-level mean inference latency covers the
baseline. `mean_post_policy_interval_s` subtracts policy latency from the
monotonic observation-to-next-observation gap; it includes execution, rendering
and control overhead and must not be labelled pure actuator latency. Use a
separate Isaac profiler if pure execution latency is required.

Fill this table only after field runs; all cells currently await user data.

| Policy | Complete trials | Both in box / success % | Mean last reward (valid N / missing N) | Decisions / holds | Inference latency | Execution + next-frame interval | Versions / asset fingerprints |
| --- | --- | --- | --- | --- | --- | --- | --- |
| molmoact2 | 待用户执行 | 待用户执行 | 待用户执行 | N/A / N/A | 待用户执行 | 待用户执行 | checkpoint/revision, MJCF SHA-256: 待用户执行 |
| jev-hybrid | 待用户执行 | 待用户执行 | 待用户执行 | 待用户执行 | 待用户执行 | 待用户执行 | Molmo revision, Jev 1.13.0, DINO/SAM revisions, calibration/MJCF SHA-256: 待用户执行 |
| jev-direct (finite candidates) | 待用户执行 | 待用户执行 | 待用户执行 | 待用户执行 | 待用户执行 | 待用户执行 | Jev 1.13.0, DINO/SAM revisions, calibration/MJCF SHA-256: 待用户执行 |

For the mean last reward, average **only present** terminal values and report
both valid and missing counts; do not publish a single comparable mean if any
policy has missing trials until the recording gap is resolved. Count both-in-box
success from nonempty JSON `success_at_end` entries only, with failed/errored
trials shown separately. The evaluator's success/reward never enters Jev
Policy, vision, or geometry inputs.
