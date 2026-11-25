# VIGIL-ROVER X

Drosophila Connectome-Informed NeuroAI for Autonomous Collision Avoidance, Directional Memory and Active Blind-Spot Recovery

Research and implementation specification · 25 November 2026

## 1. Executive decision and scope
Build a differential-drive virtual rover whose visual escape and directional memory are informed by actual Drosophila circuits. The biological core must be executable and independently tested, not simply an analogy. Retain looming detection, contact override, camera-fault handling, collision avoidance and blind-spot recovery. Use a learned artificial extension, Neuro-COVER-L, to assess whether a safe observation maneuver can restore task-relevant knowledge. The rover is not a fly digital twin. FlyGym is a separate biological reference; PyBullet is the authoritative rover environment. No physical hardware is required.

The prior 2D prototype is a regression baseline, not the final platform. A novelty target above 7.5/10 is a conditional aspiration, not a measured score.

## 2. Biological evidence and boundaries
T4/T5 direction-selective motion processing, LC4 angular velocity, LPLC2 expansion/size-related information and Giant Fiber escape output are supported by neuroscience [R1, R1a, R2]. Central complex E-PG, P-EN, P-EG and Delta7 networks contribute to heading representation; visual landmark pathways involve MeTu, TuBu and ER populations [R5–R7].

Use published computational models as references rather than inventing synaptic parameters. Flyvis provides connectome-constrained optic-lobe motion processing; do not assume it implements LC4/LPLC2/GF end-to-end. Inspect its available cell types and add separately validated output modules.

Johnston's Organ mechanosensory input to GF includes electrical and chemical connections [R8]; virtual whiskers are engineered contact proxies, not anatomical reproductions. Artificial occupancy inference, counterfactual risk and active scanning are engineering additions. Do not claim the visual escape and compass pathways directly connect without verified evidence.

## 3. Data collection and provenance
Primary dataset: FAFB v783. Secondary independent evaluation: BANC v888. Download consolidated_cell_types, connections_princeton, neuron metadata, neurotransmitter predictions and relevant neuropil/column annotations from Codex [R9–R12]. Verify which download includes weak edges before testing thresholds. The Codex connectivity view uses dataset-specific defaults [R12]. Use static releases and record data product names, version, date, SHA256, source and license. Codex sign-in and API token may be required; store token in Kaggle Secrets or local environment, never in repository. Public FlyWire releases have CC BY-NC 4.0 terms; verify each dataset/model/checkpoint separately. No raw EM volumes required.

Match cell types across datasets by published mappings and column/side where justified, not root IDs. Keep raw IDs as strings to avoid JavaScript integer precision loss. Review annotation aliases, missing types, uncertain signs and completeness before extraction.

Start with exact manually reviewed IDs; use 1-hop input expansion only when justified, inspect coverage and fan-in. Keep edge direction pre→post, sum neuropil-specific counts when pooling and record threshold (compare thresholds of 1 and 5 only on an unfiltered source; record the upstream filter). Preserve separate edge tables and masks per circuit. Use data from a single version for all joins. Do not infer an absent edge from missing annotation.

## 4. Biological core implementation
Preserve timestamped spatial image sequences and their retinal sampling geometry. A summary of three image sectors discards local motion direction and cannot serve as the biological visual input. Direct LPLC2 and LC4 inputs to GF are documented [R1]. Map T4/T5 motion input to LPLC2 using the evidence in R1a. Specify the LC4 input model separately. First reproduce the EMD-LPLC2-GF model [R2] on angular-size/velocity stimuli, approaching, receding, translating and near-miss targets. Use author MATLAB outputs as reference [R2a]; port to NumPy/PyTorch only after checking license.

Next inspect Flyvis [R3] pretrained motion responses, map calibrated spherical/hexagonal retinal samples to its expected inputs; benchmark visual field coverage and latency. Compare classical optical flow and the published looming model.

Fit a sparse directed rate network: tau_i dx_i/dt = -x_i + phi(sum_j W_ij x_j + B_i u); W_ij = M_ij S_ij softplus(theta_ij) for known-sign edges. Unknown signs get separate unconstrained/sensitivity variants. Synapse counts can initialize or regularize magnitudes, not determine conductance. Preserve biological mask during training; compare fixed-count, trainable-masked, degree-preserving shuffled and dense budget-matched networks.

Use Shiu's LIF reference [R4] for selected spiking experiments after rate-model validation; SNN is optional, not automatically superior. Include time constants, input maps and readouts in reproducibility metadata. Distinguish numerical stability from biological fidelity.

## 5. Heading and spatial memory
Implement E-PG/P-EN/P-EG/Delta7 compass using published column assignments [R5, R18, R21] and graph extraction. Decode heading from circular population vector; low bump magnitude is only a candidate confidence signal, calibrate it against actual error. Provide visual landmarks and noisy wheel odometry or angular velocity estimates to all matched models; oracle pose is evaluator-only. Test dark intervals, rotations, conflicting landmarks, drift and reacquisition. Decode the compass angle as atan2(sum of activity times sin(column angle), sum of activity times cos(column angle)). Log missing estimates when the population vector vanishes. Compare simple odometry, circular Bayesian filter, canonical ring attractor and GRU.

A heading signal is not an occupancy map. Build separate egocentric sector/occupancy belief using monocular tracking, estimated ego-motion and conservative unknown-space handling; no ground-truth depth as agent input in the main condition. Use optional stereo/depth as a separately labeled upper-bound experiment. Distinguish outside-FOV, occlusion, stale observations and sensor faults. Never label invisible space free. Age observed-free evidence under bounded possible obstacle motion.

## 6. Neuro-COVER-L proposed method
Research hypothesis: biological circuit states provide task-relevant information about whether ambiguity can be resolved safely. Encode x_visual, x_heading, uncertainty about heading, looming residuals, occupancy belief and frame diagnostics into z_t. Learn risk r(a)=P(body contact within H|history,a) and recovery g(a)=E[U_task(b_t)-U_task(b_next)|history,a], where U_task is uncertainty about collision-relevant hidden states, not generic image entropy. Use the same candidate action library for all policies: hold/brake, slow straight, small left/right scans, bounded turns. Include terminal braking and account for full swept footprint.

Train g using counterfactual observation updates across plausible hidden worlds, not just ground-truth state differences. Distinguish temporary sensor occlusion, camera fault and genuine threat; if observations are indistinguishable under every feasible action, abstain.

Use risk upper bounds calibrated on validation and conservative reachable-set checks under explicit obstacle speed/acceleration and actuation bounds. Optimize progress plus recoverability minus action cost only among feasible candidates. Predict restoration of heading as a separate target. Score circular heading error and time to reacquire a landmark against evaluator truth after each branch. A learned probability is not a safety proof; external moving agents can hit a stopped robot. On contact, brake independently of visual/learned controller; reverse/turn only if the swept escape region is supported safe. Record rejected actions and reasons.

## 7. Counterfactual experiment specification
At sampled decision states, clone/restore complete physics state, obstacle script state, sensor buffers, controller history, timers and PRNG state. Branch each action with common exogenous randomness for horizon H=1–2 s (initial configurable value); run action plus terminal brake. Verify branch equivalence before using labels. Capture unique body contact, minimum clearance, progress, threat visibility and next observations.

Generate paired ambiguous worlds: matching visible history with hidden obstacle versus empty occlusion, recoverable occlusion versus persistent camera fault. Where possible match histories by construction, quantify remaining visual differences. Label safe recoverability only when a feasible observation action changes the posterior over task-relevant hypotheses under stated assumptions. No future frames, scenario names, true hidden positions or fault injection flags in deployed inputs. Compare with ordinary information gain, passive stop, and active diagnosis methods [R13,R14].

## 8. Physics and simulation engineering
PyBullet DIRECT/TinyRenderer is the authoritative physics and camera backend. Start with a ~0.20 x 0.16 m rover, ~0.14 m wheelbase, 0.35 m/s maximum speed; configurable initial design values, not biological measurements. 120 Hz physics, 20 Hz controller, 10 Hz 128x96 camera per initial view; profile before increasing. Build two-wheel URDF with chassis collision and separate virtual left/right feeler geometry. Use camera rays/rendered RGB, never object positions disguised as vision. Test friction, braking, contact resolution, fixed timestep and sensor scheduling. Separate capture and delivery timestamps and drop/freeze/delay faults.

Start front-facing camera budget; panoramic/dual-eye experiment gets matched-FOV controls to avoid attributing wider vision to biology. Three.js visualizes backend poses, contact events and frames; do not run parallel authoritative physics. FlyGym/MuJoCo is optional reference to validate fly visual inputs, not interchangeable rover benchmark.

## 9. AI learning and Kaggle workflow
Use the official Kaggle CLI to submit a private, versioned run after the circuit-response and data gates pass. Verify CUDA allocation in the run and download its metrics, weights and split manifest onto D:. Stage 1: fixed published modules and logistic/XGBoost baselines. Stage 2: rate-based masked networks with compact GRU or TCN heads (32–128 units), same observations for all models.

Stage 3: fine-tune selected weights with biological mask maintained; freeze or unfreeze via documented ablation. Stage 4: uncertainty via independent ensembles and validation-only calibration. Train health classification and action-conditioned collision/recovery heads using BCE, proper probabilistic scoring and masked regression for censored TTC where relevant. Choose loss weights on validation. Compare frozen vs trainable biological core and remove circuit-state inputs as ablation. PPO is optional after deterministic environment and benchmarks work; recurrent PPO must use identical sensor and action budget. Use GPU for batched training; physics CPU and data generation in headless workers. Inspect torch.cuda, record accelerator and quota, checkpoint after each epoch; use one GPU initially. Export weights, normalization, config, commit hash and dataset split manifest. Kaggle internet and quotas vary; upload licensed compact data shards if offline. No GPU performance or quota promised.

## 10. Dataset and split design
Start 200–500 inspected episodes, scale only after valid physics and labels; 3,000–10,000 episodes is a planning range not a statistical claim. Parameterize corridor/warehouse/open arena, target size, speed, trajectories, near-miss offsets, occluder shape, lighting, blur, frame freeze/drop/delay, self-motion, contact noise and compound faults.

Hold out families of geometry, trajectory and fault combinations, not merely random seeds. Keep all branches of a starting state in one split.

For independent connectome evaluation, align homologous types and explicitly document transfer mapping; no retuning on held-out test. TartanGround/TartanAir and OpenLORIS may test perception generalization, not provide our action counterfactual labels. Use compressed frames with episode IDs and Parquet metadata; log input vs oracle separately. Pilot storage: 128x96 RGB is 36,864 bytes/frame raw; 100,000 frames ~3.69 GB before overhead/compression. Stream and shard rather than load whole datasets locally.

## 11. Evaluation and causal attribution
Compare contact-only, conventional looming/GF-Lite, published EMD-LPLC2-GF, classical motion/odometry, standard GRU, biological masked core, matched rewired graphs, biological core + passive risk, ordinary active perception, OA-MPC-style occlusion planner, safe active diagnosis reference and Neuro-COVER-L. Where exact reproduction is unavailable label approximations rather than claim reproduction.

Factorial experiment: topology (bio/rewired/generic) × recovery (off/ordinary/Neuro-COVER-L), matched FOV, parameters, training steps, action library and compute budget. Ablate looming, heading, memory, uncertainty, recoverability, tactile override and safety filter separately in simulation.

Use collision-free goal completion as the primary endpoint. Report body collisions per episode, progress and unnecessary stops as secondary outcomes. Separate whisker contact from body collision. Measure warning lead time, near-miss false alerts, heading error during blackout, calibration/Brier score, selective-risk coverage, recovery time, minimum clearance and CPU/GPU costs. Report paired episode-level differences with bootstrap CIs, multiple training seeds and failure examples; preregister thresholds and primary endpoints before test. No frame pseudoreplication. Compare equal budgets and retain negative results.

- **M1 · Classical vision and control.** Reference for camera, timing and tactile behavior. The starter stopping rule covers part of this controller.
- **M2 · Published fly-inspired looming controller.** Reproduce published stimulus responses before the navigation comparison.
- **M3 · Conventional GRU.** Match inputs, sensory field, action choices, parameter count and training budget.
- **M4 · Connectome-constrained network.** Test the biological visual and compass representations with recovery disabled.
- **M5 · Rewired graph.** Preserve in-degree and out-degree, sensor coverage and readout access. Repeat across rewiring seeds.
- **M6 · Biological network with conventional risk prediction.** Measure the effect of passive action-risk prediction.
- **M7 · Biological network with ordinary active perception.** Test generic information gain with the same feasible actions.
- **M8 · Neuro-COVER-L.** Condition the recovery prediction on visual and compass activity. Log the evidence behind the executed action.

Apply the same recovery choices to conventional and rewired networks. The difference between the effects of enabling recovery across topologies estimates their interaction. Report uncertainty around that difference.

## 12. Software stack and repo
Python 3.11 candidate environment (verify Flyvis/FlyGym compatibility in separate pinned environments); PyBullet, Gymnasium, NumPy, SciPy, pandas, PyArrow, OpenCV, PyTorch, scikit-learn, XGBoost, NetworkX; optional Brian2 reference and Flyvis/FlyGym; pytest, Git, uv/pip, Jupyter/Kaggle. FastAPI/Pydantic/WebSocket for API; React/TypeScript/Three.js for studio; Plotly, SQLite/Parquet; FFmpeg for recorded actual trajectories. Node LTS and browser WebGL; verify graphics support. No paid hardware/software required by design, but quota/storage constraints apply.

Repo packages: connectome/ (manifest/extract/audit), neuro/ (retina, looming, compass, dynamics), sim/ (world, robot, sensors, faults), belief/ (visibility, integrity), learning/ (datasets, models, train, calibrate), control/ (GF, risk, recovery, shield), evaluation/ (replay, branches, metrics, baselines), api/, web/, tests/, configs/, outputs/. Use observation/action contract and prohibit oracle keys at policy API. Log run ID, sim time, capture/arrival time, input hash, circuit activations, action candidates, decision, contact and metrics.

## 13. Studio and demonstrations
Interactive 3D warehouse and obstacle editor, live first-person/compound-eye image, visual expansion overlays, neural activity maps, circular compass activity bump, visible/unknown sector map, camera integrity status, candidate risks/recovery scores, actual trajectory, controls for fault injection, comparison and downloadable experiment logs. Pause and single-step simulation. Display oracle geometry only in explicitly labeled evaluator view.

Demonstrate a looming obstacle, passing near-miss, hidden moving obstacle around corner, temporary blackout with directional memory, permanently failed camera requiring abstention, bilateral contact and recovery under combined held-out faults. Export MP4 from recorded physics state, not a promotional animation. Frontend must reproduce the same run IDs, positions, and events as backend.

## 14. Development roadmap

The ten milestones below are planned. Each milestone has an evidence requirement in Implementation_Acceptance_Gates.csv. The baseline source is an external dependency for reproducing the original simulator. Work on the connectome acquisition manifest can proceed while that source is located.

**Weeks 1–2 · Baseline audit.** Recover the 2D source revision if available. Verify empty-scene health, geometry-based labels, packet freshness and collision event accounting. Record any cases that cannot be reproduced from the prior report.

**Weeks 3–4 · Connectome processing.** Record the FAFB v783 product, checksum, source and filter. Export reviewed neuron IDs, visual and compass directed graphs, isolated nodes and sign coverage.

**Weeks 5–6 · Rover physics and tactile sensing.** Measure braking and contact in PyBullet. Test bilateral stop, one-sided contact with blocked escape, and a verified-clear escape during camera outage. Preserve capture and arrival times.

**Weeks 7–8 · Biological vision.** Reproduce reference looming responses on approach, recession, translation, yaw and near misses within tolerances registered before fitting. Record FOV and sampling.

**Weeks 9–10 · Directional memory.** Measure circular error, blackout drift, bump stability and landmark reacquisition against odometry and conventional models.

**Weeks 11–12 · Blind spots and training labels.** Test indistinguishable hidden worlds, observation aging and exact branch restoration. Keep all branches in the same split and oracle fields out of inputs.

**Weeks 13–14 · Kaggle GPU training.** Fit matched biological, rewired and conventional models on versioned shards. Freeze validation-calibrated thresholds and recover checkpoints and metrics locally.

**Weeks 15–16 · Controller integration.** Execute matched scenarios with recovery off, ordinary recovery and circuit-conditioned recovery. Count separate tactile and body events and record safe abstention.

**Weeks 17–18 · Scientific validation.** Evaluate preregistered topology and recovery contrasts, ablations and paired world-level intervals. Document BANC v888 mapping coverage and transfer limits.

**Weeks 19–20 · Research studio and release.** Verify that telemetry, frames, compass activity, decisions and recordings replay the same run. Package source, configurations, manifests and limitations.

## 15. Acceptance and novelty

Register stimulus tolerances, collision-risk limits, the main outcome and the statistical comparison before examining held-out runs. A null or adverse result is a valid research outcome. In particular, if the degree-preserving rewired models match the biological networks across held-out faults, report the measured interval and the limits of the biological-wiring hypothesis. Treat any novelty rating as a research aspiration.

## 16. Current project evidence

The package includes the research specification, resource index, milestone tracker, a directed-graph extractor with synthetic tests and a provisional classical stopping rule with tests. The original simulator report and a checkout of the published looming baseline are available in the D-drive workspace. The historical 2D implementation is described in that report. Its executable source has not been identified here. The biological graphs, validated neural responses, 3D simulator, trained models and closed-loop comparisons remain implementation milestones.
