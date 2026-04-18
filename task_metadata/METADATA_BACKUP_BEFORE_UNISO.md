# Task metadata backup (before switching to UniSO-style text)

Snapshot date: 2026-04-07. Original per-task files were copied under `archive_pre_uniso/`.

---

## ant.txt

```
Task: Ant morphology
Type: continuous
Description: Quadruped robot body design in a physics simulator; a fixed policy evaluates each morphology by rollout.
Goal: Maximize forward locomotion speed (or the benchmark’s locomotion score) over a fixed horizon.
Design space: 60 bounded continuous variables (limb sizes, orientations, placements).
```

---

## dkitty.txt

```
Task: D'Kitty morphology
Type: continuous
Description: ROBEL D'Kitty robot morphology in simulation; a controller matched to each body is used for rollouts.
Goal: Navigate to a fixed target; maximize the benchmark navigation / locomotion objective.
Design space: 56 bounded continuous parameters (link lengths, link widths, joint and body layout).
```

---

## superconductor.txt

```
Task: Superconductor critical temperature
Type: continuous
Description: Design-Bench continuous task over compositional and structural features of cuprate-like superconductors; an oracle regressor scores each design.
Goal: Maximize the benchmark’s predicted critical temperature (proxy objective).
Design space: 86 bounded continuous parameters (material descriptor vector used in the Design-Bench dataset).
```

---

## tfbind8.txt

```
Task: TF-Bind-8
Type: discrete
Description: Optimize DNA sequences of length 8 for binding to the SIX6_REF_R1 transcription factor (Design-Bench).
Goal: Maximize binding affinity in the benchmark’s scoring.
Design space: combinatorial DNA sequence design (eight categorical positions with four nucleotide choices each, logits or embedding representation in models).
```

---

## tfbind10.txt

```
Task: TF-Bind-10
Type: discrete
Description: Optimize DNA sequences of length 10 for binding to the SIX6_REF_R1 transcription factor (Design-Bench).
Goal: Maximize binding affinity in the benchmark’s scoring.
Design space: combinatorial DNA sequence design (ten categorical positions with four nucleotide choices each, logits or embedding representation in models).
```

---

## gtopx2.txt

```
Task: GTOPX-2 / Cassini2
Type: continuous
Description: Interplanetary trajectory to Saturn (Earth–Venus–Venus–Earth–Jupiter–Saturn), with deep-space maneuvers; harder than a minimal Cassini-style instance.
Goal: Minimize total mission ΔV (velocity change) to the rendezvous condition.
Design space: 22 bounded continuous variables (epochs, asymptotic speeds, segment times, DSM fractions, fly-by radii, B-plane angles).
```

---

## gtopx3.txt

```
Task: GTOPX-3 / Messenger reduced
Type: continuous
Description: Interplanetary trajectory to Mercury without resonant Mercury fly-bys; sequence Earth–Venus–Venus–Mercury.
Goal: Minimize total mission ΔV.
Design space: 18 bounded continuous variables (epochs, asymptotic speeds, segment times, DSM fractions, fly-by radii, B-plane angles).
```

---

## gtopx4.txt

```
Task: GTOPX-4 / Messenger full
Type: continuous
Description: Mercury mission with resonant Mercury fly-bys; sequence Earth–Venus–Venus–Mercury (multiple Mercury passes).
Goal: Minimize total mission ΔV.
Design space: 26 bounded continuous variables (epochs, asymptotic speeds, segment times, DSM fractions, fly-by radii, B-plane angles).
```

---

## gtopx6.txt

```
Task: GTOPX-6 / Rosetta
Type: continuous
Description: Multi-gravity-assist mission to comet 67P; sequence Earth–Earth–Mars–Earth–Earth–67P with deep-space maneuvers.
Goal: Minimize total mission ΔV to comet arrival.
Design space: 22 bounded continuous variables (epochs, asymptotic speeds, segment times, DSM fractions, fly-by radii, B-plane angles).
```

---

## lunar_lander.txt

```
Task: LunarLander controller (OpenAI Gym)
Type: continuous
Description: Learn the parameters of a controller for a lunar lander to maximize the mean terminal reward across a consistent batch of 50 randomly generated landscapes (varying initial positions and velocities). The design space is 12-dimensional continuous inputs for the lander's actions.
Goal: Maximize the objective encoded in offline dataset labels (mean terminal reward / return).
Design space: 12 bounded continuous variables (controller parameters).
```

---

## rover.txt

```
Task: Rover trajectory
Type: continuous
Description: 2D trajectory optimization for a rover: design a reasonable trajectory to minimize the cost. Start and goal positions and a cost over the state space define the problem; trajectory cost c(x) is evaluated for designs x in a 60-dimensional unit hypercube.
Goal: Minimize trajectory cost (or maximize the score used in the offline dataset, depending on label convention).
Design space: 60 continuous variables in the unit hypercube parameterization.
```

---

## robot_push.txt

```
Task: RobotPush (Box2D)
Type: continuous
Description: Control the robot to push items to a designated location, minimizing the distance between a predefined target location and two objects. Motion is governed by 14 continuous parameters (e.g. position, orientation, speed, direction of motion).
Goal: Optimize the objective encoded in offline labels (typically minimize distance to target or a related score).
Design space: 14 continuous variables.
```
