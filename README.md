# Whisker 2D Localization Simulator

This project has two deliberately separate execution modes.

## 1. True online simulation

Use this for interactive development of sensing, FSM, control, odometry, and later localization.

```bash
python examples/run_realtime_sim.py --environment easy_rectangle
```

The online architecture is:

```text
50 Hz control thread
    whisker sensing
    -> FSM
    -> velocity command
    -> odometry
    -> ground-truth integration
    -> logging

GUI thread
    reads latest snapshot
    -> visualization only
```

The GUI never advances the simulator and cannot skip control iterations.

Other environments:

```bash
python examples/run_realtime_sim.py --environment angled_room
python examples/run_realtime_sim.py --environment concave_L
python examples/run_realtime_sim.py --environment repeated_bays
```

Change visualization FPS only:

```bash
python examples/run_realtime_sim.py --environment easy_rectangle --render-fps 20
```

## 2. Offline fast simulation

Use this later for Monte Carlo tests, ablations, parameter sweeps, and paper tables.

```bash
python examples/run_batch_sim.py --environment easy_rectangle --seed 10
```

This mode uses `whisker_sim/simulation.py` and runs as fast as the computer allows. It has no live GUI.

## Important files

```text
whisker_sim/
├── config.py
├── controller.py
├── environments.py
├── geometry.py
├── map.py
├── models.py
├── odometry.py
├── sensors.py
├── simulation.py             # offline/batch simulator
├── realtime_simulation.py    # true online 50 Hz simulator
├── realtime_animation.py     # live visualization only
└── visualization.py          # static plots after a run
```

The old replay-only `animation.py` has been removed.

## Whisker mounting geometry

Body frame:

```text
+x = forward
+y = left
```

Whisker bases:

```text
left  = (0.09, +0.025) m
right = (0.09, -0.025) m
```

Each whisker extends 0.20 m forward.

## Tactile depth band

```text
50 mm < contact depth < 100 mm
```

with

```text
contact depth = 200 mm - measured base-to-wall range
```

## FSM

```text
FORWARD
  -> CF_ACTION
  -> BACKWARD
  -> CHANGE_DIRECTION
  -> FORWARD
```

Current commands:

```text
FORWARD       +0.20 m/s
RIGHT SWEEP   -0.20 m/s in body y
BACKWARD      -0.20 m/s
TURN RATE     up to 25 deg/s
```

`CHANGE_DIRECTION` currently selects a random heading. It can later be replaced by the GPIS/localization planner.
