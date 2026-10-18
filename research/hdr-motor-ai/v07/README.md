# HDR Gear AI v0.7 · Physical two-port proof kit

**Status: printable and firmware-source-ready research prototype, NOT built or hardware-verified.** It is not a complete HDR relational intelligence, a digital substitute for a real physical measurement, or a proof of faster/greener computation than a CPU.

**Goal:** Take the v0.6 idea of two independent input motors with a mixed coupling, and implement the *smallest real machine* whose geometry supplies a nonadditive output without evaluating a mathematical function at runtime.

## Principle: one truly physical cross coupling

The two axes are **not in a parallel fixed-frame arrangement**. Servo A rotates link 1. Servo B is physically mounted at the end of link 1, rotating link 2 *relative to the first link*. The red marker/pen at the tip is directly observed by an overhead camera and defined as the candidate output `V₀`.

In ideal geometry, to **predict what a real machine should do**, a computer could calculate

\[
V_0(\alpha,\beta)=60\,e^{i\alpha}+45\,e^{i(\alpha+\beta)}\quad \text{(millimetres)}.
\]

The firmware **does not calculate that expression**: it issues two independently specified servo targets. Physical movement realizes the result. Camera observation measures `(x,y)` and checks the predicted ideal relation **only offline**. The second link is the `(k,l)=(1,1)` compound term from v0.6 and the first link is `(1,0)`. A nonzero rectangular mixed difference shows nonadditive dependence on both inputs. No planetary differential is required for this restricted but actual embodiment.

This **does not** resolve the unique emergent HDR `V₀` fixed point for an arbitrary collection of interacting points. It is a first mechanical mixed-interaction/output proof, with only two inputs and a 2R linkage.

## Package contents

- `hardware/cad/hdr_two_link.scad`: parametric CAD source, nominal SG90 slots, base and servo-mounted links.
- `hardware/cad/{base,link1,link2,pen}.stl`: OpenSCAD-generated printable geometry, **not mechanically fit-checked on a purchased servo**.
- `hardware/board/board.svg`: 1:1 A4 print calibration sheet with four vector ArUco fiducials (DICT_4X4_50 IDs 0–3).
- `hardware/bom/BOM.csv`: parts list including independent 5 V supply, PCA9685, optional INA219 and webcam.
- `hardware/firmware/hdr_two_servo/hdr_two_servo.ino`: Arduino/ESP32 firmware, servo range checking, 50 Hz pulses and optional power telemetry.
- `gear_ai/two_link.py`: theoretical kinematics, exact Jacobian, coupled 2R mass matrix, Coriolis model and torque-limited **approximate** digital twin.
- `gear_ai/vision.py`: real camera measurement. Four fiducials determine a geometric camera transform; a red dot supplies observed tip position.
- `scripts/measure.py`: USB/camera joint experiment, rejects unavailable frames, produces camera-only coordinate CSV and electrical trace.
- `scripts/analyze.py`: checks camera provenance, error metrics, never guesses energy/latency.
- `dist/physical_lab.html`: standalone responsive **digital concept explorer** for independent inputs and resultant.
- `tests`: deterministic mechanics and perspective-distorted synthetic camera regression tests.

## Fabrication

**Materials**: 2× SG90 microservo with horns, ESP32, PCA9685, 5 V regulated >=2 A supply, mounting M2 hardware, cable ties, USB overhead camera, bright red pen-tip marker, optional INA219 power breakout, print sheet and printed CAD. See BOM; no vendor is endorsed. **Measure actual servo body, ear spacing, shaft position and horn holes; prints contain editable nominal dimensions.** Test for interference before powering up.

1. **Inspect fit before printing.** Open `hardware/cad/hdr_two_link.scad`. Adjust `servo_x`, `servo_y`, `servo_hole_span`, `L1`, `L2` to your exact purchased parts. Print `base`, `link1`, `link2`, `pen` separately, flat; use PETG or PLA with 0.2 mm layers and 3+ perimeters. STL is geometry, not a manufacturing guarantee. Do not assume the servo horn screw holes are universal.
2. Screw or strap servo A upright into the base, shaft facing UP. Locate its rotation axis at the printed calibration sheet crosshair `(70 mm,118 mm)` measured from A4's top-left. Install link 1 on its horn; **shoulder-to-elbow centre distance 60 mm**. Use a central horn screw and secure the hub.
3. Mount servo B upright *on link 1* at the elbow. It must move with link 1, with its output shaft vertical. Install link 2 on B's horn; elbow-to-red-tip distance **45 mm**. No direct motor B connection to ground: that would destroy the intended coupled geometry. Keep cables outside both sweep envelopes.
4. Place a bright-red dot on the last output tip. Leave the arm assembly suspended high enough above the calibrated A4 sheet to swing freely. Mount an overhead camera orthogonally to the sheet. The red marker must be visible and not occluded by the arm. An opaque pen can draw the result, but the camera needs a visible red reference near the actual pen location; measure any marker offset.
5. Verify by hand, **power off**, that shoulder and elbow can each traverse ±55 degrees without collision, cable tension, paper contact or hard stops. Do not drive the servos before this check. Servo A carries the weight of servo B; a slow pace and low mass are essential. This is a low-load demonstrator, **not** a precision CNC.
6. Print `board.svg` at **actual 100% size** (210×297 mm A4), not fit-to-page. Marker centres must be 150 mm apart in X and Y. The crosshair at `(70,118)` marks the shoulder centre. Camera reading fails closed when fewer than four markers appear.

### Electrical wiring

| Item | Connection |
| --- | --- |
| ESP32 SDA21 / SCL22 | PCA9685 SDA / SCL and INA219 SDA / SCL |
| ESP32 3V3 | PCA9685 VCC and optional INA219 logic VCC (if breakout is 3.3 V compatible) |
| External 5V regulated PSU | INA219 VIN+ then VIN− to PCA9685 servo V+; omit INA219 => supply directly to servo V+ |
| ESP32 GND | Common GND with PCA9685, PSU negative and INA219 |
| Servo A | PCA9685 output channel 0, signal/red/brown correctly polarized |
| Servo B | PCA9685 output channel 1, signal/red/brown correctly polarized |

**Address conflict:** PCA9685 default address `0x40`; INA219 commonly defaults to `0x40` too. The firmware requires INA219 hardware-address jumper **`0x41`**. Do NOT attach both at the same address. INA219 commonly uses a 0.1 Ω shunt rated for ~3.2 A max: ensure the supply/load stays within **your sensor and wiring rating**. Servo loads can surge. Supply ESP32 via USB while driving servo V+ with an **external**, current-limited 5 V power source. Use a common ground. Never power servos from the ESP32 3V3 pin.

### Firmware setup

Arduino IDE + ESP32 board package. `Wire.h` from ESP32, no external servo libraries. Flash `hardware/firmware/hdr_two_servo/hdr_two_servo.ino` with 115200 baud. **Firmware compilation was not performed by this package's CI; a successful bench flash remains a required hardware gate.** Verify PCA9685 clock/pulse direction; set the per-servo `MIN_US`, `MAX_US`, `DIRECTION` and `OFFSET_US` to avoid mechanical travel limits.

On boot, all PWM outputs are **released**. Commands (line-oriented serial):

```text
HOME
SET 20 -15
STATUS
RELEASE
```

Targets are degrees, each strictly limited to ±55. `RELEASE` stops PWM but does **not** guarantee the servo remains stationary or unloads safely. Never leave the moving rig unattended.

### Software setup and actual measurement

```sh
python -m pip install -e '.[vision]'
python scripts/create_board.py
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/digital_twin.py     # tagged SYNTHETIC -- NOT physical evidence
# AFTER physically building and calibrating the device:
python scripts/measure.py --port /dev/ttyUSB0 --camera 0 --hold 1.0 --out results/hardware.csv
python scripts/analyze.py results/hardware.csv
```

Windows serial ports use `COM3` etc. The host captures 8 optical frames after each commanded pose settles; every output `(x,y)` is extracted **only from camera pixels**. Four printed fiducials must remain visible. Errors compare camera-observed tip to ideal *commanded* servo angles: this includes servo calibration offsets, flex, backlash, red-marker offset, camera registration and misassembly.

A companion `*_power_trace.csv` contains INA219 samples. Without power sensor all electrical fields become `nan`; no energy savings are inferred. Electrical bus integral is **servo-power energy over the sampled time**, not energy per isolated inference and not full-system camera/controller energy. The camera's processing and board power must be included for a true end-to-end energy comparison.

No sample collected here is a real hardware measurement: the repository deliberately includes **only** `results/digital_twin.json` and synthetic regressions.

## Falsifiable hardware acceptance gates (to run after building)

1. Mechanism survives 50 slow full test loops without collisions or stalled motors, no cables snagged, no overheating.
2. Four ArUco fiducials continuously visible, board centre distance measured at 150±0.5 mm, red marker unambiguous, observed tip readable during every settled pose.
3. Physical mixed finite difference over (A,B) corners ±40° must be **nonzero** beyond 3× optical noise; compare to the ideal model and report measured confidence interval, not only a point estimate.
4. For ≥10 repeat cycles, report repeatability RMS and max, backlash under forward/reverse approach, observed torque-related droop at different speeds, settling time with synchronized hardware trigger, and power in watts/joules including controller/camera.
5. Benchmark a **matched digital baseline** on the same input/output workload, with measurement instrumentation accounting for device, camera readout and sampling overhead; no energy or latency benefit should be claimed before that.
6. Structure-learning claims are **outside this prototype**. Changing hardware link length is a deliberate **offline** physical modification; there is no motor-powered self-modifying topology here.

## Limits

- Two rigid links implement only `(1,0)` and `(1,1)` terms, not arbitrary integer mixed gear ratios. Extended v0.6 stages still need differentials or additional mechanical arrangements.
- This is a **physical build plan plus numerical and optical toolchain**, not a device constructed in the lab.
- Camera coordinates are calculated by digital image processing **as measurements**, not by digital evaluation of input angles; this does not eliminate digital sensing energy.
- A 2R linkage has a positional Jacobian singularity at straight/fully folded poses; exact joint inverse is not globally unique. The forward physical configuration yields a unique tip for given joint angles, but is not proof of HDR global fixed-point uniqueness.
- The ~9 g SG90 servo bodies, printed dimensions, backlash, duty cycle and stall current vary. Electrical/mechanical tolerances must be measured.

## Source and licensing

Original research implementation MIT. Inspired by Christian Hohlfeld's HDR hypothesis and the gear synthesis principle in [`nostack-dev/kurzlernen-de/topology.html`](https://github.com/nostack-dev/kurzlernen-de/blob/main/topology.html); no source code copied from that application.
