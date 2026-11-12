# Hardware validation status

## L20 V10 RS485

**Status: experimental; no physical actuation validation recorded by this work.**
No test in the automated suite opens a physical serial port or sends commands
to a real hand. New `/dev/pts/*` pseudo-terminals emulate Modbus responses.

Implemented configuration: RS485 460800 baud, 8N1, right slave `0x29`, left
slave `0x2A`, 30 position registers. Mapping includes joint directions, offsets
and piecewise thumb calibration ported from the existing local driver.

Software validation covers CRC, fragmented replies, exceptions, bad echoes,
timeouts, duplicate suppression, startup cleanup, tracking freshness, independent
joint smoothing, mimic consistency and the complete dry-run pipeline. The new
mapping was compared with the original mapper on 2000 shuffled left/right poses.

The command stream is position-target output, without encoder or force feedback.
On input loss it stops updating the target; that is not a hardware emergency stop.
Shutdown closes serial; opening the hand on exit requires the explicit flag.

## Record a physical run

Fill this record only after a real run; do not infer results from a dry run.

| Field | Result |
|---|---|
| Date / operator / commit | Not recorded |
| Hand revision / firmware / side | Not recorded |
| Power, adapter and interface configuration | Not recorded |
| Checkpoint and SHA-256 | Not recorded |
| Effective port, baudrate, slave id, speed/current registers | Not recorded |
| Open pose and each active joint's direction/range | Not tested on hardware |
| Slow flexion and thumb/finger pinch alignment | Not tested on hardware |
| Tracking loss / reconnect / Ctrl+C behaviour | Not tested on hardware |
| Sustained command rate and communication errors | Not measured on hardware |
| Physical stop mechanism | Device/operator responsibility; not implemented here |

Attach the exact command and relevant logs. Record a failure as a failure rather
than changing this document to "validated" because the software tests pass.

## Pico sender

The maintainer identifies the headset application as XR-Robotics/XRoboToolkit.
Its installed APK revision and headset firmware have not yet been recorded.
See [Pico setup](PICO_SETUP.md) for verified upstream download links and the
wire-format contract. Loopback tests exercise that contract without a headset.
