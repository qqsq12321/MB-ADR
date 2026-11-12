# Pico / XRoboToolkit setup

The maintainer's sender is **XR-Robotics / XRoboToolkit**. The upstream
[project overview](https://github.com/XR-Robotics) and
[Unity client](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client)
provide installation and headset compatibility details.

## Obtain the headset app

The upstream overview links the
[v1.1.1 release](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/tag/v1.1.1)
and [XRoboToolkit-PICO-1.1.1.apk](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases/download/v1.1.1/XRoboToolkit-PICO-1.1.1.apk).
[Other releases](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client/releases)
exist; the maintainer's exact installed APK version is not yet recorded, so this
link is an upstream reference rather than a claim that every release is tested.
Follow the upstream developer-mode/ADB instructions for your headset. For the
referenced downloaded APK, upstream documents:

```bash
adb install -g XRoboToolkit-PICO-1.1.1.apk
```

APK files are not bundled or relicensed by this repository.

## Run the PC side

Keep the headset and PC on a mutually reachable LAN. Choose one PC listener.

```bash
# Terminal 1: included receiver/relay; keep it running between demos
python scripts/pico4_daemon.py

# Terminal 2: simulation or hardware dry-run, default relay mode
python scripts/demo_pico4.py --robot l20 --hand right --device cpu
python scripts/teleop_l20.py --hand right --device cpu --dry-run
```

The installed daemon command is `mbadr-pico-relay`. An already running compatible
relay can be reused; do not start a second listener on TCP 63901/63902. The
upstream XRoboToolkit-PC-Service is a different service; MB-ADR's receiver/relay
is sufficient for this hand-keypoint stream. Do not run both PC listeners on the
same direct port.

Alternatively skip the relay and use the receiver inside a single application:

```bash
python scripts/demo_pico4.py --robot l20 --hand right --mode direct --device cpu
```

In the headset's XRoboToolkit panel, select/enter the PC address, enable
**Tracking — Hand**, and turn **Send** on. These controls are described in the
[upstream Unity client README](https://github.com/XR-Robotics/XRoboToolkit-Unity-Client#unity-ui-main-panel-reference).
PC discovery uses UDP 29888; hand data arrives on TCP 63901. Applications consume
the local relay on 127.0.0.1:63902. A connected/WORKING UI does not by itself prove
that the selected hand is active or that hand transmission is enabled.

## Wire-format contract

The [upstream PC-service documentation](https://github.com/XR-Robotics/XRoboToolkit-PC-Service)
describes `Hand.leftHand` / `Hand.rightHand`, `isActive`, and 26
`HandJointLocations`. Each joint's `p` is a comma-separated pose string;
MB-ADR takes its first three coordinates in metres. It selects 21 landmarks,
centres them on the wrist and applies the same preprocessing as training.

- Direct framing: `0x3F`, command byte, uint32-LE payload length, payload,
  uint64 timestamp, `0xA5`; JSON command is `0x6D`.
- Relay framing: uint32-LE device-id length, device id, uint32-LE JSON length, JSON.
- Direct JSON and the upstream JSON-string `value` envelope are accepted.
- The relay forwards fresh packets only and emits inactive hands on disconnect;
  slow relay consumers are disconnected rather than blocking the stream forever.

Inspect listening ports with `ss -ltnp`. If discovery fails, check LAN reachability,
UDP broadcast routing and the PC address selected in the headset. The software
tests cover loopback packets; they do not replace matching an APK and firmware
to a real headset.
