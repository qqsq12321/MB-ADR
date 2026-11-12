"""Linker L20 V10 RS485/Modbus output and calibrated joint mapping.

The register layout and GEORT calibration match the local AnyDexRetarget
``drivers_linker_l20.py`` implementation. Everything needed at runtime lives
here; neither AnyDexRetarget nor the vendor SDK is imported.
"""

from __future__ import annotations

import struct
import sys
import time

import numpy as np


ACTIVE_JOINTS = (
    "THUMB_CMC_YAW", "THUMB_CMC_ROLL", "THUMB_CMC_PITCH", "THUMB_MCP",
    "INDEX_MCP_ROLL", "INDEX_MCP_PITCH", "INDEX_PIP",
    "MIDDLE_MCP_ROLL", "MIDDLE_MCP_PITCH", "MIDDLE_PIP",
    "RING_MCP_ROLL", "RING_MCP_PITCH", "RING_PIP",
    "PINKY_MCP_ROLL", "PINKY_MCP_PITCH", "PINKY_PIP",
)
JOINT_LIMITS = {
    "THUMB_CMC_YAW": (0.0, 1.57),
    "THUMB_CMC_ROLL": (0.0, 1.39),
    "THUMB_CMC_PITCH": (0.0, 0.83),
    "THUMB_MCP": (0.0, 1.25),
    **{f"{finger}_MCP_ROLL": (-0.23, 0.23) for finger in ("INDEX", "MIDDLE", "RING", "PINKY")},
    **{f"{finger}_MCP_PITCH": (0.0, 1.22) for finger in ("INDEX", "MIDDLE", "RING", "PINKY")},
    **{f"{finger}_PIP": (0.0, 1.7415730337078652) for finger in ("INDEX", "MIDDLE", "RING", "PINKY")},
}
JOINT_OFFSETS = {
    "THUMB_CMC_ROLL": 0.15,
    **{f"{finger}_MCP_PITCH": -0.15235988 for finger in ("INDEX", "MIDDLE", "RING", "PINKY")},
}
ALIGNMENT_POINTS = {
    "THUMB_CMC_ROLL": ((0.13962634015954636, 0.0), (0.7299065850398866, 0.5), (1.4423598775598299, 1.0)),
    "THUMB_CMC_PITCH": ((0.05235987755982989, 0.0), (0.4499065850398866, 0.5), (0.8823598775598299, 1.0)),
    "THUMB_MCP": ((0.13962634015954636, 0.0), (0.67735987755983, 0.5), (1.25, 1.0)),
}


class L20Mapper:
    """Name-aware conversion from URDF radians to 30 uint8-valued registers."""

    def __init__(self, joint_names):
        self.joint_names = tuple(str(n).upper() for n in joint_names)
        self.indices = {}
        for suffix in ACTIVE_JOINTS:
            matches = [i for i, name in enumerate(self.joint_names) if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one L20 joint *{suffix}, got {matches}")
            self.indices[suffix] = matches[0]

    def to_registers(self, qpos: np.ndarray) -> list[int]:
        q = np.asarray(qpos, dtype=np.float64)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("L20 qpos must be a finite vector matching joint_names")
        values = {}
        for name, index in self.indices.items():
            value = q[index] + JOINT_OFFSETS.get(name, 0.0)
            if name in ALIGNMENT_POINTS:
                angles, ratios = zip(*ALIGNMENT_POINTS[name])
                ratio = float(np.interp(value, angles, ratios))
            else:
                lower, upper = JOINT_LIMITS[name]
                ratio = float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
            # MCP roll of the four fingers is the only non-inverted channel.
            if not name.endswith("MCP_ROLL"):
                ratio = 1.0 - ratio
            values[name] = int(np.clip(round(ratio * 255.0), 0, 255))
        roll = [values["THUMB_CMC_ROLL"], 128, 128, 128, 128]
        yaw = [values["THUMB_CMC_YAW"]] + [values[f"{f}_MCP_ROLL"] for f in ("INDEX", "MIDDLE", "RING", "PINKY")]
        root = [values["THUMB_CMC_PITCH"]] + [values[f"{f}_MCP_PITCH"] for f in ("INDEX", "MIDDLE", "RING", "PINKY")]
        distal = [values["THUMB_MCP"]] + [values[f"{f}_PIP"] for f in ("INDEX", "MIDDLE", "RING", "PINKY")]
        return roll + yaw + root + distal * 3

    @staticmethod
    def open_palm_registers() -> list[int]:
        return [128] * 10 + [255] * 20


def crc16(data: bytes) -> int:
    """Modbus RTU CRC-16, transmitted least-significant byte first."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


class L20RS485:
    """Synchronous Modbus write-multiple-registers transport."""

    def __init__(self, port="/dev/ttyUSB0", baudrate=460800, slave_id=0x29, timeout=0.05):
        if not 1 <= slave_id <= 247 or baudrate <= 0 or not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid L20 slave id, baudrate or timeout")
        try:
            import serial
        except ImportError as exc:
            raise ImportError("L20 hardware needs pyserial: python -m pip install pyserial") from exc
        self.slave_id = slave_id
        self.timeout = timeout
        kwargs = {"exclusive": True} if sys.platform.startswith("linux") else {}
        self.serial = serial.Serial(
            port=port, baudrate=baudrate, bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
            timeout=timeout, write_timeout=timeout, **kwargs,
        )

    def _read_exact(self, count: int, deadline: float) -> bytes:
        result = bytearray()
        while len(result) < count:
            if time.monotonic() >= deadline:
                raise TimeoutError("L20 Modbus response timed out")
            chunk = self.serial.read(count - len(result))
            if not chunk:
                raise TimeoutError("Incomplete L20 Modbus response")
            result.extend(chunk)
        return bytes(result)

    def write_registers(self, start: int, values) -> None:
        registers = list(values)
        if not 1 <= len(registers) <= 123 or not 0 <= start <= 65536 - len(registers):
            raise ValueError("Invalid Modbus register address/count")
        if any(not isinstance(v, (int, np.integer)) or not 0 <= v <= 255 for v in registers):
            raise ValueError("L20 register values must be integers in [0, 255]")
        payload = struct.pack(">BBHHB", self.slave_id, 0x10, start, len(registers), len(registers) * 2)
        payload += struct.pack(">" + "H" * len(registers), *registers)
        frame = payload + struct.pack("<H", crc16(payload))
        self.serial.reset_input_buffer()
        if self.serial.write(frame) != len(frame):
            raise OSError("Incomplete L20 Modbus write")
        self.serial.flush()
        deadline = time.monotonic() + self.timeout
        header = self._read_exact(2, deadline)
        response = header + self._read_exact(3 if header[1] & 0x80 else 6, deadline)
        if crc16(response[:-2]) != struct.unpack("<H", response[-2:])[0]:
            raise RuntimeError("L20 Modbus response CRC mismatch")
        if response[0] != self.slave_id:
            raise RuntimeError("L20 Modbus response slave id mismatch")
        if response[1] == 0x90:
            raise RuntimeError(f"L20 Modbus exception code: {response[2]}")
        if response[:-2] != payload[:6]:
            raise RuntimeError("L20 Modbus response function/address/count mismatch")

    def configure(self, speed=200, current_limit=200, clear_faults=False) -> None:
        if any(not isinstance(v, int) or not 0 <= v <= 255 for v in (speed, current_limit)):
            raise ValueError("L20 speed/current-limit must be integers in [0, 255]")
        self.write_registers(30, [speed] * 30)
        self.write_registers(150, [current_limit] * 30)
        if clear_faults:
            self.write_registers(90, [1] * 30)
            time.sleep(0.02)
            self.write_registers(90, [0] * 30)

    def set_positions(self, registers) -> None:
        if len(registers) != 30:
            raise ValueError("L20 position command requires 30 registers")
        self.write_registers(0, registers)

    def close(self) -> None:
        self.serial.close()


class L20Output:
    """Mapped output; dry-run never imports serial or opens a device."""

    def __init__(
        self, joint_names, *, port="/dev/ttyUSB0", baudrate=460800,
        slave_id=0x29, speed=200, current_limit=200, clear_faults=False,
        open_on_exit=False, print_registers=False, dry_run=False,
    ):
        self.mapper = L20Mapper(joint_names)
        self.open_on_exit = open_on_exit
        self.print_registers = print_registers or dry_run
        self.last_registers = None
        self.send_count = 0
        self._hand = None
        self._closed = False
        if not dry_run:
            hand = L20RS485(port, baudrate, slave_id)
            try:
                hand.configure(speed, current_limit, clear_faults)
            except BaseException:
                hand.close()
                raise
            self._hand = hand

    def send(self, qpos: np.ndarray) -> bool:
        if self._closed:
            raise RuntimeError("L20 output is closed")
        registers = self.mapper.to_registers(qpos)
        if registers == self.last_registers:
            return False
        if self.print_registers:
            print(f"L20 registers: {registers}")
        if self._hand is not None:
            self._hand.set_positions(registers)
        self.last_registers = registers
        self.send_count += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._hand is not None:
            try:
                if self.open_on_exit:
                    self._hand.set_positions(self.mapper.open_palm_registers())
                    time.sleep(0.2)
            finally:
                self._hand.close()
