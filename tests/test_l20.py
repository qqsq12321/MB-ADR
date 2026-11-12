"""Hardware-free checks for L20 calibration and the Modbus wire protocol."""

import contextlib
import io
import os
import select
import struct
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mbadr.chain import KinematicChain
from mbadr.config import HandSpec
from mbadr.l20 import ACTIVE_JOINTS, L20Mapper, L20Output, L20RS485, crc16


def with_crc(payload):
    return payload + struct.pack("<H", crc16(payload))


class FakeSerial:
    def __init__(self, responder=None):
        self.responder = responder or (lambda frame: with_crc(frame[:6]))
        self.writes = []
        self.pending = b""
        self.closed = False

    def reset_input_buffer(self):
        self.pending = b""

    def write(self, frame):
        self.writes.append(frame)
        self.pending = self.responder(frame)
        return len(frame)

    def flush(self):
        pass

    def read(self, count):
        # Deliberately fragment both the header and the rest of the reply.
        count = min(count, 1)
        chunk, self.pending = self.pending[:count], self.pending[count:]
        return chunk

    def close(self):
        self.closed = True


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = KinematicChain.from_urdf(HandSpec.from_name("l20").urdf_path)
        cls.mapper = L20Mapper(cls.chain.joint_names)

    def test_calibrated_zero_pose(self):
        # Golden fixture from the existing GEORT/AnyDex real-hand mapper.
        expected = [253, 128, 128, 128, 128, 255, 128, 128, 128, 128] + [255] * 20
        self.assertEqual(self.mapper.to_registers(np.zeros(self.chain.nq)), expected)

    def test_calibrated_upper_pose(self):
        expected = [0, 128, 128, 128, 128, 0, 255, 255, 255, 255, 15, 32, 32, 32, 32] + [0] * 15
        self.assertEqual(self.mapper.to_registers(self.chain.joint_limits()[:, 1]), expected)

    def test_joint_order_and_side_prefix_do_not_change_mapping(self):
        rng = np.random.default_rng(42)
        q = rng.uniform(*self.chain.joint_limits().T)
        order = rng.permutation(self.chain.nq)
        expected = self.mapper.to_registers(q)
        for side in ("left", "right"):
            names = [f"{side}_{self.chain.joint_names[i]}" for i in order]
            self.assertEqual(L20Mapper(names).to_registers(q[order]), expected)

    def test_missing_ambiguous_and_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            L20Mapper(ACTIVE_JOINTS[:-1])
        with self.assertRaises(ValueError):
            L20Mapper(ACTIVE_JOINTS + ("right_THUMB_MCP",))
        for q in (np.zeros(16), np.full(self.chain.nq, np.nan), np.full(self.chain.nq, np.inf)):
            with self.assertRaises(ValueError):
                self.mapper.to_registers(q)

    def test_out_of_range_angles_are_clipped(self):
        for value in (-100.0, 100.0):
            registers = self.mapper.to_registers(np.full(self.chain.nq, value))
            self.assertEqual(len(registers), 30)
            self.assertTrue(all(0 <= v <= 255 for v in registers))

    def test_dry_run_and_duplicate_suppression(self):
        with patch("mbadr.l20.L20RS485") as transport, contextlib.redirect_stdout(io.StringIO()):
            output = L20Output(self.chain.joint_names, dry_run=True, open_on_exit=True)
            self.assertTrue(output.send(np.zeros(self.chain.nq)))
            self.assertFalse(output.send(np.zeros(self.chain.nq)))
            output.close()
            output.close()
            transport.assert_not_called()
            self.assertEqual(output.send_count, 1)

    def test_startup_failure_closes_serial(self):
        with patch("mbadr.l20.L20RS485") as transport:
            transport.return_value.configure.side_effect = TimeoutError("no response")
            with self.assertRaises(TimeoutError):
                L20Output(self.chain.joint_names)
            transport.return_value.close.assert_called_once()

    def test_default_exit_sends_no_extra_pose(self):
        with patch("mbadr.l20.L20RS485") as transport:
            output = L20Output(self.chain.joint_names)
            output.send(np.zeros(self.chain.nq))
            output.close()
            self.assertEqual(transport.return_value.set_positions.call_count, 1)
            transport.return_value.close.assert_called_once()


class ProtocolTests(unittest.TestCase):
    def make_transport(self, responder=None):
        port = FakeSerial(responder)
        module = types.SimpleNamespace(Serial=Mock(return_value=port), EIGHTBITS=8, PARITY_NONE="N", STOPBITS_ONE=1)
        with patch.dict("sys.modules", {"serial": module}):
            transport = L20RS485()
        return transport, port

    def test_crc_known_modbus_vector(self):
        self.assertEqual(with_crc(bytes.fromhex("01030000000a")), bytes.fromhex("01030000000ac5cd"))

    def test_write_layout_and_fragmented_acknowledgement(self):
        transport, port = self.make_transport()
        transport.set_positions(list(range(30)))
        frame = port.writes[0]
        self.assertEqual(frame[:7], bytes.fromhex("29100000001e3c"))
        self.assertEqual(struct.unpack(">30H", frame[7:-2]), tuple(range(30)))
        self.assertEqual(crc16(frame[:-2]), int.from_bytes(frame[-2:], "little"))
        transport.close()
        self.assertTrue(port.closed)

    def test_configuration_registers(self):
        transport, port = self.make_transport()
        transport.configure(speed=80, current_limit=90, clear_faults=True)
        self.assertEqual([int.from_bytes(f[2:4], "big") for f in port.writes], [30, 150, 90, 90])
        self.assertEqual([int.from_bytes(f[7:9], "big") for f in port.writes], [80, 90, 1, 0])

    def test_timeout_crc_exception_and_wrong_echo(self):
        responders = [
            (lambda f: b"", TimeoutError),
            (lambda f: with_crc(f[:6])[:-1] + bytes([with_crc(f[:6])[-1] ^ 1]), RuntimeError),
            (lambda f: with_crc(bytes([0x29, 0x90, 0x02])), RuntimeError),
            (lambda f: with_crc(bytes([0x2A]) + f[1:6]), RuntimeError),
            (lambda f: with_crc(f[:3] + bytes([1]) + f[4:6]), RuntimeError),
            (lambda f: with_crc(f[:1] + bytes([0x06]) + f[2:6]), RuntimeError),
        ]
        for responder, exception in responders:
            with self.subTest(exception=exception, responder=responder):
                transport, _ = self.make_transport(responder)
                with self.assertRaises(exception):
                    transport.set_positions([128] * 30)

    def test_invalid_commands_are_not_written(self):
        transport, port = self.make_transport()
        for values in ([0] * 29, [256] * 30, [-1] * 30, [1.2] * 30):
            with self.assertRaises(ValueError):
                transport.set_positions(values)
        with self.assertRaises(ValueError):
            transport.configure(speed=300)
        self.assertEqual(port.writes, [])

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires a Linux pseudo-terminal")
    def test_real_pyserial_transport_against_virtual_l20(self):
        # A freshly allocated /dev/pts/* endpoint, never a physical device.
        import pty
        master, slave = pty.openpty()
        stop = threading.Event()
        requests = []
        errors = []

        def emulate():
            pending = bytearray()
            try:
                while not stop.is_set():
                    ready, _, _ = select.select([master], [], [], 0.05)
                    if not ready:
                        continue
                    pending.extend(os.read(master, 4096))
                    while len(pending) >= 7:
                        size = 7 + pending[6] + 2
                        if len(pending) < size:
                            break
                        frame = bytes(pending[:size])
                        del pending[:size]
                        if crc16(frame[:-2]) != int.from_bytes(frame[-2:], "little"):
                            raise AssertionError("Invalid request CRC")
                        requests.append(frame)
                        reply = with_crc(frame[:6])
                        os.write(master, reply[:3])
                        stop.wait(0.001)
                        os.write(master, reply[3:])
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=emulate, daemon=True)
        thread.start()
        output = None
        try:
            chain = KinematicChain.from_urdf(HandSpec.from_name("l20").urdf_path)
            output = L20Output(chain.joint_names, port=os.ttyname(slave))
            self.assertTrue(output.send(np.zeros(chain.nq)))
            self.assertEqual(len(requests), 3)  # speed, current limit, position
            self.assertEqual([int.from_bytes(f[2:4], "big") for f in requests], [30, 150, 0])
            self.assertEqual(list(struct.unpack(">30H", requests[-1][7:-2])),
                             L20Mapper(chain.joint_names).to_registers(np.zeros(chain.nq)))
        finally:
            if output is not None:
                output.close()
            stop.set()
            thread.join(timeout=2)
            os.close(master)
            os.close(slave)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
