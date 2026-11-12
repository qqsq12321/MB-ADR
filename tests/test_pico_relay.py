"""Loopback-only XRoboToolkit framing -> bundled relay -> receiver checks."""

import json
import socket
import struct
import threading
import time
import unittest

from mbadr.pico4 import _RelayFrameParser
from mbadr.pico_relay import Pico4Daemon, PortBindingError


class RelayTests(unittest.TestCase):
    def setUp(self):
        self.daemon = Pico4Daemon(direct_port=0, relay_host="127.0.0.1", relay_port=0,
                                 broadcast_port=29888, device_id="pico4",
                                 direct_host="127.0.0.1", broadcast=False)
        self.errors = []

        def run():
            try:
                self.daemon.run()
            except Exception as exc:
                self.errors.append(exc)

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        self.assertTrue(self.daemon.ready.wait(3), self.errors)
        self.sockets = []

    def tearDown(self):
        for conn in self.sockets:
            conn.close()
        self.daemon.stop()
        self.thread.join(timeout=4)
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.errors, [])

    def connect(self, port):
        conn = socket.create_connection(("127.0.0.1", port), timeout=2)
        self.sockets.append(conn)
        return conn

    def read_payload(self, conn):
        parser = _RelayFrameParser()
        while True:
            parser.feed(conn.recv(65536))
            result = parser.try_parse()
            if result is not None:
                return result

    def test_forward_fragmented_json_to_multiple_clients_and_invalidate_disconnect(self):
        direct_port, relay_port = self.daemon.bound_ports
        clients = [self.connect(relay_port), self.connect(relay_port)]
        deadline = time.monotonic() + 2
        while len(self.daemon._hub._clients) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(len(self.daemon._hub._clients), 2)
        headset = self.connect(direct_port)
        payload = b'{"Hand":{"rightHand":{"isActive":1,"count":26}}}'
        frame = b"\x3f\x6d" + struct.pack("<I", len(payload)) + payload + struct.pack("<Q", 123) + b"\xa5"
        headset.sendall(frame[:11])
        time.sleep(0.005)
        headset.sendall(frame[11:])
        for client in clients:
            self.assertEqual(self.read_payload(client), payload)
        headset.close()
        for client in clients:
            inactive = json.loads(self.read_payload(client))
            self.assertEqual(inactive["Hand"]["rightHand"]["isActive"], 0)

    def test_bind_failure_is_synchronous(self):
        direct_port, _ = self.daemon.bound_ports
        other = Pico4Daemon(direct_port=direct_port, relay_host="127.0.0.1", relay_port=0,
                            broadcast_port=29888, device_id="other", direct_host="127.0.0.1",
                            broadcast=False)
        with self.assertRaises(PortBindingError):
            other.run()


if __name__ == "__main__":
    unittest.main()
