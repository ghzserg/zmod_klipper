# Проверка хостового протокола без принтера.
import importlib.util
import pathlib
import struct
import sys
import types
import unittest
from contextlib import nullcontext

sys.modules.setdefault('chelper', types.ModuleType('chelper'))
path = pathlib.Path(__file__).resolve().parents[1] / 'klippy/extras/zmod_pa.py'
spec = importlib.util.spec_from_file_location('zmod_pa_test_module', path)
pa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pa)

FREQ = 144000000
CLOCK = 0xffff0000


class MCU:
    def __init__(self):
        self.calls = []
        self.reject = 0
        self.bad = False
        self.invalid = False
        self.count = 0

    def try_lookup_command(self, fmt):
        return True

    def lookup_query_command(self, fmt, *args, **kwargs):
        name = fmt.split()[0]
        return types.SimpleNamespace(send=lambda values: self.send(name, values))

    def print_time_to_clock(self, t):
        return CLOCK + round(t * FREQ)

    def clock_to_print_time(self, clock):
        return (clock - CLOCK) / FREQ

    def send(self, name, values):
        self.calls.append(name)
        token = values[1]
        if name == 'tmcuart_buffer_start':
            self.count = values[3]
            self.clock = values[4]
            return dict(token=token, status=self.reject, available=80000)
        if name == 'tmcuart_buffer_status':
            return dict(token=token, status=0, total=self.count,
                        errors=int(self.invalid), query_min=250000,
                        query_max=260000)
        first = values[2]
        raw = b''.join(struct.pack('<IH',
            (self.clock + i*288000) & 0xffffffff,
            0xffff if self.invalid and i == 3 else 200)
            for i in range(first, min(first+7, self.count)))
        return dict(token=token, status=0, first=first,
                    data=raw[:-1] if self.bad else raw)


class TestBuffer(unittest.TestCase):
    def setUp(self):
        self.mcu = MCU()
        self.shutdown = []
        self.messages = []
        printer = types.SimpleNamespace(is_shutdown=lambda: bool(self.shutdown),
                                        invoke_shutdown=self.shutdown.append)
        self.obj = pa.ZModPA.__new__(pa.ZModPA)
        self.obj.printer = printer
        self.obj.reactor = types.SimpleNamespace(monotonic=lambda: 0.)
        self.obj.sg_token = 0
        self.gcmd = types.SimpleNamespace(error=ValueError,
                                          respond_info=self.messages.append)
        uart = types.SimpleNamespace(analog_mux=None, instances={0: True},
                                     requested_baud=80000, oid=1, cmd_queue=1)
        self.driver = types.SimpleNamespace(get_mcu=lambda: self.mcu,
                                            mcu_uart=uart, addr=0,
                                            mutex=nullcontext())
        self.stepper = types.SimpleNamespace(get_mcu=lambda: self.mcu)

    def capture(self):
        return self.obj.capture_sg(self.gcmd, self.driver, self.stepper,
                                   .10, .025, 2, 1)

    def test_complete_and_wrap(self):
        with self.capture() as (rows, counts):
            self.assertEqual(self.mcu.calls, ['tmcuart_buffer_start'])
            self.assertEqual(rows, [])
        self.assertEqual(counts, [63, 0])
        self.assertEqual(len(rows), 63)
        self.assertAlmostEqual(rows[-1][0] - rows[0][0], .124)
        self.assertFalse(self.shutdown)
        self.assertIn('80000 bytes', self.messages[0])

    def test_memory_refusal(self):
        self.mcu.reject = 3
        with self.assertRaisesRegex(ValueError, 'available'):
            with self.capture():
                self.fail('must not move')
        self.assertFalse(self.shutdown)

    def test_bad_packet_after_stop(self):
        self.mcu.bad = True
        with self.assertRaisesRegex(ValueError, 'packet'):
            with self.capture():
                pass
        self.assertFalse(self.shutdown)

    def test_invalid_sample(self):
        self.mcu.invalid = True
        with self.capture() as (rows, counts):
            pass
        self.assertEqual(counts, [63, 1])
        self.assertEqual(len(rows), 62)

    def test_move_failure(self):
        with self.assertRaises(RuntimeError):
            with self.capture():
                raise RuntimeError('move failed')
        self.assertTrue(self.shutdown)


if __name__ == '__main__':
    unittest.main()
