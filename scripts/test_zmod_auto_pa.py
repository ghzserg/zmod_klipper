# Проверки математики и восстановления датчиков без принтера.
import importlib.util
import pathlib
import sys
import types
import unittest
import numpy as np

sys.modules.setdefault('chelper', types.ModuleType('chelper'))
root = pathlib.Path(__file__).resolve().parents[1] / 'klippy/extras'

def load(name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

fit = load('zmod_pa_fit')
pa = load('zmod_pa')

class TestAutoPA(unittest.TestCase):
    def test_filter(self):
        np.testing.assert_allclose(fit.smooth(np.arange(5.)),
                                   [.8, 1.2, 2., 2.8, 3.2])

    def test_known_candidate_and_null(self):
        data = np.zeros((3, 146))
        data[0] = 100
        data[2] = 4
        data[1, 32] = 10
        result = fit.fit([data])
        self.assertAlmostEqual(result['pa'], .037)
        self.assertAlmostEqual(result['gain'], 25.)
        self.assertAlmostEqual(fit.score(data, result), 25.)
        data[1] = 0
        self.assertEqual(fit.fit([data])['gain'], 0.)

    def test_empty_trace(self):
        import io
        with self.assertRaises(ValueError):
            fit.load_trace(io.StringIO('advance,t,e,sg\n'))

    def test_sensor_restore(self):
        for failure in (False, True):
            enabled = types.SimpleNamespace(sensor_enabled=True)
            disabled = types.SimpleNamespace(sensor_enabled=False)
            obj = pa.ZModPA.__new__(pa.ZModPA)
            obj.printer = types.SimpleNamespace(lookup_objects=lambda kind:
                [('sensor a', types.SimpleNamespace(runout_helper=enabled)),
                 ('sensor b', types.SimpleNamespace(runout_helper=disabled))]
                if kind == 'filament_switch_sensor' else [])
            try:
                with obj.suspend_filament_sensors():
                    self.assertFalse(enabled.sensor_enabled)
                    self.assertFalse(disabled.sensor_enabled)
                    # Даже если сторонний макрос изменил состояние датчика.
                    disabled.sensor_enabled = True
                    if failure:
                        raise RuntimeError('test')
            except RuntimeError:
                self.assertTrue(failure)
            self.assertTrue(enabled.sensor_enabled)
            self.assertFalse(disabled.sensor_enabled)

    def test_no_sensors(self):
        obj = pa.ZModPA.__new__(pa.ZModPA)
        obj.printer = types.SimpleNamespace(lookup_objects=lambda kind: [])
        with obj.suspend_filament_sensors():
            pass

if __name__ == '__main__':
    unittest.main()
