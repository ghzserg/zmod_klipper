# Проверки математики и восстановления датчиков без принтера.
import importlib.util
import pathlib
import json
import zipfile
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
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

    def test_dated_directories(self):
        obj = pa.ZModPA.__new__(pa.ZModPA)
        with tempfile.TemporaryDirectory() as root:
            obj.baseline_dir = root
            with patch.object(pa.time, 'strftime', return_value='20261127_213000'):
                first = pathlib.Path(obj.new_run_dir('empty'))
                second = pathlib.Path(obj.new_run_dir('empty'))
                loaded = pathlib.Path(obj.new_run_dir('pa'))
            self.assertEqual(first.name, '20261127_213000_empty')
            self.assertEqual(second.name, '20261127_213000_empty_1')
            self.assertEqual(loaded.name, '20261127_213000_pa')
            self.assertTrue(all(x.is_dir() for x in (first, second, loaded)))

    def test_archive(self):
        obj = pa.ZModPA.__new__(pa.ZModPA)
        messages = []
        gcmd = types.SimpleNamespace(respond_info=messages.append, error=ValueError)
        with tempfile.TemporaryDirectory() as root:
            obj.baseline_dir = root
            with patch.object(pa.time, 'strftime', return_value='20261127_213000'):
                for fail in (False, True):
                    try:
                        with obj.run_archive(gcmd, 'pa') as folder:
                            pathlib.Path(folder, 'trace.csv').write_text('sample')
                            if fail:
                                raise RuntimeError('test failure')
                    except RuntimeError:
                        self.assertTrue(fail)
                    self.assertFalse(pathlib.Path(folder).exists())
                    with zipfile.ZipFile(folder + '.zip') as z:
                        self.assertEqual(z.namelist(), ['trace.csv'])
                        self.assertEqual(z.read('trace.csv'), b'sample')
            self.assertEqual(sorted(x.name for x in pathlib.Path(root).iterdir()),
                             ['20261127_213000_pa.zip', '20261127_213000_pa_1.zip'])

    def test_processed_baseline(self):
        t = 34567. + np.arange(1475)*.004
        e = t*t
        sg = (np.arange(len(t)) % 89 + 130).astype(float)
        starts = dict(zip(range(2, 7), t[[200, 400, 600, 800, 1000]]))
        signature = (.6, .6, .08692, 8., 1000., 4., .04, 40000.)
        base = (signature, 210., {x: (t, e, sg, starts) for x in fit.ADVANCES})
        with tempfile.TemporaryDirectory() as root:
            settings = pathlib.Path(root, 'settings.json')
            settings.write_text('{}')
            path = pathlib.Path(root, 'empty.zip')
            trace = pathlib.Path(root, 'trace.csv')
            trace.touch()
            with patch.object(fit, 'load_trace', return_value=base):
                result = fit.prepare_baseline(path, settings, [trace, trace])
            self.assertTrue(result['valid'])
            with zipfile.ZipFile(path) as z:
                self.assertEqual(set(z.namelist()), {'settings.json', 'baseline.json'})
            bases, null = fit.read_baseline(path)
            self.assertEqual(null, result['baseline_null_gain'])
            for b in bases:
                self.assertEqual(b[:2], base[:2])
                for x in fit.ADVANCES:
                    bt, be, bs, bw = b[2][x]
                    self.assertIsNone(be)
                    np.testing.assert_array_equal(bt, t)
                    np.testing.assert_array_equal(bs, sg)
                    self.assertEqual(bw, starts)
            original = fit.statistics(base, [[base]])[0]
            processed = fit.statistics(base, [bases[:1]])[0]
            np.testing.assert_array_equal(original, processed)

    def test_no_sensors(self):
        obj = pa.ZModPA.__new__(pa.ZModPA)
        obj.printer = types.SimpleNamespace(lookup_objects=lambda kind: [])
        with obj.suspend_filament_sensors():
            pass

if __name__ == '__main__':
    unittest.main()
