# Расчёт Auto PA на стороне Linux, вне процесса Klipper.
# This file may be distributed under the terms of the GNU GPLv3 license.
import csv
import json
import math
import os
import sys

ADVANCES = (0., .02, .04, .06, .08)
COEFFICIENT = 1.7


def load_trace(stream):
    import numpy as np
    groups = {}
    signature = None
    temperatures = []
    for row in csv.DictReader(stream):
        pa = float(row['advance'])
        meta = tuple(float(row[k]) for k in (
            'run_current', 'hold_current', 'electrical_period', 'fast',
            'xy_accel', 'poll_ms', 'smooth_time', 'uart_baud'))
        if (not all(math.isfinite(x) for x in meta)
                or meta[3:] != (8., 1000., 4., .04, 40000.)
                or any(abs(x - .6) > .04 for x in meta[:2])
                or meta[2] <= 0. or row['source'] != 'mcu'
                or int(row['burst_errors']) != 0 or pa not in ADVANCES):
            raise ValueError('Incompatible Auto PA trace settings')
        if signature is None:
            signature = meta
        if meta != signature:
            raise ValueError('Auto PA trace settings changed')
        run = int(row['run'])
        stroke = int(row['stroke'])
        start = float(row['cruise_start']) if stroke > 0 else 0.
        values = (float(row['t']), float(row['e']), float(row['sg']),
                  stroke, start)
        temp = float(row['target'])
        if not all(math.isfinite(x) for x in values + (temp,)):
            raise ValueError('Invalid Auto PA samples')
        if not 0 <= values[2] <= 1023:
            raise ValueError('Invalid SG value')
        if pa not in groups:
            groups[pa] = (run, [])
            temperatures.append(temp)
        if (groups[pa][0] != run or len(groups[pa][1]) >= 2000
                or temp != temperatures[0]):
            raise ValueError('Duplicate, inconsistent or oversized Auto PA run')
        groups[pa][1].append(values)
    if set(groups) != set(ADVANCES) or len(set(temperatures)) != 1:
        raise ValueError('Incomplete Auto PA trace')
    result = {}
    for pa, (_, rows) in groups.items():
        arr = np.asarray(rows, dtype=float).T
        t, e, sg, stroke, start = arr
        dt = np.diff(t)
        if len(t) < 1200 or np.any(dt < .0035) or np.any(dt > .0045):
            raise ValueError('Incomplete or irregular Auto PA sampling')
        windows = {}
        for st in range(2, 7):
            b = start[stroke == st]
            if (not len(b) or np.any(b != b[0])
                    or not t[0] < b[0]-.1 < b[0]+.1 < t[-1]):
                raise ValueError('Incomplete Auto PA stroke')
            windows[st] = float(b[0])
        result[pa] = (t, e, sg, windows)
    return signature, temperatures[0], result


def prepare_baseline(path, settings_path, paths):
    if len(paths) != 2:
        raise ValueError('Two empty extruder traces required')
    bases = []
    for name in paths:
        with open(name, encoding='utf-8') as f:
            bases.append(load_trace(f))
    if bases[0][:2] != bases[1][:2]:
        raise ValueError('Incompatible empty extruder traces')
    null = [fit(statistics(bases[i], [[bases[1-i]]]))['gain'] for i in (0, 1)]
    result = {'baseline_null_gain': null, 'valid': max(null) < 1.}
    if not result['valid']:
        return result
    # Для интерполяции нужны только t, SG и границы разгона.
    # Не округляем время и не усредняем независимые записи.
    data = {'format': 1, 'signature': bases[0][0], 'temperature': bases[0][1],
            'null_gain': null, 'traces': [
                [[t.tolist(), sg.astype(int).tolist(),
                  [windows[st] for st in range(2, 7)]]
                 for t, _, sg, windows in (base[2][pa] for pa in ADVANCES)]
                for base in bases]}
    with open(settings_path, encoding='utf-8') as f:
        data['settings'] = json.load(f)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'), allow_nan=False)
    return result


def read_baseline(path):
    import numpy as np
    if os.path.getsize(path) > 2_000_000:
        raise ValueError('Invalid processed baseline; repeat TEST_PA_EMPTY')
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if (data['format'] != 1 or len(data['traces']) != 2
            or not isinstance(data['settings'], dict)):
        raise ValueError('Unsupported Auto PA baseline format')
    signature = tuple(data['signature'])
    temp, null = data['temperature'], data['null_gain']
    if (len(signature) != 8 or signature[3:] != (8., 1000., 4., .04, 40000.)
            or not all(math.isfinite(x) for x in signature + (temp,))
            or any(abs(x-.6) > .04 for x in signature[:2])
            or signature[2] <= 0. or len(null) != 2
            or not all(math.isfinite(x) and 0. <= x < 1. for x in null)):
        raise ValueError('Invalid Auto PA baseline metadata')
    bases = []
    for trace in data['traces']:
        if len(trace) != len(ADVANCES):
            raise ValueError('Incomplete Auto PA baseline')
        groups = {}
        for pa, (t, sg, starts) in zip(ADVANCES, trace):
            t, sg = np.asarray(t, dtype=float), np.asarray(sg, dtype=float)
            if (t.ndim != 1 or sg.shape != t.shape or not 1200 <= len(t) <= 2000
                    or not np.all(np.isfinite(t)) or not np.all(np.isfinite(sg))
                    or np.any(sg < 0.) or np.any(sg > 1023.)
                    or np.any(sg != np.floor(sg))
                    or np.any(np.diff(t) < .0035) or np.any(np.diff(t) > .0045)
                    or len(starts) != 5 or not all(
                        math.isfinite(b) and t[0] < b-.1 < b+.1 < t[-1]
                        for b in starts)):
                raise ValueError('Invalid Auto PA baseline samples')
            groups[pa] = (t, None, sg, dict(zip(range(2, 7), starts)))
        bases.append((signature, temp, groups))
    return bases, null, data['settings']


def smooth(x):
    import numpy as np
    # 20 мс при 250 Гц; отражение границы как в uniform_filter1d.
    pad = [(2, 2)] + [(0, 0)] * (x.ndim - 1)
    y = np.pad(x, pad, mode='symmetric')
    return sum(y[i:i+len(x)] for i in range(5)) / 5.


def statistics(trace, bases):
    import numpy as np
    taus = np.arange(5, 151) * .001
    out = [np.zeros((3, len(taus))) for _ in bases]
    for pa, (t, e, sg, windows) in trace[2].items():
        dt = np.diff(t)
        step = np.median(dt)
        # Savitzky-Golay, 3 отсчёта, полином второй степени.
        v = np.zeros_like(e)
        a = np.zeros_like(e)
        v[1:-1] = (e[2:] - e[:-2]) / (2*step)
        a[1:-1] = (e[2:] - 2*e[1:-1] + e[:-2]) / step**2
        p = np.empty((len(t), len(taus)))
        p[0] = 1.
        decay = np.exp(-dt[:, None] / taus)
        for i in range(1, len(t)):
            p[i] = p[i-1]*decay[i-1] + (e[i]-e[i-1])/dt[i-1]*(1-decay[i-1])
        p = smooth(p)
        for st, b in windows.items():
            m = np.abs(t-b) <= .08
            vv, aa = v[m], a[m]
            X = np.column_stack((np.ones(m.sum()), vv, np.abs(vv), aa,
                                 vv**2, np.abs(vv)**3, aa**2, vv*aa))
            deltas = []
            for base in bases:
                signals = []
                for bt in base:
                    et, _, esg, ew = bt[2][pa]
                    signals.append(np.interp(t+ew[st]-b, et, esg))
                deltas.append(smooth(np.mean(signals, axis=0)-sg)[m])
            Y = np.column_stack((*deltas, p[m]))
            R = Y-X@np.linalg.lstsq(X, Y, rcond=None)[0]
            q = R[:, len(bases):]
            pp = np.sum(q*q, axis=0)
            for j, stats in enumerate(out):
                y = R[:, j]
                stats[0] += y@y
                stats[1] += q.T@y
                stats[2] += pp
    return out


def fit(stats):
    import numpy as np
    yy, py, pp = sum(stats)
    beta = np.maximum(0, py/np.maximum(pp, 1e-12))
    gain = (2*beta*py-beta**2*pp)/np.maximum(yy, 1e-12)
    i = int(gain.argmax())
    return {'pa': (i+5)*.001, 'gain': float(gain[i]*100),
            'beta': float(beta[i]), 'index': i}


def score(stats, model):
    yy, py, pp = stats[:, model['index']]
    b = model['beta']
    return float(100*(2*b*py-b*b*pp)/max(yy, 1e-12))


def analyze(baseline, paths):
    bases, null, settings = read_baseline(baseline)
    if not paths:
        return {'baseline_null_gain': null, 'valid': max(null) < 1.,
                'settings': settings}
    all_stats = []
    temps = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            trace = load_trace(f)
        if trace[0] != bases[0][0]:
            raise ValueError('Settings differ from empty extruder calibration')
        temps.append(trace[1])
        all_stats.append(statistics(trace, [bases, bases[:1], bases[1:]]))
    if len(set(temps)) != 1:
        raise ValueError('Temperature changed between passes')
    pooled = fit([s[0] for s in all_stats])
    own = [fit([s[0]]) for s in all_stats]
    sensitivity = [fit([s[j] for s in all_stats]) for j in (1, 2)]
    cv, cross = [], []
    if len(all_stats) > 1:
        for i, s in enumerate(all_stats):
            train = [x for k, x in enumerate(all_stats) if k != i]
            cv.append(score(s[0], fit([x[0] for x in train])))
            for j in (1, 2):
                cross.append(score(s[3-j], fit([x[j] for x in train])))
    pa = pooled['pa']
    stable = [x['pa'] for x in own + sensitivity]
    valid = (len(paths) >= 3 and .005 < pa < .150 and pooled['gain'] >= 1.
             and min(cv+cross) > 0. and max(stable)-min(stable) <= .03
             and max(null) < 1.)
    return {'raw_pa': pa, 'pa': pa*COEFFICIENT, 'coefficient': COEFFICIENT,
            'gain': pooled['gain'], 'passes': own, 'baseline_sensitivity': sensitivity,
            'cv_gain': cv, 'cross_gain': cross, 'baseline_null_gain': null,
            'temperature': temps[0], 'baseline_temperature': bases[0][1],
            'valid': bool(valid)}


if __name__ == '__main__':
    if sys.argv[1:] == ['--check']:
        import numpy
        print(json.dumps({'numpy': numpy.__version__}))
    elif sys.argv[1] == '--prepare':
        print(json.dumps(prepare_baseline(sys.argv[2], sys.argv[3], sys.argv[4:]),
                         allow_nan=False))
    else:
        print(json.dumps(analyze(sys.argv[1], sys.argv[2:]), allow_nan=False))
