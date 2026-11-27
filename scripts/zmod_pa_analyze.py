#!/usr/bin/env python3
# Экспериментальный анализ Auto PA на ПК, не часть процесса Klipper.
# This file may be distributed under the terms of the GNU GPLv3 license.
import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

TAUS = np.arange(5, 151) * .001
SETTINGS = ('fast', 'target', 'run_current', 'hold_current',
            'smooth_time', 'electrical_period')
REQUIRED = set(SETTINGS + ('run', 'advance', 't', 'e', 'sg', 'v', 'a',
                          'stroke', 'phase', 'cruise_start', 'cruise_end'))


def read_trace(source):
    data = np.atleast_1d(np.genfromtxt(source, delimiter=',', names=True,
                                     dtype=None, encoding='utf-8'))
    if not REQUIRED.issubset(data.dtype.names or ()) or not len(data):
        raise ValueError('Expected new ZMOD_TEST_PA CSV format')
    return data


def load(paths):
    traces = []
    for path in paths:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                for name in sorted(z.namelist()):
                    if name.lower().endswith('.csv'):
                        traces.append((str(path) + ':' + name,
                                       read_trace(io.BytesIO(z.read(name)))))
        else:
            traces.append((str(path), read_trace(path)))
    if not traces:
        raise ValueError('No CSV traces')
    for key in SETTINGS:
        expected = traces[0][1][key][0]
        if any(not np.allclose(d[key], expected, rtol=1.e-5, atol=1.e-6)
               for _, d in traces):
            raise ValueError('Do not mix different settings: ' + key)
    return traces


def load_baseline(paths, reference):
    # Калибровка только для того же двигателя и того же тока.
    keys = ('run_current', 'hold_current', 'electrical_period')
    required = set(keys + ('speed', 'sg', 'phase', 't',
                           'cruise_start', 'cruise_end', 'repeat'))
    groups = {}
    for path in paths:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                parts = [z.read(n) for n in z.namelist()
                         if n.lower().endswith('.csv')]
        else:
            parts = [path.read_bytes()]
        for part in parts:
            d = np.atleast_1d(np.genfromtxt(io.BytesIO(part), delimiter=',',
                                          names=True, dtype=None, encoding='utf-8'))
            if not len(d) or not required.issubset(d.dtype.names or ()):
                raise ValueError('Expected ZMOD_TEST_SG CSV format')
            if any(not np.allclose(d[k], reference[k][0], rtol=1.e-5,
                                   atol=1.e-6) for k in keys):
                continue
            m = ((d['phase'] == 'fast') & (d['t']-d['cruise_start'] >= .1)
                 & (d['cruise_end']-d['t'] >= .1))
            for speed in np.unique(d['speed'][m]):
                for repeat in np.unique(d['repeat'][m & (d['speed'] == speed)]):
                    sg = d['sg'][m & (d['speed'] == speed) & (d['repeat'] == repeat)]
                    if len(sg) >= 20 and np.all(np.isfinite(sg)):
                        groups.setdefault(float(speed), []).append(np.median(sg))
    speeds = np.array(sorted(k for k, v in groups.items() if len(v) >= 2))
    if len(speeds) < 4 or speeds[0] > reference['fast'][0] or speeds[-1] < reference['fast'][0]:
        raise ValueError('Insufficient matching unloaded SG calibration')
    levels = np.array([np.median(groups[k]) for k in speeds])
    print('Unloaded SG curve: ' + ', '.join('%.1f:%.1f' % (v, sg)
                                          for v, sg in zip(speeds, levels)))
    return speeds, levels


def blocks(traces, guard=.05, delay=0., baseline=None):
    result = []
    for name, data in traces:
        for run in np.unique(data['run']):
            g = data[data['run'] == run]
            t, e, v, a, sg = (g[k].astype(float) for k in
                              ('t', 'e', 'v', 'a', 'sg'))
            dt = np.diff(t)
            if (not len(dt) or np.any(dt <= 0.) or np.max(dt) > .10
                    or not np.all(np.isfinite([t, e, v, a, sg]))):
                continue
            # Модель давления первого порядка: p' = (v_E - p) / tau.
            # E — история запланированных шагов, а не энкодер двигателя.
            decay = np.exp(-dt[:, None] / TAUS)
            p = np.empty((len(t), len(TAUS)))
            p[0] = 1.
            for i in range(1, len(t)):
                p[i] = (p[i-1] * decay[i-1] + (e[i]-e[i-1]) / dt[i-1]
                        * (1. - decay[i-1]))
            if delay:
                p = np.column_stack([np.interp(t-delay, t, col) for col in p.T])
                e, v, a = (np.interp(t-delay, t, col) for col in (e, v, a))
            # Не используем переходы, на которых SG зависит от скорости,
            # сглаживания PA и задержки чтения UART сильнее, чем от давления.
            m = ((g['phase'] == 'fast') & (g['stroke'] > 1)
                 & (t-g['cruise_start'] >= guard)
                 & (g['cruise_end']-t >= .05))
            if baseline is None:
                bad = np.unique(g['stroke'][(g['stroke'] > 0) & (sg <= 20)])
            else:
                speeds, levels = baseline
                m &= (v >= speeds[0]) & (v <= speeds[-1])
                empty = np.interp(v, speeds, levels)
                # Эвристика аномально низкого SG, не детектор пропуска шагов.
                # На переходах и вне диапазона калибровки порог не применяем.
                bad = np.unique(g['stroke'][m & (sg <= .1 * empty)])
            m &= ~np.isin(g['stroke'], bad)
            if m.sum() < 20:
                continue
            phase = 2. * np.pi * e[m] / g['electrical_period'][0]
            dv = v[m] - g['fast'][0]
            # Убираем смещения каждого цикла, зависимость от скорости,
            # ускорения и четыре гармоники электрического периода.
            x = np.column_stack([*(g['stroke'][m] == k for k in
                                   np.unique(g['stroke'][m])),
                                 *([dv, dv*dv] if baseline is None else []), a[m],
                                 *(fn(k*phase) for k in range(1, 5)
                                   for fn in (np.cos, np.sin))])
            # Не подгоняем скоростную кривую повторно по нагруженному мотору.
            signal = sg[m] if baseline is None else sg[m] - empty[m]
            y = np.column_stack((signal, p[m]))
            r = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
            y, p = r[:, 0], r[:, 1:]
            result.append((name, run, float(g['advance'][0]), len(y),
                           y@y, p.T@y, np.sum(p*p, axis=0)))
    return result


def fit(rows):
    yy = sum(r[4] for r in rows)
    py = sum(r[5] for r in rows)
    pp = sum(r[6] for r in rows)
    if yy <= 0. or len({r[2] for r in rows}) < 3:
        raise ValueError('Insufficient independent PA levels or SG variation')
    # При росте нагрузки SG должен уменьшаться.
    beta = np.minimum(0., py / np.maximum(pp, 1.e-12))
    gain = 2.*beta*py - beta*beta*pp
    i = int(np.argmax(gain))
    return TAUS[i], beta[i], gain[i]/yy, i


def analyze(traces, baseline=None):
    rows = blocks(traces, baseline=baseline)
    tau, beta, gain, _ = fit(rows)
    print('Samples: %d, runs: %d' % (sum(r[3] for r in rows), len(rows)))
    print('Candidate PA: %.3f; SG coefficient: %.3f; residual gain: %.3f%%'
          % (tau, beta, 100.*gain))
    names = sorted({r[0] for r in rows})
    reasons = []
    if len(names) < 3:
        reasons.append('need at least three independent traces')
    for name in names:
        test = [r for r in rows if r[0] == name]
        train = [r for r in rows if r[0] != name]
        own, _, _, _ = fit(test)
        if not train:
            continue
        tr_tau, b, _, i = fit(train)
        # Подгоняем только параметры помех на исключённой записи;
        # tau и коэффициент нагрузки получены из остальных записей.
        score = sum(2.*b*r[5][i] - b*b*r[6][i] for r in test)
        score /= sum(r[4] for r in test)
        print('%s: own PA %.3f, train PA %.3f, held-out gain %+.3f%%'
              % (name, own, tr_tau, 100.*score))
        if score <= 0.:
            reasons.append('no held-out improvement: ' + name)
        if abs(tr_tau-tau) > .02:
            reasons.append('unstable leave-one-trace-out PA: ' + name)
    candidates = []
    for guard in (.04, .05, .06):
        for delay in (0., .004, .008):
            p, _, improvement, _ = fit(blocks(traces, guard, delay, baseline))
            candidates.append(p)
            print('guard %.0f ms, delay %.0f ms: PA %.3f, gain %.3f%%'
                  % (guard*1000., delay*1000., p, improvement*100.))
    lo, hi = min(candidates), max(candidates)
    print('Sensitivity range (not a confidence interval): %.3f-%.3f' % (lo, hi))
    # Предварительные эвристики, не доказательство точности PA.
    if gain < .01:
        reasons.append('residual gain below 1%')
    if hi-lo > .03 or lo <= TAUS[0] or hi >= TAUS[-1]:
        reasons.append('unstable or boundary sensitivity result')
    if reasons:
        print('REJECTED: ' + '; '.join(reasons))
        return False
    print('EXPERIMENTAL CANDIDATE ONLY: validate against printed PA test')
    return True


def main():
    parser = argparse.ArgumentParser(description='Offline experimental Auto PA')
    parser.add_argument('paths', nargs='+', type=Path,
                        help='CSV/ZIP traces of one filament and equal settings')
    parser.add_argument('--baseline', nargs='+', type=Path,
                        help='Unloaded ZMOD_TEST_SG CSV/ZIP, matching current')
    args = parser.parse_args()
    try:
        traces = load(args.paths)
        baseline = load_baseline(args.baseline, traces[0][1]) if args.baseline else None
        return 0 if analyze(traces, baseline) else 1
    except (ValueError, OSError, np.linalg.LinAlgError) as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
