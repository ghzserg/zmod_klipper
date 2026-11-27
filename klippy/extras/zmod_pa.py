# Диагностический сбор StallGuard для подбора Pressure Advance.
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import csv
import logging
import math
import os
import tempfile
import chelper


def nominal_phases(trapq, start, end, fast, strokes):
    # История E содержит плановую скорость до применения PA.
    ffi, lib = chelper.get_ffi()
    data = ffi.new('struct pull_move[128]')
    count = lib.trapq_extract_old(trapq, data, len(data), start, end)
    if count >= len(data):
        raise ValueError('Too many nominal motion phases')
    phases = []
    for i in range(count - 1, -1, -1):
        m = data[i]
        if m.move_t <= 0. or m.print_time + m.move_t <= start:
            continue
        phases.append((m.print_time, m.print_time + m.move_t,
                       m.start_v, m.accel))
    windows = {}
    stroke = 0
    for i, (a, b, v, acc) in enumerate(phases):
        if abs(acc) > 1.e-6 or abs(v - fast) > 1.e-5:
            continue
        stroke += 1
        first, last = i, i
        if i and phases[i-1][3] > 0. and abs(phases[i-1][1]-a) < 1.e-6:
            first = i - 1
        if i+1 < len(phases) and phases[i+1][3] < 0. and (
                abs(phases[i+1][0]-b) < 1.e-6):
            last = i + 1
        for j in range(first, last + 1):
            label = 'fast' if j == i else ('accel' if j < i else 'decel')
            windows[j] = (stroke, label, phases[first][0], a, b,
                          phases[last][1])
    if stroke != strokes:
        raise ValueError('Expected %d fast strokes, found %d' %
                         (strokes, stroke))
    return phases, windows


class ZModPA:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.gcode = self.printer.lookup_object('gcode')
        self.log_dir = config.get('log_dir', '/opt/config/mod_data/log')
        self.gcode.register_command('ZMOD_TEST_PA', self.cmd_ZMOD_TEST_PA,
                                    desc='Record extruder StallGuard trace')

        self.gcode.register_command('ZMOD_TEST_SG', self.cmd_ZMOD_TEST_SG,
                                    desc='Record unloaded extruder StallGuard')

    def cmd_ZMOD_TEST_SG(self, gcmd):
        # Только пустой экструдер. Защиту штатной экструзии не изменяем.
        if gcmd.get_int('CONFIRM', 0) != 1:
            raise gcmd.error('Remove filament completely, cool hotend, '
                             'enable extruder motor, then use CONFIRM=1')
        try:
            speeds = [float(v) for v in gcmd.get(
                'SPEEDS', '1,2,3,4,5,6,8,10,12,15,18,20').split(',')]
        except ValueError:
            raise gcmd.error('Invalid SPEEDS')
        if (not 2 <= len(speeds) <= 16 or len(set(speeds)) != len(speeds)
                or any(not math.isfinite(v) or not 1. <= v <= 20.
                       for v in speeds)):
            raise gcmd.error('SPEEDS requires 2-16 distinct values in 1-20 mm/s')
        poll_ms = gcmd.get_int('POLL_MS', 2, minval=2, maxval=20)
        repeats = gcmd.get_int('REPEATS', 2, minval=1, maxval=3)

        printer, reactor = self.printer, self.reactor
        now = reactor.monotonic()
        stats = printer.lookup_object('print_stats', None)
        sd = printer.lookup_object('virtual_sdcard', None)
        pause = printer.lookup_object('pause_resume', None)
        if ((stats and stats.get_status(now)['state'] in ('printing', 'paused'))
                or (sd and sd.is_active())
                or (pause and pause.get_status(now)['is_paused'])):
            raise gcmd.error('SG test is only allowed outside printing')
        toolhead = printer.lookup_object('toolhead')
        extruder = toolhead.get_extruder()
        tmc = printer.lookup_object('tmc2209 extruder', None)
        if (tmc is None or extruder.get_name() != 'extruder'
                or extruder.extruder_stepper is None):
            raise gcmd.error('SG test requires one TMC2209 extruder')
        driver = tmc.mcu_tmc
        mcu = driver.get_mcu()
        if mcu.is_fileoutput():
            raise gcmd.error('SG test requires a real MCU')
        stepper = extruder.extruder_stepper.stepper
        toolhead.wait_moves()
        toolhead.flush_step_generation()
        if stepper.get_trapq() is not extruder.get_trapq():
            raise gcmd.error('Extruder is not attached to its motion queue')
        enable = printer.lookup_object('stepper_enable').lookup_enable(
            stepper.get_name())
        if not enable.is_motor_enabled():
            raise gcmd.error('Extruder motor must already be enabled')
        heater = extruder.get_status(reactor.monotonic())
        if heater['target'] or heater['temperature'] > 60.:
            raise gcmd.error('Turn off hotend and cool below 60 C')
        accel = min(400., extruder.max_e_accel)
        if max(speeds) > extruder.max_e_velocity:
            raise gcmd.error('SPEEDS exceeds configured extruder velocity limit')
        if max(speeds) * (.8 + max(speeds)/accel) > extruder.max_e_dist:
            raise gcmd.error('SG move exceeds configured extrude-only distance')
        force = printer.lookup_object('force_move')
        step_dist = stepper.get_step_dist()
        old_pos = stepper.get_commanded_position()
        old_sk = stepper.get_stepper_kinematics()
        fields = driver.get_fields()
        settings = {'en_spreadcycle': 0, 'tpwmthrs': 0,
                    'tcoolthrs': 0xfffff, 'semin': 0}
        registers = {fields.lookup_register(f): fields.registers[
                     fields.lookup_register(f)] for f in settings}
        period = step_dist * 4 * (256 >> fields.get_field('mres'))
        current = tmc.get_status(reactor.monotonic())
        os.makedirs(self.log_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', newline='',
                                         prefix='auto_sg_', suffix='.csv',
                                         dir=self.log_dir, delete=False) as out:
            writer = csv.writer(out)
            writer.writerow(('run', 'repeat', 'speed', 't', 'e', 'sg',
                             'v', 'a', 'v_nom', 'a_nom', 'phase',
                             'cruise_start', 'cruise_end', 'query_start',
                             'query_ms', 'temperature', 'temperature_end',
                             'run_current', 'hold_current', 'electrical_period',
                             'poll_ms', 'e_accel'))
            gcmd.respond_info('SG trace: %s' % out.name)
            try:
                for field, value in settings.items():
                    fields.set_field(field, value)
                for reg in registers:
                    driver.set_register(reg, fields.registers[reg])
                run = 0
                for repeat in range(repeats):
                    order = speeds if repeat % 2 == 0 else speeds[::-1]
                    for speed in order:
                        if printer.is_shutdown():
                            raise gcmd.error('SG test interrupted')
                        heater = extruder.get_status(reactor.monotonic())
                        if (heater['target'] or heater['temperature'] > 60.
                                or not enable.is_motor_enabled()):
                            raise gcmd.error('SG test conditions changed')
                        toolhead.dwell(.15)
                        start = toolhead.get_last_move_time()
                        rows, errors = [], []
                        active = True

                        def sample(eventtime):
                            try:
                                while active and not printer.is_shutdown():
                                    if len(rows) >= 1500:
                                        raise gcmd.error('SG sample limit reached')
                                    query_start = reactor.monotonic()
                                    result = driver.get_register_raw('SG_RESULT')
                                    query_end = reactor.monotonic()
                                    rows.append((mcu.estimated_print_time(
                                        result['#receive_time']),
                                        result['data'] & 0x3ff,
                                        mcu.estimated_print_time(query_start),
                                        (query_end-query_start)*1000.))
                                    reactor.pause(max(eventtime + poll_ms*.001,
                                                      query_end))
                                    eventtime = reactor.monotonic()
                            except Exception as exc:
                                errors.append(exc)

                        capture = reactor.register_callback(sample)
                        try:
                            # Штатный диагностический ход только мотора E.
                            force.manual_move(stepper,
                                              speed * (.8 + speed/accel),
                                              speed, accel)
                            end = toolhead.get_last_move_time()
                            toolhead.dwell(.10)
                            toolhead.wait_moves()
                        except Exception:
                            printer.invoke_shutdown('SG diagnostic move failed')
                            raise
                        finally:
                            active = False
                            capture.wait()
                        if printer.is_shutdown():
                            raise gcmd.error('SG test interrupted')
                        if (stepper.get_stepper_kinematics() is not old_sk
                                or stepper.get_trapq() is not extruder.get_trapq()
                                or abs(stepper.get_commanded_position()
                                       - old_pos) > 1.e-6):
                            printer.invoke_shutdown('SG extruder state changed')
                            raise gcmd.error('SG extruder state changed')
                        if errors:
                            raise errors[0]
                        # Границы разгона берём из истории force_move.
                        try:
                            phases, windows = nominal_phases(
                                force.trapq, start, end, speed, 1)
                        except ValueError as exc:
                            raise gcmd.error(str(exc))
                        cruise = next(w for w in windows.values()
                                      if w[1] == 'fast')
                        _, _, begin, c0, c1, finish = cruise
                        find = stepper.get_past_mcu_position
                        temp_end = extruder.get_status(
                            reactor.monotonic())['temperature']
                        count = 0
                        for t, sg, query_start, query_ms in rows:
                            if not begin <= t <= finish:
                                continue
                            e = find(t) * step_dist
                            v0 = (e-find(t-.02)*step_dist)/.02
                            v1 = (find(t+.02)*step_dist-e)/.02
                            if t < c0:
                                label, vn, an = 'accel', (t-begin)*accel, accel
                            elif t < c1:
                                label, vn, an = 'fast', speed, 0.
                            else:
                                label, vn, an = 'decel', max(
                                    0., speed-(t-c1)*accel), -accel
                            writer.writerow((run, repeat, speed, '%.6f' % (t-begin),
                                             '%.6f' % e, sg, '%.5f' % ((v0+v1)*.5),
                                             '%.5f' % ((v1-v0)/.02), vn, an, label,
                                             c0-begin, c1-begin, query_start-begin,
                                             query_ms, heater['temperature'],
                                             temp_end, current['run_current'],
                                             current['hold_current'], period,
                                             poll_ms, accel))
                            count += 1
                        out.flush()
                        if not count:
                            raise gcmd.error('No StallGuard samples during motion')
                        gcmd.respond_info('SG %.1f mm/s: %d samples' %
                                          (speed, count))
                        run += 1
            finally:
                if not printer.is_shutdown():
                    try:
                        toolhead.wait_moves()
                        toolhead.flush_step_generation()
                        for reg, value in registers.items():
                            fields.registers[reg] = value
                            driver.set_register(reg, value)
                    except Exception:
                        logging.exception('SG test restore failed')
                        printer.invoke_shutdown('SG test restore failed')
                        raise
            gcmd.respond_info('SG trace complete: %s' % out.name)

    def cmd_ZMOD_TEST_PA(self, gcmd):
        # Температуру и безопасную физическую позицию задаёт оператор.
        if gcmd.get_int('CONFIRM', 0) != 1:
            raise gcmd.error('Park the hot nozzle over a safe purge area, '
                             'enable motors, then use CONFIRM=1')
        fast = gcmd.get_float('FAST', 5., minval=2., maxval=10.4)
        strokes = gcmd.get_int('STROKES', 6, minval=2, maxval=6)
        accel = gcmd.get_float('ACCEL', 1000., minval=500., maxval=5000.)
        poll_ms = gcmd.get_int('POLL_MS', 5, minval=2, maxval=20)
        try:
            advances = [float(v) for v in gcmd.get(
                'ADVANCES', '0,0.02,0.04,0.06,0.08').split(',')]
        except ValueError:
            raise gcmd.error('Invalid ADVANCES')
        if (not math.isfinite(fast) or not math.isfinite(accel)
                or not 2 <= len(advances) <= 8
                or any(not math.isfinite(v) or not 0 <= v <= .10
                       for v in advances)):
            raise gcmd.error('Invalid FAST, ACCEL or ADVANCES')

        printer, reactor = self.printer, self.reactor
        now = reactor.monotonic()
        stats = printer.lookup_object('print_stats', None)
        sd = printer.lookup_object('virtual_sdcard', None)
        pause = printer.lookup_object('pause_resume', None)
        if ((stats and stats.get_status(now)['state'] in ('printing', 'paused'))
                or (sd and sd.is_active())
                or (pause and pause.get_status(now)['is_paused'])):
            raise gcmd.error('PA test is only allowed outside printing')
        toolhead = printer.lookup_object('toolhead')
        extruder = toolhead.get_extruder()
        tmc = printer.lookup_object('tmc2209 extruder', None)
        if (tmc is None or extruder.get_name() != 'extruder'
                or len(toolhead.get_position()) != 4
                or extruder.extruder_stepper is None):
            raise gcmd.error('PA test requires one TMC2209 extruder')
        driver = tmc.mcu_tmc
        mcu = driver.get_mcu()
        if mcu.is_fileoutput():
            raise gcmd.error('PA test requires a real MCU')
        estep = extruder.extruder_stepper
        if estep.stepper.get_trapq() is not extruder.get_trapq():
            raise gcmd.error('Extruder is not attached to its motion queue')

        toolhead.wait_moves()
        toolhead.flush_step_generation()
        now = reactor.monotonic()
        status = toolhead.get_status(now)
        heater = extruder.get_status(now)
        pos = toolhead.get_position()
        if (status['homed_axes'] != 'xyz' or pos[2] < 20.
                or not heater['can_extrude'] or heater['temperature'] < 170.
                or abs(heater['temperature'] - heater['target']) > 10.):
            raise gcmd.error('Home XYZ, lower bed to Z >= 20 and stabilize '
                             'hotend at the filament printing temperature')
        steppers = [s for s in toolhead.get_kinematics().get_steppers()
                    if s.is_active_axis('x') or s.is_active_axis('y')]
        enable = printer.lookup_object('stepper_enable')
        if (not steppers or any(s.is_active_axis('z') for s in steppers)
                or any(s.get_trapq() is not toolhead.get_trapq()
                       for s in steppers)
                or any(not enable.lookup_enable(s.get_name()).is_motor_enabled()
                       for s in steppers + [estep.stepper])):
            raise gcmd.error('XY and extruder motors must already be enabled')
        # Виртуальный ход только вдоль X, без изменения физического положения.
        ratio = .4
        segments = [(1., .4)] + [(fast, .5), (1., .4)] * strokes
        travel = sum(v * duration / ratio for v, duration in segments)
        left = pos[0] - status['axis_minimum'][0] - 1.
        right = status['axis_maximum'][0] - pos[0] - 1.
        if max(left, right) < travel:
            raise gcmd.error('Not enough virtual X travel: %.1f mm' % travel)
        direction = 1. if right >= left else -1.
        if fast / ratio > status['max_velocity']:
            raise gcmd.error('FAST exceeds the configured velocity limit')

        if accel > status['max_accel']:
            raise gcmd.error('ACCEL exceeds the current toolhead limit')
        # Сохраняем регистры и не меняем ток двигателя.
        fields = driver.get_fields()
        settings = {'en_spreadcycle': 0, 'tpwmthrs': 0,
                    'tcoolthrs': 0xfffff, 'semin': 0}
        registers = {fields.lookup_register(f): fields.registers[
                     fields.lookup_register(f)] for f in settings}
        current = tmc.get_status(now)
        electrical_period = (estep.stepper.get_step_dist() * 4 *
                             (256 >> fields.get_field('mres')))
        queues = []
        xy_steps = [s.get_mcu_position() for s in steppers]
        motion = printer.lookup_object('motion_queuing')
        gmove = printer.lookup_object('gcode_move')
        old_pa, smooth = heater['pressure_advance'], heater['smooth_time']
        old_accel, old_scv = status['max_accel'], status['square_corner_velocity']
        os.makedirs(self.log_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', newline='',
                                         prefix='auto_pa_', suffix='.csv',
                                         dir=self.log_dir, delete=False) as out:
            writer = csv.writer(out)
            writer.writerow(('run', 'advance', 't', 'e', 'sg', 'v', 'a',
                             'v_nom', 'a_nom', 'stroke', 'phase',
                             'accel_start', 'cruise_start', 'cruise_end',
                             'decel_end', 'query_start', 'query_ms',
                             'temperature', 'temperature_end', 'target',
                             'run_current', 'hold_current', 'fast',
                             'xy_accel', 'poll_ms', 'smooth_time',
                             'electrical_period'))
            gcmd.respond_info('PA trace: %s' % out.name)
            self.gcode.run_script_from_command('SAVE_GCODE_STATE NAME=_ZMOD_PA')
            try:
                for field, value in settings.items():
                    fields.set_field(field, value)
                for reg in registers:
                    driver.set_register(reg, fields.registers[reg])
                toolhead.set_max_velocities(None, accel, old_scv, None)
                for s in steppers:
                    queues.append((s, s.get_trapq()))
                    s.set_trapq(None)
                motion.check_step_generation_scan_windows()

                for run, advance in enumerate(advances):
                    if printer.is_shutdown():
                        raise gcmd.error('PA test interrupted')
                    run_heater = extruder.get_status(reactor.monotonic())
                    if (not run_heater['can_extrude'] or abs(
                            run_heater['temperature'] - heater['target']) > 10.
                            or run_heater['target'] != heater['target']):
                        raise gcmd.error('Hotend temperature changed')
                    estep._set_pressure_advance(advance, smooth)
                    # Обеспечиваем начало записи до первого виртуального хода.
                    toolhead.dwell(.15)
                    start = toolhead.get_last_move_time()
                    rows = []
                    active = True
                    errors = []

                    def sample(eventtime):
                        try:
                            while active and not printer.is_shutdown():
                                if len(rows) >= 5000:
                                    raise gcmd.error('PA sample limit reached')
                                query_start = reactor.monotonic()
                                result = driver.get_register_raw('SG_RESULT')
                                query_end = reactor.monotonic()
                                rows.append((mcu.estimated_print_time(
                                    result['#receive_time']),
                                    result['data'] & 0x3ff,
                                    mcu.estimated_print_time(query_start),
                                    (query_end - query_start) * 1000.))
                                reactor.pause(max(eventtime + poll_ms * .001,
                                                  query_end))
                                eventtime = reactor.monotonic()
                        except Exception as exc:
                            errors.append(exc)

                    capture = reactor.register_callback(sample)
                    try:
                        p = toolhead.get_position()
                        for speed, duration in segments:
                            if errors or printer.is_shutdown():
                                break
                            p[0] += direction * speed * duration / ratio
                            p[3] += speed * duration
                            toolhead.move(p, speed / ratio)
                        end = toolhead.get_last_move_time()
                        toolhead.dwell(.10)
                        toolhead.wait_moves()
                    finally:
                        active = False
                        capture.wait()
                    if printer.is_shutdown():
                        raise gcmd.error('PA test interrupted')
                    if errors:
                        raise errors[0]
                    find = extruder.find_past_position
                    # Считываем историю после завершения движения, вне опроса UART.
                    try:
                        phases, windows = nominal_phases(extruder.get_trapq(),
                                                         start, end, fast, strokes)
                    except ValueError as exc:
                        raise gcmd.error(str(exc))
                    valid = [row for row in rows if start <= row[0] <= end]
                    run_end_temp = extruder.get_status(
                        reactor.monotonic())['temperature']
                    phase_idx = 0
                    for t, sg, query_start, query_ms in valid:
                        e = find(t)
                        v0 = (e - find(t - .02)) / .02
                        v1 = (find(t + .02) - e) / .02
                        while (phase_idx + 1 < len(phases)
                               and t >= phases[phase_idx][1]):
                            phase_idx += 1
                        a, b, v_nom, a_nom = phases[phase_idx]
                        if a <= t < b:
                            v_nom += a_nom * (t - a)
                            stroke, label, *bounds = windows.get(
                                phase_idx, (0, 'slow'))
                        else:
                            v_nom, a_nom, stroke, label, bounds = (
                                0., 0., 0, 'idle', [])
                        times = ['%.6f' % (x - start) for x in bounds]
                        times += [''] * (4 - len(times))
                        writer.writerow((run, advance, '%.6f' % (t-start),
                                         '%.6f' % e, sg,
                                         '%.5f' % ((v0+v1)*.5),
                                         '%.5f' % ((v1-v0)/.02),
                                         '%.5f' % v_nom, '%.5f' % a_nom,
                                         stroke, label, *times,
                                         '%.6f' % (query_start-start),
                                         '%.3f' % query_ms,
                                         run_heater['temperature'], run_end_temp,
                                         heater['target'], current['run_current'],
                                         current['hold_current'], fast, accel,
                                         poll_ms, smooth, electrical_period))
                    out.flush()
                    if not valid:
                        raise gcmd.error('No StallGuard samples during motion')
                    hz = ((len(valid)-1) / (valid[-1][0]-valid[0][0])
                          if len(valid) > 1 and valid[-1][0] > valid[0][0]
                          else 0.)
                    gcmd.respond_info('PA %.3f: %d samples, %.1f Hz' %
                                      (advance, len(valid), hz))
                    # Сброс виртуальной X только после завершения движений.
                    toolhead.flush_step_generation()
                    if [s.get_mcu_position() for s in steppers] != xy_steps:
                        raise gcmd.error('XY steps changed during PA test')
                    p = toolhead.get_position()
                    p[:2] = pos[:2]
                    toolhead.set_position(p)
            finally:
                if not printer.is_shutdown():
                    try:
                        # Сначала дожидаемся уже поставленных в очередь шагов.
                        toolhead.wait_moves()
                        toolhead.flush_step_generation()
                        if queues:
                            if ([s.get_mcu_position() for s in steppers]
                                    != xy_steps
                                    or any(not enable.lookup_enable(
                                        s.get_name()).is_motor_enabled()
                                           for s in steppers)):
                                raise gcmd.error('XY position is uncertain')
                            p = toolhead.get_position()
                            p[:2] = pos[:2]
                            toolhead.set_position(p)
                            for s, queue in queues:
                                s.set_trapq(queue)
                            motion.check_step_generation_scan_windows()
                        estep._set_pressure_advance(old_pa, smooth)
                        toolhead.set_max_velocities(None, old_accel,
                                                    old_scv, None)
                        for reg, value in registers.items():
                            fields.registers[reg] = value
                            driver.set_register(reg, value)
                        gmove.reset_last_position()
                        self.gcode.run_script_from_command(
                            'RESTORE_GCODE_STATE NAME=_ZMOD_PA MOVE=0')
                    except Exception:
                        logging.exception('PA test restore failed')
                        printer.invoke_shutdown('PA test restore failed')
                        raise
            gcmd.respond_info('PA trace complete: %s' % out.name)


def load_config(config):
    return ZModPA(config)
