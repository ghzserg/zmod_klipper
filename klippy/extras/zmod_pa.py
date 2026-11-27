# Диагностический сбор StallGuard для подбора Pressure Advance.
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import csv
import logging
import math
import os
import tempfile


class ZModPA:
    def __init__(self, config):
        self.printer = config.get_printer()
        self.reactor = self.printer.get_reactor()
        self.gcode = self.printer.lookup_object('gcode')
        self.log_dir = config.get('log_dir', '/opt/config/mod_data/log')
        self.gcode.register_command('ZMOD_TEST_PA', self.cmd_ZMOD_TEST_PA,
                                    desc='Record extruder StallGuard trace')

    def cmd_ZMOD_TEST_PA(self, gcmd):
        # Температуру и безопасную физическую позицию задаёт оператор.
        if gcmd.get_int('CONFIRM', 0) != 1:
            raise gcmd.error('Park the hot nozzle over a safe purge area, '
                             'enable motors, then use CONFIRM=1')
        fast = gcmd.get_float('FAST', 5., minval=2., maxval=10.4)
        strokes = gcmd.get_int('STROKES', 6, minval=2, maxval=6)
        try:
            advances = [float(v) for v in gcmd.get(
                'ADVANCES', '0,0.02,0.04,0.06,0.08').split(',')]
        except ValueError:
            raise gcmd.error('Invalid ADVANCES')
        if (not math.isfinite(fast) or not 2 <= len(advances) <= 8
                or any(not math.isfinite(v) or not 0 <= v <= .10
                       for v in advances)):
            raise gcmd.error('Invalid FAST or ADVANCES')

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

        # Сохраняем регистры и не меняем ток двигателя.
        fields = driver.get_fields()
        settings = {'en_spreadcycle': 0, 'tpwmthrs': 0,
                    'tcoolthrs': 0xfffff, 'semin': 0}
        registers = {fields.lookup_register(f): fields.registers[
                     fields.lookup_register(f)] for f in settings}
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
            writer.writerow(('run', 'advance', 't', 'e', 'sg', 'v', 'a'))
            gcmd.respond_info('PA trace: %s' % out.name)
            self.gcode.run_script_from_command('SAVE_GCODE_STATE NAME=_ZMOD_PA')
            try:
                for field, value in settings.items():
                    fields.set_field(field, value)
                for reg in registers:
                    driver.set_register(reg, fields.registers[reg])
                toolhead.set_max_velocities(None, min(1000., old_accel),
                                            old_scv, None)
                for s in steppers:
                    queues.append((s, s.get_trapq()))
                    s.set_trapq(None)
                motion.check_step_generation_scan_windows()

                for run, advance in enumerate(advances):
                    if printer.is_shutdown():
                        raise gcmd.error('PA test interrupted')
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
                                if len(rows) >= 3000:
                                    raise gcmd.error('PA sample limit reached')
                                result = driver.get_register_raw('SG_RESULT')
                                rows.append((mcu.estimated_print_time(
                                    result['#receive_time']),
                                    result['data'] & 0x3ff))
                                reactor.pause(max(eventtime + .005,
                                                  reactor.monotonic()))
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
                        toolhead.wait_moves()
                    finally:
                        active = False
                        capture.wait()
                    if printer.is_shutdown():
                        raise gcmd.error('PA test interrupted')
                    if errors:
                        raise errors[0]
                    end = toolhead.get_last_move_time()
                    find = extruder.find_past_position
                    valid = [(t, sg) for t, sg in rows
                             if start <= t <= end]
                    for t, sg in valid:
                        e = find(t)
                        v0 = (e - find(t - .02)) / .02
                        v1 = (find(t + .02) - e) / .02
                        writer.writerow((run, advance, '%.6f' % (t-start),
                                         '%.6f' % e, sg,
                                         '%.5f' % ((v0+v1)*.5),
                                         '%.5f' % ((v1-v0)/.02)))
                    out.flush()
                    if not valid:
                        raise gcmd.error('No StallGuard samples during motion')
                    hz = ((len(valid)-1) / (valid[-1][0]-valid[0][0])
                          if len(valid) > 1 and valid[-1][0] > valid[0][0]
                          else 0.)
                    gcmd.respond_info('PA %.3f: %d samples, %.1f Hz' %
                                      (advance, len(valid), hz))
                    # Сброс виртуальной X только после завершения движений.
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
