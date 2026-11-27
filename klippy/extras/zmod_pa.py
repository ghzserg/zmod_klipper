# Диагностический сбор StallGuard для подбора Pressure Advance.
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import csv
import json
import logging
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
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
        self.sg_token = 0
        root = os.path.join(os.path.dirname(
            self.printer.get_start_args()['config_file']), 'mod_data', 'auto_pa')
        self.baseline_dir = config.get('baseline_dir', root)
        # Старый параметр диагностики оставлен для тестовых конфигураций.
        self.log_dir = config.get('log_dir', self.baseline_dir)
        self.model = config.get('model', 'AD5M')
        if self.model not in ('AD5M', 'AD5M Pro'):
            raise config.error('Auto PA is experimental on AD5M / AD5M Pro only')
        self.gcode.register_command('ZMOD_AUTO_PA', self.cmd_ZMOD_AUTO_PA,
                                    desc='Experimental automatic PA calibration')
        self.gcode.register_command('ZMOD_AUTO_PA_EMPTY',
                                    self.cmd_ZMOD_AUTO_PA_EMPTY,
                                    desc='Calibrate the empty extruder')
        self.gcode.register_command('ZMOD_TEST_PA', self.cmd_ZMOD_TEST_PA,
                                    desc='Record extruder StallGuard trace')

        self.gcode.register_command('ZMOD_TEST_SG', self.cmd_ZMOD_TEST_SG,
                                    desc='Record unloaded extruder StallGuard')

    def new_run_dir(self, kind, root=None):
        # Как VFA: отдельный каталог каждого запуска, без перезаписи.
        root = self.baseline_dir if root is None else root
        os.makedirs(root, exist_ok=True)
        name = time.strftime('%Y%m%d_%H%M%S') + '_' + kind
        index = 0
        while True:
            folder = os.path.join(root, name if not index else
                                  '%s_%d' % (name, index))
            if not os.path.exists(folder + '.zip'):
                try:
                    os.mkdir(folder)
                    return folder
                except FileExistsError:
                    pass
            index += 1

    @contextmanager
    def run_archive(self, gcmd, kind, root=None):
        folder = self.new_run_dir(kind, root)
        archive = folder + '.zip'
        gcmd.respond_info('Auto PA data: %s' % archive)
        try:
            yield folder
        finally:
            failed = sys.exc_info()[0] is not None
            try:
                if not os.listdir(folder):
                    os.rmdir(folder)
                else:
                    # Даже при ошибке сохраняем уже записанные данные.
                    with tempfile.NamedTemporaryFile(dir=os.path.dirname(folder),
                                                     suffix='.tmp', delete=False) as tmp:
                        pending = tmp.name
                    try:
                        with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED) as z:
                            for name in sorted(os.listdir(folder)):
                                z.write(os.path.join(folder, name), name)
                        os.replace(pending, archive)
                    finally:
                        if os.path.exists(pending):
                            os.unlink(pending)
                    shutil.rmtree(folder)
                    gcmd.respond_info('Auto PA archive: %s' % archive)
            except Exception:
                # Если архивация не удалась, исходные файлы не удаляем.
                logging.exception('Auto PA archive failed: %s', folder)
                if not failed:
                    raise gcmd.error('Auto PA archive failed; data: %s' % folder)

    @contextmanager
    def suspend_filament_sensors(self):
        # Не изменяем конфигурацию и восстанавливаем даже при исключении.
        saved = []
        try:
            for kind in ('filament_switch_sensor', 'filament_motion_sensor'):
                for _, sensor in self.printer.lookup_objects(kind):
                    helper = sensor.runout_helper
                    saved.append((helper, helper.sensor_enabled))
                    if helper.sensor_enabled:
                        helper.sensor_enabled = False
            yield saved
        finally:
            for helper, enabled in saved:
                helper.sensor_enabled = enabled

    def run_fit(self, gcmd, args):
        # NumPy загружается только в дочернем процессе, не в reactor Klipper.
        script = os.path.join(os.path.dirname(__file__), 'zmod_pa_fit.py')
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1')
        with tempfile.TemporaryFile() as output:
            proc = subprocess.Popen([sys.executable, script] + args,
                                    stdout=output, stderr=output, env=env)
            try:
                deadline = self.reactor.monotonic() + 180.
                while proc.poll() is None:
                    now = self.reactor.monotonic()
                    if self.printer.is_shutdown() or now >= deadline:
                        raise gcmd.error('Auto PA calculation interrupted')
                    self.reactor.pause(now + .1)
                output.seek(0)
                result = output.read(65536).decode('utf-8', errors='replace')
                if proc.returncode:
                    logging.error('Auto PA calculation failed: %s', result)
                    raise gcmd.error('Auto PA calculation failed; see klippy.log')
                return json.loads(result)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()

    def cmd_ZMOD_AUTO_PA(self, gcmd):
        self.auto_pa(gcmd, False)

    def cmd_ZMOD_AUTO_PA_EMPTY(self, gcmd):
        self.auto_pa(gcmd, True)

    def auto_pa(self, gcmd, empty):
        printer, reactor = self.printer, self.reactor
        now = reactor.monotonic()
        stats = printer.lookup_object('print_stats', None)
        sd = printer.lookup_object('virtual_sdcard', None)
        pause = printer.lookup_object('pause_resume', None)
        if ((stats and stats.get_status(now)['state'] in ('printing', 'paused'))
                or (sd and sd.is_active())
                or (pause and pause.get_status(now)['is_paused'])):
            raise gcmd.error('Auto PA is only allowed outside printing')
        if gcmd.get_int('T', 0) != 0:
            raise gcmd.error('Auto PA supports only T=0 on AD5M')
        baseline = os.path.join(self.baseline_dir, 'empty.json')
        if not empty and not os.path.isfile(baseline):
            raise gcmd.error('Run TEST_PA_EMPTY first')
        pa_min = gcmd.get_float('PA_MIN', .01, minval=.01, maxval=.10)
        pa_max = gcmd.get_float('PA_MAX', .10, minval=.02, maxval=.10)
        count = 2 if empty else gcmd.get_int('PA_COUNT', 3, minval=1, maxval=10)
        if not math.isfinite(pa_min+pa_max) or pa_max < pa_min:
            raise gcmd.error('Invalid PA_MIN or PA_MAX')
        toolhead = printer.lookup_object('toolhead')
        extruder = toolhead.get_extruder()
        tmc = printer.lookup_object('tmc2209 extruder', None)
        if (tmc is None or extruder.get_name() != 'extruder'
                or extruder.extruder_stepper is None):
            raise gcmd.error('Auto PA requires a single TMC2209 extruder')
        driver = tmc.mcu_tmc
        uart = driver.mcu_uart
        mcu = driver.get_mcu()
        if ((uart.requested_baud or 40000) != 40000
                or uart.analog_mux is not None or len(uart.instances) != 1
                or extruder.extruder_stepper.stepper.get_mcu() is not mcu
                or mcu.is_fileoutput()
                or any(mcu.try_lookup_command(fmt) is None for fmt in (
                    'tmcuart_buffer_start oid=%c token=%hu addr=%c count=%hu'
                    ' clock=%u interval_us=%u',
                    'tmcuart_buffer_status oid=%c token=%hu',
                    'tmcuart_buffer_read oid=%c token=%hu first=%hu'))):
            raise gcmd.error('Auto PA requires updated MCU and stock 40000 baud')
        sensor = printer.lookup_object('filament_switch_sensor e0_sensor', None)
        if sensor is not None:
            present = sensor.get_status(now)['filament_detected']
            if bool(present) == empty:
                raise gcmd.error('Remove all filament for TEST_PA_EMPTY'
                                 if empty else 'Load filament before TEST_PA')
        client = printer.lookup_object('gcode_macro _CLIENT_VARIABLE', None)
        if client is None:
            raise gcmd.error('Auto PA requires configured Z-Mod parking position')
        x, y = (float(client.variables[k]) for k in
                ('custom_park_x', 'custom_park_y'))
        status = toolhead.get_status(now)
        low, high = status['axis_minimum'], status['axis_maximum']
        if not (low[0] <= x <= high[0] and low[1] <= y <= high[1]
                and high[2] >= 50. and status['max_accel'] >= 1000.
                and max(x-low[0], high[0]-x) >= 68.):
            raise gcmd.error('Invalid parking position or motion limits for Auto PA')
        heater = extruder.get_heater()
        heaters = printer.lookup_object('heaters')
        old = extruder.get_status(now)
        old_target = old['target']
        temp = gcmd.get_float('TEMP', 210. if empty else old_target,
                              minval=170.)
        if (not math.isfinite(temp) or not max(170., heater.min_extrude_temp)
                <= temp <= heater.max_temp):
            raise gcmd.error('Set a valid filament printing temperature')
        os.makedirs(self.baseline_dir, exist_ok=True)
        saved = None
        self.run_fit(gcmd, ['--check'])
        if not empty:
            try:
                checked = self.run_fit(gcmd, [baseline])
            except printer.command_error:
                raise gcmd.error('Repeat TEST_PA_EMPTY: invalid baseline; see klippy.log')
            if not checked['valid']:
                raise gcmd.error('Repeat TEST_PA_EMPTY: inconsistent baseline')
            saved = checked['settings']

        estep = extruder.extruder_stepper
        enable = printer.lookup_object('stepper_enable')
        ename = estep.stepper.get_name()
        was_enabled = enable.lookup_enable(ename).is_motor_enabled()
        fields = driver.get_fields()
        current = tmc.current_helper
        old_hold = current.req_hold_current
        old_regs = {k: fields.registers[k] for k in ('CHOPCONF', 'IHOLD_IRUN')}
        with self.run_archive(gcmd, 'empty' if empty else 'pa') as folder:
            if not empty:
                # Сохраняем именно ту базу, с которой рассчитан этот результат.
                snapshot = os.path.join(folder, 'empty.json')
                shutil.copyfile(baseline, snapshot)
                baseline = snapshot
            paths = []
            orders = ('0,0.02,0.04,0.06,0.08', '0.08,0.06,0.04,0.02,0',
                      '0.04,0,0.08,0.02,0.06')
            gmove = printer.lookup_object('gcode_move')
            with self.suspend_filament_sensors() as sensors:
                self.gcode.run_script_from_command('SAVE_GCODE_STATE NAME=_ZMOD_AUTO_PA')
                try:
                    gcmd.respond_info('Experimental Auto PA: clear the bed and '
                                      'keep the purge area unobstructed')
                    if toolhead.get_status(reactor.monotonic())['homed_axes'] != 'xyz':
                        self.gcode.run_script_from_command('G28')
                    if toolhead.get_status(reactor.monotonic())['homed_axes'] != 'xyz':
                        raise gcmd.error('Homing did not complete')
                    # Макрос homing может изменить состояние датчиков.
                    for helper, _ in sensors:
                        helper.sensor_enabled = False
                    # Физическая парковка без пользовательского G-code offset.
                    z = max(50., toolhead.get_position()[2])
                    toolhead.manual_move([None, None, z], 10.)
                    toolhead.manual_move([x, y, None], 50.)
                    toolhead.wait_moves()
                    motors = [s.get_name() for s in toolhead.get_kinematics().get_steppers()
                              if s.is_active_axis('x') or s.is_active_axis('y')]
                    enable.set_motors_enable(motors + [ename], True)
                    heaters.set_temperature(heater, temp, wait=True)
                    toolhead.wait_moves()
                    current.set_current(.6, .6, toolhead.get_last_move_time())
                    estep._set_pressure_advance(old['pressure_advance'], .04)
                    toolhead.wait_moves()
                    fingerprint = {
                        'model': self.model,
                        'algorithm': 'sg250-v1',
                        'step_dist': estep.stepper.get_step_dist(),
                        'registers': {k: fields.registers.get(k) for k in (
                            'GCONF', 'CHOPCONF', 'PWMCONF', 'COOLCONF',
                            'TPOWERDOWN', 'TPWMTHRS', 'TCOOLTHRS', 'IHOLD_IRUN')}}
                    if saved is not None and fingerprint != saved:
                        raise gcmd.error('Driver settings changed; repeat TEST_PA_EMPTY')
                    for i in range(count):
                        params = {'CONFIRM': '1', 'MCU': '1', 'FAST': '8',
                                  'ACCEL': '1000', 'POLL_MS': '4',
                                  'ADVANCES': orders[i % len(orders)]}
                        cmd = self.gcode.create_gcode_command('ZMOD_TEST_PA',
                                                             'ZMOD_TEST_PA', params)
                        paths.append(self.test_pa(cmd, folder))
                finally:
                    if not printer.is_shutdown():
                        try:
                            toolhead.wait_moves()
                            estep._set_pressure_advance(old['pressure_advance'],
                                                        old['smooth_time'])
                            current.req_hold_current = old_hold
                            for reg, value in old_regs.items():
                                fields.registers[reg] = value
                                driver.set_register(reg, value,
                                                    toolhead.get_last_move_time())
                            heaters.set_temperature(heater, old_target)
                            if not was_enabled:
                                enable.set_motors_enable([ename], False)
                            gmove.reset_last_position()
                            self.gcode.run_script_from_command(
                                'RESTORE_GCODE_STATE NAME=_ZMOD_AUTO_PA MOVE=0')
                        except Exception:
                            logging.exception('Auto PA settings restore failed')
                            printer.invoke_shutdown('Auto PA settings restore failed')
                            raise
            if printer.is_shutdown():
                raise gcmd.error('Auto PA interrupted')
            if empty:
                name = os.path.join(folder, 'empty.json')
                settings = os.path.join(folder, 'settings.json')
                with open(settings, 'w') as f:
                    json.dump(fingerprint, f)
                result = self.run_fit(gcmd, ['--prepare', name, settings] + paths)
                with open(os.path.join(folder, 'result.json'), 'w') as f:
                    json.dump(result, f, indent=2)
                if not result['valid']:
                    raise gcmd.error('Empty extruder traces inconsistent; '
                                     'previous baseline retained')
                # Архив предыдущих калибровок остаётся в датированных каталогах.
                with tempfile.NamedTemporaryFile(dir=self.baseline_dir, suffix='.json',
                                                 delete=False) as tmp:
                    pending = tmp.name
                try:
                    shutil.copyfile(name, pending)
                    os.replace(pending, baseline)
                finally:
                    if os.path.exists(pending):
                        os.unlink(pending)
                gcmd.respond_info('Empty extruder calibration saved: %s' % baseline)
                return
            result = self.run_fit(gcmd, [baseline] + paths)
            result['valid'] = (result['valid'] and
                               pa_min <= result['pa'] <= pa_max)
            zmod_version = None
            try:
                with open('/tmp/version.txt', encoding='utf-8') as f:
                    for line in f.read(8192).splitlines():
                        key, _, value = line.partition('=')
                        if key.strip() == 'ZMOD_VERSION':
                            value = value.strip().strip('"').strip("'")
                            if (0 < len(value) <= 80 and all(
                                    c.isalnum() or c in '.-+_' for c in value)):
                                zmod_version = value
                            break
            except OSError:
                pass
            result.update(model=self.model, algorithm='sg250-v1',
                          zmod_version=zmod_version,
                          klipper_version=printer.get_start_args().get('software_version'),
                          pa_min=pa_min, pa_max=pa_max, settings=fingerprint)
            # Отчёт содержит только измерения калибровки, без общего конфига и логов.
            with open(os.path.join(folder, 'result.json'), 'w') as f:
                json.dump(result, f, indent=2)
            gcmd.respond_info('Experimental Auto PA: raw %.3f, corrected %.4f, '
                              'gain %.2f%%, %s; NOT applied\nReport: %s' %
                              (result['raw_pa'], result['pa'], result['gain'],
                               'candidate' if result['valid'] else 'unreliable/out of range',
                               folder + '.zip'))

    @contextmanager
    def capture_sg(self, gcmd, driver, stepper, start, duration, poll_ms,
                   mcu_mode):
        printer, reactor = self.printer, self.reactor
        mcu, uart = driver.get_mcu(), driver.mcu_uart
        rows, counts = [], [0, 0]
        if not mcu_mode:
            active, errors = True, []

            def sample(eventtime):
                try:
                    while active and not printer.is_shutdown():
                        if len(rows) >= 5000:
                            raise gcmd.error('SG sample limit reached')
                        query_start = reactor.monotonic()
                        result = driver.get_register_raw('SG_RESULT')
                        query_end = reactor.monotonic()
                        rows.append((mcu.estimated_print_time(
                            result['#receive_time']), result['data'] & 0x3ff,
                            mcu.estimated_print_time(query_start),
                            (query_end-query_start)*1000.))
                        reactor.pause(max(eventtime + poll_ms*.001, query_end))
                        eventtime = reactor.monotonic()
                except Exception as exc:
                    errors.append(exc)

            capture = reactor.register_callback(sample)
            try:
                yield rows, counts
            finally:
                active = False
                capture.wait()
            if errors:
                raise errors[0]
            return

        start_fmt = ('tmcuart_buffer_start oid=%c token=%hu addr=%c count=%hu'
                     ' clock=%u interval_us=%u')
        status_fmt = 'tmcuart_buffer_status oid=%c token=%hu'
        read_fmt = 'tmcuart_buffer_read oid=%c token=%hu first=%hu'
        if (uart.analog_mux is not None or len(uart.instances) != 1
                or stepper.get_mcu() is not mcu
                or any(mcu.try_lookup_command(fmt) is None
                       for fmt in (start_fmt, status_fmt, read_fmt))):
            raise gcmd.error('SG MCU mode requires updated buffer firmware '
                             'and dedicated TMC UART')
        baud = uart.requested_baud or 40000
        if poll_ms < (2 if baud >= 80000 else 3 if baud >= 60000 else 4):
            raise gcmd.error('POLL_MS too short for configured UART baud')
        count = math.ceil((duration + .10) * 1000. / poll_ms)
        if not 1 <= count <= 6000 or count * poll_ms > 12000:
            raise gcmd.error('SG MCU capture exceeds sample limit')
        start_cmd = mcu.lookup_query_command(
            start_fmt, 'tmcuart_buffer_started oid=%c token=%hu status=%c'
            ' available=%u', oid=uart.oid, cq=uart.cmd_queue, is_async=True)
        status_cmd = mcu.lookup_query_command(
            status_fmt, 'tmcuart_buffer_state oid=%c token=%hu status=%c'
            ' total=%hu errors=%hu query_min=%u query_max=%u',
            oid=uart.oid, cq=uart.cmd_queue, is_async=True)
        read_cmd = mcu.lookup_query_command(
            read_fmt, 'tmcuart_buffer_data oid=%c token=%hu first=%hu status=%c'
            ' data=%*s', oid=uart.oid, cq=uart.cmd_queue, is_async=True)
        self.sg_token = (self.sg_token + 1) & 0xffff
        token = self.sg_token
        with driver.mutex:
            uncertain = True
            try:
                clock = mcu.print_time_to_clock(start - .10)
                ack = start_cmd.send([uart.oid, token, driver.addr, count,
                                      clock & 0xffffffff, poll_ms*1000])
                if ack['token'] != token:
                    raise gcmd.error('SG MCU token mismatch')
                if ack['status']:
                    # При отказе по памяти или параметрам захват не запущен.
                    uncertain = ack['status'] not in (2, 3)
                    raise gcmd.error('SG MCU start rejected: %s, available %d bytes'
                                     % (ack['status'], ack['available']))
                # До окончания движения данные остаются в RAM головы.
                yield rows, counts
                deadline = reactor.monotonic() + 3.
                while not printer.is_shutdown():
                    status = status_cmd.send([uart.oid, token])
                    if status['token'] != token or status['status'] > 1:
                        raise gcmd.error('Invalid SG MCU buffer status')
                    if not status['status']:
                        uncertain = False
                        break
                    if reactor.monotonic() >= deadline:
                        raise gcmd.error('SG MCU buffer timeout')
                    reactor.pause(reactor.monotonic() + .05)
                if printer.is_shutdown():
                    raise gcmd.error('SG MCU capture interrupted')
                if status['total'] != count:
                    raise gcmd.error('Incomplete SG MCU capture')
                # История шагов хранится 30 с. Не допускаем слишком долгой выгрузки.
                deadline = reactor.monotonic() + 15.
                for first in range(0, count, 7):
                    if printer.is_shutdown() or reactor.monotonic() >= deadline:
                        raise gcmd.error('SG MCU buffer download interrupted')
                    reply = read_cmd.send([uart.oid, token, first])
                    raw = reply['data']
                    if (reply['token'] != token or reply['first'] != first
                            or reply['status']
                            or len(raw) != 6 * min(7, count-first)):
                        raise gcmd.error('Invalid SG MCU buffer packet')
                    for rx, sg in struct.iter_unpack('<IH', raw):
                        counts[0] += 1
                        if sg == 0xffff:
                            counts[1] += 1
                            continue
                        if sg > 1023:
                            raise gcmd.error('Invalid SG MCU value')
                        # Время ответа MCU; захват короче половины периода clock32.
                        rx64 = clock + (((rx-clock+0x80000000)
                                         & 0xffffffff)-0x80000000)
                        t = mcu.clock_to_print_time(rx64)
                        if rows and t <= rows[-1][0]:
                            raise gcmd.error('Non-monotonic SG MCU time')
                        # Индивидуальные TX/end не хранятся в шестибайтной записи.
                        rows.append((t, sg, math.nan, math.nan))
                if (reactor.monotonic() >= deadline or counts[0] != count
                        or counts[1] != status['errors']):
                    raise gcmd.error('Incomplete SG MCU buffer download')
                base = mcu.clock_to_print_time(clock)
                qmin = 1000. * (mcu.clock_to_print_time(
                    clock + status['query_min']) - base)
                qmax = 1000. * (mcu.clock_to_print_time(
                    clock + status['query_max']) - base)
                gcmd.respond_info('SG MCU: %d bytes available before buffer, '
                                  'query %.3f..%.3f ms, %d errors' %
                                  (ack['available'], qmin, qmax, counts[1]))
            except Exception:
                # Не восстанавливаем регистры поверх неизвестного захвата.
                if uncertain and not printer.is_shutdown():
                    printer.invoke_shutdown('SG MCU buffer capture failed')
                raise

    def cmd_ZMOD_TEST_SG(self, gcmd):
        with self.run_archive(gcmd, 'sg', self.log_dir) as folder:
            self.test_sg(gcmd, folder)

    def test_sg(self, gcmd, folder):
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
        mcu_mode = gcmd.get_int('MCU', 0, minval=0, maxval=1)
        poll_ms = gcmd.get_int('POLL_MS', 4 if mcu_mode else 2,
                              minval=2, maxval=5 if mcu_mode else 20)
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
        uart = driver.mcu_uart
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
        with tempfile.NamedTemporaryFile(mode='w', newline='',
                                         prefix='auto_sg_', suffix='.csv',
                                         dir=folder, delete=False) as out:
            writer = csv.writer(out)
            writer.writerow(('run', 'repeat', 'speed', 't', 'e', 'sg',
                             'v', 'a', 'v_nom', 'a_nom', 'phase',
                             'cruise_start', 'cruise_end', 'query_start',
                             'query_ms', 'temperature', 'temperature_end',
                             'run_current', 'hold_current', 'electrical_period',
                             'poll_ms', 'e_accel', 'source', 'uart_baud',
                             'burst_total', 'burst_errors'))
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
                        toolhead.dwell(.35 if mcu_mode else .15)
                        start = toolhead.get_last_move_time()
                        duration = .8 + 2*speed/accel + .45
                        with self.capture_sg(gcmd, driver, stepper, start,
                                             duration, poll_ms, mcu_mode) as (
                                                 rows, burst_counts):
                            try:
                                force.manual_move(stepper,
                                                  speed * (.8 + speed/accel),
                                                  speed, accel)
                                end = toolhead.get_last_move_time()
                                toolhead.dwell(.10)
                                toolhead.wait_moves()
                            except Exception:
                                printer.invoke_shutdown('SG diagnostic move failed')
                                raise
                        if printer.is_shutdown():
                            raise gcmd.error('SG test interrupted')
                        if (stepper.get_stepper_kinematics() is not old_sk
                                or stepper.get_trapq() is not extruder.get_trapq()
                                or abs(stepper.get_commanded_position()
                                       - old_pos) > 1.e-6):
                            printer.invoke_shutdown('SG extruder state changed')
                            raise gcmd.error('SG extruder state changed')
                        # Границы разгона берём из истории force_move.
                        try:
                            phases, windows = nominal_phases(
                                force.trapq, start, end, speed, 1)
                        except ValueError as exc:
                            raise gcmd.error(str(exc))
                        cruise = next(w for w in windows.values()
                                      if w[1] == 'fast')
                        _, _, begin, c0, c1, finish = cruise
                        if mcu_mode and (not rows or rows[0][0] > begin
                                         or rows[-1][0] < finish):
                            raise gcmd.error('SG buffer does not cover motion')
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
                                             poll_ms, accel,
                                             'mcu' if mcu_mode else 'host',
                                             uart.requested_baud or 40000,
                                             *burst_counts))
                            count += 1
                        out.flush()
                        if not count:
                            raise gcmd.error('No StallGuard samples during motion')
                        gcmd.respond_info('SG %.1f mm/s: %d samples, '
                                          'MCU reads %d, invalid %d' %
                                          (speed, count, *burst_counts))
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
        with self.run_archive(gcmd, 'test_pa', self.log_dir) as folder:
            self.test_pa(gcmd, folder)

    def test_pa(self, gcmd, folder):
        # Температуру и безопасную физическую позицию задаёт оператор.
        if gcmd.get_int('CONFIRM', 0) != 1:
            raise gcmd.error('Park the hot nozzle over a safe purge area, '
                             'enable motors, then use CONFIRM=1')
        fast = gcmd.get_float('FAST', 5., minval=2., maxval=10.4)
        strokes = gcmd.get_int('STROKES', 6, minval=2, maxval=6)
        accel = gcmd.get_float('ACCEL', 1000., minval=500., maxval=5000.)
        mcu_mode = gcmd.get_int('MCU', 0, minval=0, maxval=1)
        poll_ms = gcmd.get_int('POLL_MS', 2 if mcu_mode else 5,
                              minval=2, maxval=5 if mcu_mode else 20)
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
        with tempfile.NamedTemporaryFile(mode='w', newline='',
                                         prefix='auto_pa_', suffix='.csv',
                                         dir=folder, delete=False) as out:
            writer = csv.writer(out)
            writer.writerow(('run', 'advance', 't', 'e', 'sg', 'v', 'a',
                             'v_nom', 'a_nom', 'stroke', 'phase',
                             'accel_start', 'cruise_start', 'cruise_end',
                             'decel_end', 'query_start', 'query_ms',
                             'temperature', 'temperature_end', 'target',
                             'run_current', 'hold_current', 'fast',
                             'xy_accel', 'poll_ms', 'smooth_time',
                             'electrical_period', 'source', 'uart_baud',
                             'burst_total', 'burst_errors'))
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
                    # Захват начинается за 100 мс до первого движения.
                    toolhead.dwell(.35 if mcu_mode else .15)
                    start = toolhead.get_last_move_time()
                    duration = sum(d for _, d in segments) + 1.5
                    with self.capture_sg(gcmd, driver, estep.stepper, start,
                                         duration, poll_ms, mcu_mode) as (
                                             rows, burst_counts):
                        p = toolhead.get_position()
                        for speed, duration in segments:
                            if printer.is_shutdown():
                                break
                            p[0] += direction * speed * duration / ratio
                            p[3] += speed * duration
                            toolhead.move(p, speed / ratio)
                        end = toolhead.get_last_move_time()
                        toolhead.dwell(.10)
                        toolhead.wait_moves()
                    if printer.is_shutdown():
                        raise gcmd.error('PA test interrupted')
                    if mcu_mode and (not rows or rows[0][0] > start
                                     or rows[-1][0] < end):
                        raise gcmd.error('SG buffer does not cover PA motion')
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
                                         poll_ms, smooth, electrical_period,
                                         'mcu' if mcu_mode else 'host',
                                         driver.mcu_uart.requested_baud or 40000,
                                         *burst_counts))
                    out.flush()
                    if not valid:
                        raise gcmd.error('No StallGuard samples during motion')
                    hz = ((len(valid)-1) / (valid[-1][0]-valid[0][0])
                          if len(valid) > 1 and valid[-1][0] > valid[0][0]
                          else 0.)
                    gcmd.respond_info('PA %.3f: %d samples, %.1f Hz, '
                                      'MCU reads %d, invalid %d' %
                                      (advance, len(valid), hz, *burst_counts))
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
            return out.name


def load_config(config):
    return ZModPA(config)
