"""Hardware-in-loop optical measurement. Requires REAL USB camera and ESP32.

Never substitutes commanded positions or kinematic predictions for vision.
Records raw commanded angles, camera-observed V0, electrical telemetry,
uncertainty and inferred residuals to a CSV; can be run before hardware with
--self-test which uses explicitly synthetic frames only.
"""
from __future__ import annotations
import argparse,csv,math,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gear_ai.two_link import TwoLink
from gear_ai.vision import detect_homography,detect_tip
from gear_ai.protocol import parse_telemetry,integrate_joules


def sequence():
    # Includes true two-port interaction, boundary cases and loop replay.
    points=[(0,0),(20,0),(0,20),(20,20),(-20,-20),(35,-15),(-35,15),
            (45,35),(30,-45),(-45,-30),(0,0)]
    for a,b in points:yield a,b


def run(port:str,camera:int,out:Path,hold_s=1.0,frames_per_pose=8):
    try:import cv2,serial
    except ImportError as exc:raise RuntimeError('pip install -e ".[vision]"') from exc
    if hold_s<0.2 or frames_per_pose<3:raise ValueError('Hold >=0.2 s, at least 3 frames required')
    cap=cv2.VideoCapture(camera)
    if not cap.isOpened():raise RuntimeError('Camera cannot open; no fabricated result will be emitted')
    ser=None
    all_telemetry=[]
    try:
        ser=serial.Serial(port,115200,timeout=0.1)
        time.sleep(2.0)  # MCU auto reset
        ser.reset_input_buffer()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        header=['pose','a_cmd_deg','b_cmd_deg','timestamp','x_observed_mm','y_observed_mm',
                'x_ideal_mm','y_ideal_mm','error_mm','max_board_error_mm','volts','amps','watts','source']
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('w',newline='',encoding='utf8') as fp:
            writer=csv.DictWriter(fp,fieldnames=header);writer.writeheader()
            geom=TwoLink()
            for pose,(a,b) in enumerate(sequence()):
                ser.write(f'SET {a:.2f} {b:.2f}\n'.encode())
                deadline=time.monotonic()+hold_s
                last=[]
                while time.monotonic()<deadline:
                    ok,img=cap.read()
                    if ok and time.monotonic()>deadline-hold_s*.65:
                        last.append((time.monotonic(),img.copy()))
                        if len(last)>frames_per_pose:last.pop(0)
                if len(last)<frames_per_pose:
                    raise RuntimeError(f'Only {len(last)} camera frames at pose {pose}')
                tele=None
                while ser.in_waiting:
                    msg=ser.readline().decode(errors='replace').strip()
                    try:t=parse_telemetry(msg)
                    except ValueError:continue
                    if t is not None:
                        tele=t
                        all_telemetry.append(t)
                p0=geom.tip(math.radians(a),math.radians(b))*1000
                for now,frame in last:
                    try:
                        H,e=detect_homography(frame)
                        xy=detect_tip(frame,H)
                    except ValueError as exc:
                        raise RuntimeError(f'Camera invalid at pose {pose}: {exc}') from exc
                    writer.writerow(dict(pose=pose,a_cmd_deg=a,b_cmd_deg=b,
                        timestamp=f'{now:.6f}',x_observed_mm=f'{xy[0]:.3f}',
                        y_observed_mm=f'{xy[1]:.3f}',x_ideal_mm=f'{p0[0]:.3f}',
                        y_ideal_mm=f'{p0[1]:.3f}',error_mm=f'{np.linalg.norm(xy-p0):.3f}',
                        max_board_error_mm=f'{e:.3f}',
                        volts=tele.volts if tele else float('nan'),
                        amps=tele.amps if tele else float('nan'),
                        watts=tele.watts if tele else float('nan'),source='CAMERA'))
                fp.flush()
        ser.write(b'RELEASE\n')
        power_path=out.with_name(out.stem+'_power_trace.csv')
        with power_path.open('w',newline='') as power_file:
            w=csv.writer(power_file);w.writerow(['ms','cmd_a_deg','cmd_b_deg','volts','amps','watts','source'])
            for t in all_telemetry:
                w.writerow([t.millisecond,t.target_a_deg,t.target_b_deg,t.volts,t.amps,t.watts,'INA219' if np.isfinite(t.watts) else 'NO_SENSOR'])
        if all_telemetry:
            try:
                energy=integrate_joules(all_telemetry)
            except ValueError:
                energy=None
            print('Servo bus electrical energy during logged windows:',energy,'J (None means unmeasured)')
    finally:
        if ser is not None:
            try:ser.write(b'RELEASE\n')
            except Exception:pass
            ser.close()
        cap.release()
    return out


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--port',help='USB serial, e.g. /dev/ttyUSB0 or COM3')
    pa.add_argument('--camera',type=int,default=0)
    pa.add_argument('--out',type=Path,default=Path('results/hardware_measurement.csv'))
    pa.add_argument('--hold',type=float,default=1.)
    a=pa.parse_args()
    if not a.port:pa.error('--port is required; synthetic data cannot be called a hardware result')
    try:print(run(a.port,a.camera,a.out,a.hold))
    except Exception as exc:print(f'MEASUREMENT FAILED: {exc}',file=sys.stderr);sys.exit(1)
if __name__=='__main__':main()
