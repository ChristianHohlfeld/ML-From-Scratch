"""Strict host parser for microcontroller telemetry. A command log is not a position sensor."""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Telemetry:
    millisecond: int
    target_a_deg: float
    target_b_deg: float
    volts: float
    amps: float
    watts: float


def parse_telemetry(line:str)->Telemetry|None:
    parts=line.strip().split()
    if not parts or parts[0]!='T':return None
    if len(parts)!=7:raise ValueError(f'Malformed T packet: {line!r}')
    ts=int(parts[1]);a,b,v,i,w=map(float,parts[2:])
    if ts<0 or not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError('Invalid timestamp or targets')
    if not all(math.isnan(z) for z in (v,i,w)):
        if not all(math.isfinite(z) for z in (v,i,w)):
            raise ValueError('Incomplete electrical measurement')
        if not 0<=v<15 or not -5<i<5 or abs(w-v*i)>max(0.05,abs(w)*0.04):
            raise ValueError('Implausible electrical telemetry')
    return Telemetry(ts,a,b,v,i,w)


def integrate_joules(packets):
    """Trapezoidal integration of real current-sensor samples, milliseconds."""
    rows=list(packets)
    if len(rows)<2 or any(not math.isfinite(t.watts) for t in rows):
        return None
    if any(b.millisecond<=a.millisecond for a,b in zip(rows,rows[1:])):
        raise ValueError('Non-increasing telemetry timestamps')
    return sum((a.watts+b.watts)*0.5*(b.millisecond-a.millisecond)/1000 for a,b in zip(rows,rows[1:]))
