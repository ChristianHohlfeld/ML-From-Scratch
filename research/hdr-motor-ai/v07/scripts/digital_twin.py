"""Offline digital twin only. Never label synthetic outputs hardware evidence."""
import json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gear_ai.two_link import TwoLink, PhysicalTwin


def run():
    m=TwoLink();s=PhysicalTwin(m)
    trace=[]
    for a,b in [(0,0),(20,0),(0,20),(20,20),(-20,-20),(0,0)]:
        pos=s.settle((math.radians(a),math.radians(b)),seconds=.5)
        exact=m.tip(math.radians(a),math.radians(b))
        trace.append({'a_deg':a,'b_deg':b,'tip_xy_mm':(pos*1000).round(4).tolist(),
                      'target_ideal_mm':(exact*1000).round(4).tolist(),
                      'settling_error_mm':round(float(np.linalg.norm(pos-exact)*1000),4)})
    return {'source':'SYNTHETIC DIGITAL TWIN; NO HARDWARE MEASUREMENTS','model':'2R horizontal planar linkage',
            'backreaction_mass_matrix_first_pose':m.mass_matrix(0).tolist(),
            'mixed_difference_mm':(m.rectangular_interaction(-.4,.4,-.4,.4)*1000).tolist(),
            'trace':trace,'limitations':'Finite-torque simplified simulation without backlash, gravity, contacts, servo electronics or bearing friction'}

if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'results'/'digital_twin.json';path.write_text(json.dumps(run(),indent=2)+'\n');print(path)
