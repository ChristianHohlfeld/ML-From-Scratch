"""Held-out ablation of the *already trained* mechanical motor model.

No ablation retraining. Removing a hub/group coupler is done by reducing its
physical spring stiffness rather than solving V0 again.
"""
import argparse
import json
from pathlib import Path
import torch
from hdr_motor_ai.mechanics import GearNetwork
from hdr_motor_ai.tasks import make_dataset


def score(m, x, y, steps):
    with torch.no_grad():
        out,_=m(x,steps=steps)
        return round(float(((out>=0).float()==y).float().mean()),6)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('results/group_contrast_run'))
    p.add_argument('--output',type=Path,default=Path('results/ablation.json'))
    args=p.parse_args()
    state=torch.load(args.run/'motor_best.pt',weights_only=True,map_location='cpu')
    m=GearNetwork(channels=state['channels'])
    m.load_state_dict(state['model_state'])
    task=state['task'];seed=state['seed'];steps=state['steps']
    x,y=make_dataset(512,seed+539,task)
    m.eval()
    outcomes={'task':task,'seed':seed,'steps':steps,'full':score(m,x,y,steps)}
    with torch.no_grad():
        logk=m.log_k.clone();lr=m.ratio_logits.clone();q=m.log_cubic.clone()
        for label,kinds in [('without_hub',['hub']),('without_group',['group']),('without_pairs',['pair']),('without_neighbors',['neighbor'])]:
            for i,kind in enumerate(m.edge_kinds):
                if kind in kinds: m.log_k[i].fill_(-8);m.log_cubic[i].fill_(-10)
            outcomes[label]=score(m,x,y,steps)
            m.log_k.copy_(logk);m.log_cubic.copy_(q)
        m.ratio_logits.zero_()
        outcomes['fixed_ratios']=score(m,x,y,steps)
        m.ratio_logits.copy_(lr)
        outcomes['stiffness_mean']=round(float(m.log_k.exp().mean()),4)
        outcomes['stiffness_range']=[round(float(m.log_k.exp().min()),4),round(float(m.log_k.exp().max()),4)]
        outcomes['ratio_deviation_abs_mean']=round(float(m.ratio_logits.tanh().abs().mean()),4)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(outcomes,indent=2),encoding='utf-8')
    print(json.dumps(outcomes,indent=2))
if __name__ == '__main__':main()
