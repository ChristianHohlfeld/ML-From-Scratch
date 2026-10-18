import json
from pathlib import Path
from benchmark import run
seeds=[20261018,20261019,20261020,20261021,20261022]
all_results=[]
for seed in seeds:
 r=run(n_per_class=36,seed=seed)
 all_results.append({'seed':seed,'metrics':r['metrics']})
names=list(all_results[0]['metrics'])
summary={'n_seeds':len(seeds),'n_test_per_seed':72,'replicates':all_results,
    'mean_accuracy':{k:sum(r['metrics'][k] for r in all_results)/len(all_results) for k in names},
    'min_accuracy':{k:min(r['metrics'][k] for r in all_results) for k in names},
    'max_accuracy':{k:max(r['metrics'][k] for r in all_results) for k in names}}
Path('results/replicates.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
