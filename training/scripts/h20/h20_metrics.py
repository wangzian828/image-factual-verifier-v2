import json,csv,re,os
from pathlib import Path
root=Path(os.environ['IFV_H20_RUN_ROOT'])
for run in root.iterdir():
 if not run.is_dir() or not (run/'gpu.csv').exists():continue
 peaks={};utils={}
 for row in csv.reader((run/'gpu.csv').open()):
  if len(row)<6:continue
  i=int(row[1]);peaks[i]=max(peaks.get(i,0),float(row[2]));utils[i]=max(utils.get(i,0),float(row[4]))
 print(run.name,'peak_MiB',peaks,'max_util_pct',utils)
 if (run/'result.json').exists():print((run/'result.json').read_text())
 for p in run.rglob('logging.jsonl'):
  for line in p.read_text().splitlines()[-8:]:print(line)
