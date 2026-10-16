"""Predict using the exact model/calibration pair assessed in the report."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--input',default='example_input.json');args=parser.parse_args()
path=Path(args.input);path=path if path.is_absolute() else ROOT/path
row=json.loads(path.read_text(encoding='utf-8-sig'))
bundle=joblib.load(ROOT/'outputs/prediction_bundle.joblib');names=bundle['features']
if set(row)!=set(names):raise SystemExit('Input must contain exactly the documented 20 feature names')
values=np.asarray([row[n] for n in names],dtype=float)
if not np.isfinite(values).all() or (values<0).any():raise SystemExit('Use finite nonnegative feature scores')
p=float(bundle['model'].predict(pd.DataFrame([row],columns=names))[0]);q=bundle['radius']
print(json.dumps({'prediction':p,'lower_95':p-q,'upper_95':p+q,'clipped':False,'scope':'Teaching scores; not a real-world flood warning'},indent=2))
