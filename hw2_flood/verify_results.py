from pathlib import Path
import json,ast,hashlib
import pandas as pd,numpy as np,joblib
from sklearn.metrics import mean_squared_error
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
j=lambda n:json.loads((ROOT/'outputs'/n).read_text(encoding='utf-8-sig'))
s=j('split_indices.json');a,b,c=map(set,[s['development'],s['calibration'],s['test']]);assert not(a&b or a&c or b&c);assert len(a|b|c)==50000
rank=pd.read_csv(ROOT/'outputs/cv_ranking.csv');folds=pd.read_csv(ROOT/'outputs/cv_folds.csv');assert len(rank)==34 and len(folds)==170;assert rank.CV_MSE.is_monotonic_increasing
assert j('selection_decision.json')['winner']=='OLS full'
assert not j('selection_decision.json')['test_used_for_selection']
pddata=pd.read_csv(ROOT/'data/flood.csv');bundle=joblib.load(ROOT/'outputs/prediction_bundle.joblib');Xt=pddata.loc[s['test'],bundle['features']];yt=pddata.loc[s['test'],'FloodProbability'];pr=bundle['model'].predict(Xt)
pred=pd.read_csv(ROOT/'outputs/test_predictions.csv');np.testing.assert_allclose(pred.predicted,pr,rtol=0,atol=1e-15)
interval=j('prediction_interval.json');np.testing.assert_allclose(np.mean((yt>=pr-bundle['radius'])&(yt<=pr+bundle['radius'])),interval['test_coverage'],atol=1e-12)
web=j('dashboard_data.json');mask={m['scheme']:m for m in web['models']};m=mask['OLS full'];wp=Xt.to_numpy()@np.array(m['coefficients'])+m['intercept'];np.testing.assert_allclose(wp,pr,rtol=0,atol=1e-14)
features=j('selected_features.json')
for method in ['SelectKBest','RFE','Forward SFS','Lasso selection','Permutation']:
    for k in [14,16,18,20]:assert len(features[f'{method} k={k}'])==k
reader=PdfReader(ROOT/'output/pdf/5115056023_hw2_report.pdf');assert len(reader.pages)==15
text='\n'.join(p.extract_text() for p in reader.pages)
for word in ['NotebookLM','5115056023','Business Understanding','Data Preparation','Modeling','Evaluation','Deployment','GPT','95%']:assert word in text,word
assert not j('submission_status.json')['ready_to_submit']
checks={'split_disjoint':True,'cv_schemes':34,'cv_fold_records':170,'feature_counts_checked':20,'web_vs_python_prediction':True,'interval_coverage_recomputed':True,'pdf_pages':len(reader.pages),'evidence_status_honest':True}
(ROOT/'outputs/verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks,indent=2))
