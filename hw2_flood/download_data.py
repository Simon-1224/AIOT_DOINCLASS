from pathlib import Path
import urllib.request,json,zipfile,io,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parent
REF='kamrulhasan0077/flood-prediction-dataset'
HEAD={'User-Agent':'kaggle-api-v1.7.4'}
def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers=HEAD),timeout=60).read()
meta=json.loads(fetch('https://www.kaggle.com/api/v1/datasets/view/'+REF))
(ROOT/'data/kaggle_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
archive=fetch('https://www.kaggle.com/api/v1/datasets/download/'+REF+'?datasetVersionNumber=1')
z=zipfile.ZipFile(io.BytesIO(archive))
print('Official Kaggle archive:',z.namelist())
files=[n for n in z.namelist() if n.lower().endswith('.csv')]
if len(files)!=1: raise RuntimeError('Inspect archive before choosing CSV')
data=z.read(files[0])
(ROOT/'data/flood.csv').write_bytes(data)
df=pd.read_csv(io.BytesIO(data))
assert df.shape==(50000,21),df.shape
assert 'FloodProbability' in df and len(df.drop(columns='FloodProbability').columns)==20
source={'title':meta['title'],'ref':REF,'version':1,'url':'https://www.kaggle.com/datasets/'+REF,'license':meta['licenseName'],'download_url':'https://www.kaggle.com/api/v1/datasets/download/'+REF+'?datasetVersionNumber=1','retrieved':'2026-10-16','sha256':hashlib.sha256(data).hexdigest(),'rows':len(df),'columns':df.columns.tolist(),'provenance':'Uploader states external source, collection methods and geographic coverage are unknown. Synthetic-like teaching data; synthetic status cannot be independently verified.'}
(ROOT/'data/source.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
print(json.dumps(source,indent=2))
print(df.dtypes.to_string());print(df.head(2).to_string(index=False));print('Target range:',df.FloodProbability.min(),df.FloodProbability.max())
