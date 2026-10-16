"""Build a ZIP with an explicit evidence status. Never label missing evidence complete."""
from pathlib import Path
import json,zipfile,hashlib
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
status=json.loads((ROOT/'outputs/submission_status.json').read_text(encoding='utf-8-sig'))
report=ROOT/'output/pdf/5115056023_hw2_report.pdf'
assert report.is_file() and len(PdfReader(str(report)).pages)>=10
required=['5115056023_hw2.py','hw2_pipeline.py','requirements.txt','README.md','data/source.json','data/flood.csv','outputs/test_metrics.csv','outputs/prediction_interval.json','outputs/prediction_bundle.joblib','web/index.html']
for name in required:assert (ROOT/name).is_file(),name
package_status='READY_FOR_STUDENT_REVIEW' if status['ready_to_submit'] else 'INCOMPLETE_EVIDENCE_DO_NOT_SUBMIT_AS_COMPLETE'
readme=f'Student: 5115056023\nStatus: {package_status}\nPending: '+', '.join(status['pending'])+'\nSee README.md and evidence instructions.\n'
(ROOT/'SUBMISSION_STATUS.txt').write_text(readme,encoding='utf-8')
archive=ROOT.parent/'5115056023_hw2.zip'
excluded={'.venv','__pycache__','.git','tmp'}
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for file in sorted(ROOT.rglob('*')):
        rel=file.relative_to(ROOT)
        if not file.is_file() or any(part in excluded for part in rel.parts) or file.suffix=='.pyc':continue
        if rel.as_posix().startswith('output/pdf/'):
            if file!=report:continue
            dest=file.name
        else:dest=rel.as_posix()
        z.write(file,arcname='5115056023_hw2/'+dest)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert '5115056023_hw2/5115056023_hw2_report.pdf' in z.namelist()
print('ZIP:',archive)
print('Status:',package_status)
print('Bytes:',archive.stat().st_size)
print('SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())
