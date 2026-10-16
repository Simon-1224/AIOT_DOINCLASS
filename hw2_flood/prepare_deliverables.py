from pathlib import Path
import json,shutil,importlib.metadata as md
import nbformat
p=Path(__file__).resolve().parent
d=json.loads((p/'outputs/dashboard_data.json').read_text(encoding='utf-8'))
(p/'example_input.json').write_text(json.dumps(dict(zip(d['features'],d['defaults'])),indent=2),encoding='utf-8')
(p/'web/index.html').write_text((p/'web/template.html').read_text(encoding='utf-8-sig').replace('__DATA__',json.dumps(d,ensure_ascii=False,allow_nan=False)),encoding='utf-8')
shutil.copytree(p/'outputs/figures',p/'web/figures',dirs_exist_ok=True)
pkgs=['numpy','pandas','scipy','scikit-learn','matplotlib','joblib','threadpoolctl','reportlab','pypdf','pillow','nbformat','nbclient','ipykernel']
(p/'requirements.txt').write_text('\n'.join(f'{x}=={md.version(x)}' for x in pkgs)+'\n',encoding='utf-8')
nb=nbformat.v4.new_notebook()
nb.cells=[nbformat.v4.new_markdown_cell('# 5115056023 HW2：洪水機率多元線性回歸\n\nCRISP-DM。資料來源與限制見 README.md；完整 GPT 對話仍待本人匯出。此 Notebook 呼叫與主程式相同的實驗入口，避免維護兩份不同邏輯。'),
nbformat.v4.new_markdown_cell('## Business / Data Understanding\nKaggle 50,000 筆，20 個輸入；目標實際為浮點數。原始外部來源不明，近似固定線性生成公式，不代表真實洪水資料。'),
nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd, json\nROOT=Path.cwd()\nassert (ROOT/'5115056023_hw2.py').is_file(), '請在作業資料夾執行'\ndf=pd.read_csv(ROOT/'data/flood.csv')\nprint(df.shape)\ndf.head()"),
nbformat.v4.new_markdown_cell('## Preparation / Modeling\n先切分 60/20/20；五折 CV 與選特徵完全限於開發資料；校準集只建區間。執行下格會重跑全部實驗。'),
nbformat.v4.new_code_cell("import runpy\nrunpy.run_path(str(ROOT/'5115056023_hw2.py'),run_name='__main__')"),
nbformat.v4.new_markdown_cell('## Evaluation\n依 CV 排名，不能以 Test 分數重新選模型。'),
nbformat.v4.new_code_cell("pd.read_csv(ROOT/'outputs/test_metrics.csv').head(20)"),
nbformat.v4.new_code_cell("from IPython.display import display, Image\ndisplay(Image(filename=str(ROOT/'outputs/figures/05_prediction_intervals.png')))\ndisplay(Image(filename=str(ROOT/'outputs/figures/09_reduced_prediction_intervals.png')))"),
nbformat.v4.new_markdown_cell('## Deployment / AI 證據\nweb/index.html 可離線開啟。用 predict_sample.py 載入同一模型與校準區間。報告製作執行 build_report.py；對話 PDF 匯出後再合併。'),
nbformat.v4.new_code_cell("print((ROOT/'evidence/notebooklm_summary.txt').read_text(encoding='utf-8-sig'))")]
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nbformat.write(nb,p/'5115056023_hw2.ipynb')
print('Requirements, example input, UI, and notebook created')
