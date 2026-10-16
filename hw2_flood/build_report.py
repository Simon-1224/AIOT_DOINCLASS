# -*- coding: utf-8 -*-
"""Generate an honest Chinese PDF report; append only supplied, student-confirmed transcripts."""
from pathlib import Path
import csv,json,html,os
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT,TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from pypdf import PdfReader,PdfWriter
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';FIG=OUT/'figures';EVID=ROOT/'evidence'
PDF=ROOT/'output/pdf';PDF.mkdir(parents=True,exist_ok=True)
def load(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def readcsv(name):return list(csv.DictReader((OUT/name).open(encoding='utf-8-sig',newline='')))
font=os.environ.get('HW2_CJK_FONT',r'C:\Windows\Fonts\msjh.ttc')
if Path(font).is_file():pdfmetrics.registerFont(TTFont('CJK',font,subfontIndex=0))
else:pdfmetrics.registerFont(UnicodeCIDFont('MSung-Light'));font='MSung-Light'
FONT='CJK' if Path(font).is_file() else 'MSung-Light'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyZH',fontName=FONT,fontSize=10,leading=17,spaceAfter=8,wordWrap='CJK',textColor=colors.HexColor('#213a44')))
styles.add(ParagraphStyle(name='TitleZH',parent=styles['BodyZH'],fontSize=26,leading=36,spaceAfter=15))
styles.add(ParagraphStyle(name='H1ZH',parent=styles['BodyZH'],fontSize=18,leading=26,spaceAfter=14,textColor=colors.HexColor('#087f8c')))
styles.add(ParagraphStyle(name='H2ZH',parent=styles['BodyZH'],fontSize=12,leading=20,spaceBefore=10,spaceAfter=8))
styles.add(ParagraphStyle(name='SmallZH',parent=styles['BodyZH'],fontSize=8,leading=12,spaceAfter=5))
styles.add(ParagraphStyle(name='CellZH',parent=styles['BodyZH'],fontSize=8,leading=12,spaceAfter=0))
story=[]
def p(text,style='BodyZH'):story.append(Paragraph(html.escape(str(text)).replace('\n','<br/>'),styles[style]))
def h(text):p(text,'H1ZH')
def sub(text):p(text,'H2ZH')
def new():story.append(PageBreak())
def num(x):
    x=float(x)
    return f'{x:.3e}' if x and abs(x)<1e-5 else f'{x:.6f}'
def table(headers,rows,widths=None):
    data=[[Paragraph(html.escape(str(x)),styles['CellZH']) for x in row] for row in [headers]+rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7f3f3')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#164f5b')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#9bbdc0')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f9f9')])]))
    story.append(t);story.append(Spacer(1,9))
def image(name,width=505,maxheight=275,caption=''):
    from PIL import Image as PILImage
    with PILImage.open(FIG/name) as im:w,he=im.size
    ratio=min(width/w,maxheight/he)
    story.append(Image(str(FIG/name),width=w*ratio,height=he*ratio))
    if caption:p(caption,'SmallZH')
    story.append(Spacer(1,7))
manifest=load('run_manifest.json');quality=load('data_quality.json');diag=load('diagnostics.json');rd=load('reduced_diagnostics.json');relation=load('target_relationship.json');interval=load('prediction_interval.json');reduced=load('reduced_prediction_interval.json')
ranking=readcsv('cv_ranking.csv');tests=readcsv('test_metrics.csv');by={r['scheme']:r for r in tests};winner=manifest['winner']
summary_file=EVID/'notebooklm_summary.txt';raw_file=EVID/'notebooklm_raw.txt'
notebook_ok=summary_file.exists() and raw_file.exists() and len(summary_file.read_text(encoding='utf-8-sig'))>=100
conv_files=[];conv_ok=False
cm=EVID/'conversation_manifest.json'
if cm.exists():
    c=json.loads(cm.read_text(encoding='utf-8-sig'))
    for name in c.get('files',[]):
        path=(EVID/name).resolve()
        if path.parent!=EVID.resolve() or path.suffix.lower()!='.pdf' or not path.is_file():raise ValueError('Invalid conversation evidence path')
        if len(PdfReader(str(path)).pages)<1:raise ValueError('Empty conversation PDF')
        conv_files.append(path)
    conv_ok=bool(c.get('complete') and c.get('confirmed_by_student') and conv_files)
status={'model_analysis_complete':True,'notebooklm_student_provided':notebook_ok,'notebooklm_url_verified':False,'conversation_pdf_complete':conv_ok,'ready_to_submit':bool(notebook_ok and conv_ok),'pending':[] if conv_ok and notebook_ok else (['完整原始 GPT 對話 PDF 與本人完整性確認'] if not conv_ok else [])+(['NotebookLM 真實摘要'] if not notebook_ok else [])}
(OUT/'submission_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
# Cover
p('HW2 / CRISP-DM','H2ZH');p('洪水機率預測\n多元線性回歸與特徵選擇','TitleZH')
p('學號：5115056023　｜　實驗日期：2026-10-16')
p('資料：Kaggle Flood Prediction Dataset，version 1，50,000 筆、20 個輸入特徵。')
p('文件狀態：'+('研究與證據齊備；繳交前請本人核對。' if status['ready_to_submit'] else '分析報告已產出；完整 GPT 對話 PDF 尚待補入，這不是完整繳交版。'),'H2ZH')
sub('主要發現')
p('開發資料顯示目標幾乎是 20 個特徵的等權線性和，因此完整 OLS 的測試 R² 為 1。這是資料生成結構的結果，不能解讀為對真實洪災有完美預測能力。')
table(['資料切分','筆數','用途'],[['開發集','30,000','五折交叉驗證、特徵選擇、模型訓練'],['校準集','10,000','僅建立 split-conformal 預測區間'],['最終測試集','10,000','方案鎖定後評估，不用來挑模型']],[100,70,335])
p(f'正式模型：{winner}。測試 MSE = {num(by[winner]["MSE"])}；RMSE = {num(by[winner]["RMSE"])}；R² = {num(by[winner]["R2"])}。')
p(f'完整模型區間涵蓋率 {interval["test_coverage"]:.2%}，平均寬度 {interval["mean_width"]:.3e}（浮點誤差尺度）。14 特徵示範涵蓋率 {reduced["test_coverage"]:.2%}，平均寬度 {reduced["mean_width"]:.6f}。')
sub('閱讀路線');p('商業與資料理解 → 資料準備 → 建模與特徵選擇 → 評估與預測區間 → 假設診斷 → 研究比較 → AI 協作與證據 → 部署與參考來源。')
new();h('1. Business Understanding｜商業理解')
p('任務是用 20 項環境與社會因素分數，預測連續目標 FloodProbability。此為監督式迴歸，使用 scikit-learn 線性模型比較特徵保留數與正則化的影響。')
p('教學驗收以正確可重現的流程、相對平均值基準的誤差、區間涵蓋率，以及對資料限制的說明為主。沒有實際防災成本資料，也沒有指定可上線的誤差門檻。')
sub('1.1 資料來源與授權')
p('實際採用 Kaggle 上 Kamrul Hasan 重新上傳的公開版本：[1]，而非需要邀請的 Predict the Flood 競賽，也不是 Playground S4E5 的百萬筆競賽版本。原先常見的 brijlaldhankour 原始連結目前無法取得。')
p(manifest['source']['url'],'SmallZH');p('Kaggle 官方 API 標示 CC0: Public Domain，version 1。下載後保存原始 metadata、來源 JSON 和 SHA-256。這項授權標示不代表已驗證資料真實來源。')
p('上傳者承認外部原始來源、蒐集方式與地理覆蓋範圍不明。本地觀察到近乎確定的生成公式，只能說具有生成式教學資料的特徵；不能聲稱已獨立查證它是真實量測或確定某種合成方式。')
sub('1.2 能與不能解釋的事情')
p('可以展示線性擬合、正則化、選特徵、資料洩漏防範及預測區間。不能將資料中分數視為具真實物理單位，也不能將迴歸係數當作防災措施的因果效果。')
p('未提交 Kaggle，沒有競賽名次。外部競賽解法的分數只作研究脈絡，不當成同一資料切分下的公平實驗對照。')
new();h('2. Data Understanding｜資料理解')
p(f'實際 CSV 有 {quality["rows"]:,} 筆、21 欄。其中 20 個輸入欄位為 int64，FloodProbability 為 float64；來源描述「全部 int64」與實際檔案不符，分析以 CSV 為準。')
p(f'缺值總數：{sum(quality["missing"].values())}；重複列：{quality["duplicate_rows"]}；目標範圍：{quality["target_range"]}。無 id、時間或地理識別欄位。')
dictionary=readcsv('data_dictionary.csv')
table(['輸入特徵（不含目標）','中文理解','開發集範圍'],[[r['feature'],r['zh'],f'{float(r["dev_min"]):g} ~ {float(r["dev_max"]):g}'] for r in dictionary],[230,145,130])
p('欄位中文僅協助理解名稱。對照真實量測的編碼規則及「高分是好或壞」並未得到可信證實，因此不按字面強加係數方向。完整統計量另見 outputs/development_description.csv。','SmallZH')
new();h('2.1 探索圖與生成機制核對')
image('01_correlation.png',maxheight=335,caption='圖 1：僅使用開發資料的相關矩陣。各輸入彼此相關性低，不代表資料具有真實水文意義。')
image('02_eda.png',maxheight=200,caption='圖 2：開發集目標分布及特徵與目標的相關性。')
p(f'額外以開發集的「特徵總和」擬合一維直線，得到斜率 {relation["sum_slope"]:.12f}、截距 {relation["intercept"]:.3e}，最大絕對誤差 {relation["max_abs_error"]:.3e}。因此近似關係為 FloodProbability = 0.005 × sum(20 features)。')
p('此核對不使用校準或測試目標，不將目標放入輸入欄位。它揭露資料設計限制，並解釋為何少留任何特徵都可能失去必要資訊。')
new();h('3. Data Preparation｜資料準備')
table(['步驟','具體設定'],[['固定切分','先以 random_state=42 保留 20% 測試，再將其餘資料的 25% 作校準；形成 60/20/20。'],['前處理','SimpleImputer(median) → StandardScaler → selector → regressor。每個 CV 訓練折自行擬合，驗證折只 transform。'],['五折驗證','KFold(n_splits=5, shuffle=True, random_state=42)，所有方案使用同樣折。'],['選擇器抽樣','每個 CV 訓練折內固定種子最多抽 2,500 筆學習特徵子集，降低 SFS 與 permutation 的成本；最後回歸用完整訓練折。'],['保留資料','校準標籤不參與排名；測試標籤不參與排名或調參。所有切分索引保存於 split_indices.json。']],[100,405])
p('選擇器學習所用子樣本，是實驗的計算預算設計。完整資料仍參與每折回歸和驗證。不能把選擇器抽樣造成的排名當成穩定、唯一的重要性結論。')
p('SFS 在每個訓練折內的子樣本做三折前向搜尋；已計算的相同訓練資料順序可快取重用，不跨驗證折分享選特徵結果。Permutation 在子樣本內另切 70/30，僅用其中驗證部分計算負 MSE 重要性。')
p('Lasso 特徵排序使用子樣本內三折 LassoCV 的係數絕對值；最後保留指定 k 個後以 OLS 重擬合。指定 k 的排序法不等同單純只保留非零係數。前處理的參數仍完全限制在外層訓練折內。')
sub('可重現與誠實比較')
p('來源版本、檔案 SHA-256、套件版本、抽樣上限、種子、原始 CV 各折分數與選定特徵列表均保存。未因最終測試表現修改超參數。多方案搜尋的 CV 最小值可能樂觀，保留測試作最終評估。')
new();h('4. Modeling｜建模與特徵選擇')
table(['方案','實際設定'],[['平均值基準','DummyRegressor，預測開發資料平均值。'],['完整 OLS','LinearRegression，20 特徵，截距項。'],['Ridge','alpha = 0.01, 0.1, 1, 10, 100, 1000；使用標準化輸入。'],['Lasso','alpha = 1e-6, 1e-5, 1e-4, 1e-3, 0.01，max_iter=20000。'],['SelectKBest','f_regression 單變量 F 分數排序。'],['RFE','標準化後以 LinearRegression 逐步消除低係數特徵。'],['Forward SFS','逐一加入讓內部三折 MSE 最小的特徵。'],['Lasso selection','LassoCV 九個 alpha，以絕對係數選指定 k。'],['Permutation','訓練內部驗證，五次置換的負 MSE 下降量排序。'],['非線性對照','HistGradientBoostingRegressor，固定參數，同樣 CV；不參與主要線性模型選擇。']],[110,395])
p('五種特徵選擇各比較 k = 14、16、18、20，合計 20 個設定。k=20 時不刪特徵，因此五種名稱其實都回到完整 OLS。這些重複設定用於方法對照，不能解讀成五個不同的最佳模型。')
p(f'全部比較 {manifest["candidate_schemes"]} 個設定（包含基準及非線性對照）。以線性方案的開發集五折平均 MSE 最小者為正式模型；完全同分時採用名稱清楚的 OLS full。正式模型為 {winner}。')
image('06_feature_selection.png',maxheight=220,caption='圖 3：特徵數與交叉驗證 RMSE。刪特徵後誤差增加符合等權生成機制。')
new();h('4.1 前 20 名開發集 CV 設定')
p('排名依 CV MSE；同分名稱排序僅方便列示。k=20 的六個 OLS 設定實際等價。此表不是以 Test MSE 排名。','SmallZH')
table(['設定','k','CV MSE','CV RMSE'],[[r['scheme'],r['k'],num(r['CV_MSE']),num(r['CV_RMSE'])] for r in ranking[:20]],[245,35,110,115])
p('完整各折分數見 outputs/cv_folds.csv；所有 34 個設定見 cv_ranking.csv。特徵選擇的各方案最終欄位列表見 selected_features.json。','SmallZH')
new();h('5. Evaluation｜模型評估')
comparison=['Mean baseline','OLS full']
for prefix in ['Ridge alpha=','Lasso alpha=']:
    comparison.append(next(r['scheme'] for r in ranking if r['scheme'].startswith(prefix)))
comparison+=['SelectKBest k=14','SelectKBest k=16','SelectKBest k=18','Histogram gradient boosting (comparison)']
table(['模型','MAE','MSE','RMSE','R²'],[[n.replace('Histogram gradient boosting (comparison)','HistGB 對照'),num(by[n]['MAE']),num(by[n]['MSE']),num(by[n]['RMSE']),num(by[n]['R2'])] for n in comparison],[165,85,85,85,85])
p('MAE 為平均絕對誤差；MSE 為誤差平方平均；RMSE 與目標同尺度；R² 相對常數平均值解釋變異，可以小於零。Adjusted R² = 1 - (1-R²)(n-1)/(n-k-1)。本作業測試 n=10000，OLS 使用名目輸入數 k；對懲罰或非線性模型只作描述，不等同有效自由度調整。')
p(f'正式模型的 Adjusted R² = {num(by[winner]["Adjusted_R2"])}。原始輸出超出 0～1 的筆數為 {by[winner]["outside_0_1"]}；不自動裁切。裁切後 MSE = {num(interval["clipped_metrics"]["MSE"])}，只作敏感度比較，不改變正式排名。')
image('07_model_comparison.png',maxheight=230,caption='圖 4：同一資料與切分下，基準、線性與非線性方法比較。此版本結構線性，複雜模型未必更好。')
new();h('5.1 預測圖與 95% 預測區間')
image('03_test_predictions.png',maxheight=220,caption='圖 5：正式模型實際值對預測值及測試殘差。微小殘差在 1e-16 尺度，只是浮點計算誤差。')
p('區間採 split conformal [5]。模型在 30,000 筆開發資料擬合後凍結，校準集分數 s_i = |y_i - f(x_i)|。取 ceil((10000+1) × 0.95) = 9501 的排序分數 q，輸出 [f(x)-q, f(x)+q]。')
p('它是新觀測目標的預測區間，不是平均迴歸線的信賴區間。在資料交換性假設下提供有限樣本邊際涵蓋率；不保證每個固定輸入的條件涵蓋率，也不保證分布改變後仍有效。校準集不可再用來調整模型或挑選區間寬度。')
image('05_prediction_intervals.png',maxheight=220,caption=f'圖 6：完整模型 95% 區間。全體測試涵蓋率 {interval["test_coverage"]:.2%}，平均寬度 {interval["mean_width"]:.3e}。圖中 120 筆依預測值排序，並非時間順序。')
p('此資料的完整模型幾乎沒有隨機殘差，區間退化到浮點精度。離散分數與數值誤差會使實測涵蓋率高於名目值；不代表真實洪災風險沒有不確定性。','SmallZH')
new();h('5.2 刪減特徵的區間示範')
image('09_reduced_prediction_intervals.png',maxheight=300,caption='圖 7：固定 SelectKBest k=14 的預測區間。刪去因素後產生資訊損失，區間有可辨識的寬度。')
table(['項目','完整 OLS','14 特徵示範'],[['名目涵蓋率','95%','95%'],['半寬 q',num(interval['radius']),num(reduced['radius'])],['測試涵蓋率',f'{interval["test_coverage"]:.2%}',f'{reduced["test_coverage"]:.2%}'],['平均寬度',num(interval['mean_width']),num(reduced['mean_width'])]],[160,170,175])
p('14 特徵模型是預先定義的資訊損失示範，不因為區間較好看就改選它為正式模型。兩個模型分別用同一組獨立校準資料建立區間；沒有挑選校準分數最漂亮的模型。單次測試的 94.58% 不必恰好等於 95%，不能把一次實驗誤認為保證失效。')
p('本機評估使用原始、未裁切的區間。若實際應用要將區間與 [0,1] 相交，必須揭露處理方式並另外檢查交換性、量測定義與外部有效性。')
new();h('5.3 五項回歸假設檢查')
table(['假設','完整 OLS 與資料限制','14 特徵探索診斷'],[['線性','開發集生成關係近似精確線性；殘差為浮點誤差，RESET 不解讀。',f'RESET p={rd["RESET_p"]:.4g}；配合殘差圖，不以未拒絕當成已證明。'],['誤差獨立性','沒有時間／空間座標，不能建立合理排序，獨立性未被驗證。',f'DW={rd["DW_row_order_descriptive_only"]:.4f}，僅原始列序描述，不作正式獨立性結論。'],['等變異','浮點殘差下 BP 會受數值結構支配，標為不適用。',f'BP p={rd["BP_p"]:.4g}，僅選特徵後探索性檢查。'],['常態性','殘差近零不形成可解釋的隨機分布，JB 不適用。',f'JB p={rd["JB_p"]:.4g}；配合 Q-Q 圖，留意大樣本敏感性。'],['多重共線性',f'最大 VIF={diag["max_VIF"]:.4f}；標準化設計矩陣秩={diag["full_design_rank"]}。','欄位近獨立；低 VIF 不等於具有因果意義。']],[75,215,215])
image('10_reduced_assumptions.png',maxheight=220,caption='圖 8：14 特徵模型在開發集的殘差、分布及 Q-Q 圖，提供非退化的診斷示範。')
p('14 特徵診斷是在選特徵後進行，p 值未校正選擇程序，僅供探索。完整模型殘差標準差約 '+num(diag['residual_sd'])+'；程式刻意將不適用的正式檢定輸出設為 null，而不呈現誤導的顯著性。')
p('OLS 點估計不要求誤差必然常態；常態性主要關係傳統小樣本推論。Split conformal 不依賴殘差常態或等變異，但仍需要交換性，且固定寬度區間不保證條件涵蓋。','SmallZH')
new();h('6. 同題研究與主流方法比較')
p('研究來源在報告 [1]～[5] 與 RESEARCH_SOURCES.md 中可追溯。不同版本的洪水資料不應只憑相同欄名就混用。')
table(['方法／來源','可借鏡之處','比較限制'],[['同題線性回歸 [2]','以 EDA、切分、線性模型和指標串成可重現流程。','使用 Playground 競賽資料；標題與版本分數可能不同，不搬用其分數當本地結果。'],['特徵總和與排序 [3]','研究資料生成機制；考慮總和、排序等對稱特徵。','競賽資料與此 50,000 筆版本不同；不複製其排行榜結論。'],['Ridge / Lasso','正則化在多重共線性或雜訊時可能有益。','此版本低共線性且目標近確定，懲罰會引入偏差。'],['梯度提升對照','同一分割實際執行 HistGradientBoosting，比較非線性彈性。','它不是所有樹模型的最佳調參結果，也不能代表 XGBoost／CatBoost 上限。'],['Conformal [5]','校準獨立殘差，明確區分點預測與區間涵蓋。','沒有真實時空資料，不能將交換性或外部涵蓋率視為已被驗證。']],[115,195,195])
p('本作業的公平量化比較，僅採同一本地資料、同樣切分與 CV 的實際重跑分數。沒有引用外部 Kaggle 名次作自己的成績，也沒有宣稱主流複雜模型必然優於線性回歸。')
sub('限制與可改善方向')
p('若目標是實際洪災預報，應先取得可驗證來源、有時間／地理標記與量測定義的資料，採時間或空間外推驗證，檢查分布漂移與各群組涵蓋率。對此教學版本增加模型複雜度，不能解決外部有效性不足。')
new();h('7. NotebookLM 摘要與 AI 協作')
if notebook_ok:
    text=summary_file.read_text(encoding='utf-8-sig')
    for para in text.strip().split('\n\n'):p(para)
else:p('尚未取得真實 NotebookLM 回答。需本人操作後補入；此段不以 GPT 模擬摘要替代。')
sub('7.1 GPT 輔助內容')
p('GPT 協助釐清作業要求、查詢可取得的資料來源、建立與除錯 Python 程式、規劃不洩漏的模型比較、產生圖表、製作互動頁面與本報告。所有實驗數字取自實際執行輸出，不由文字模型填造。')
p('學生貼回的 NotebookLM 回答已保存為 evidence/notebooklm_raw.txt 的閱讀版；Markdown 與公式轉為純文字。閱讀版用於來源對照，不是原始完整 GPT 對話 PDF。GPT 編修摘要與學生提供的研究內容分開標示，避免把核對改寫誤當 NotebookLM 原始逐字輸出。')
sub('7.2 原始回答核對')
table(['需修正內容','本報告採用的敘述'],[['全部 int64','輸入為 int64；目標 FloodProbability 實際為 float64。'],['選特徵提高準確率','可能改善效率與泛化，但不保證；本次完整特徵最好。'],['任何新樣本都 95%','需交換性；邊際涵蓋，不是每個固定情境的條件保證。']],[165,340])
new();h('8. Deployment｜部署與可重現操作')
p('主程式：5115056023_hw2.py。先安裝 requirements.txt，執行主程式產生模型、圖表、CSV 和 dashboard_data.json；再執行 build_report.py 與 package_submission.py。')
p('單筆預測：python predict_sample.py --input example_input.json。此程式載入 prediction_bundle.joblib，使用開發集正式模型及其獨立校準半寬。沒有在全部資料重訓後沿用舊區間。')
p('網頁位於 web/index.html，可本機離線打開；部署後位於原 GitHub Pages 的 /hw2/ 子路徑。它只展示 Python 匯出的原始尺度係數，不在 GitHub Pages 執行 Python。舊 50 Startups 首頁保留。')
p('依輸入觀測範圍標示外推；不自動把無效輸入當成零，不把未知分數包裝成真實洪災警報。')
sub('8.1 繳交完整性')
table(['項目','狀態'],[['主程式、資料來源、圖表、評估','已執行並輸出'],['NotebookLM 摘要','學生已貼回研究回答，GPT 核對編修；未獨立查驗筆記本連結' if notebook_ok else '待補'],['完整 GPT 對話 PDF','已附本人確認的原始 PDF' if conv_ok else '待補：需本人匯出舊聊天與新聊天並確認完整'],['Kaggle 名次','未提交，無名次'],['最終可繳交狀態','待本人核對' if status['ready_to_submit'] else '待補完整原始對話 PDF，不可宣稱齊備']],[160,345])
p('原始對話 PDF 請依 CONVERSATION_EXPORT_GUIDE.md 放入 evidence/ 並填寫本人確認的 conversation_manifest.json。build_report.py 只會合併實際存在且被確認的文件；不會以摘要或其他聊天內容充當完整紀錄。')
new();h('參考來源與實驗紀錄')
refs=[('[1] Kamrul Hasan. Flood Prediction Dataset. Kaggle, version 1.','https://www.kaggle.com/datasets/kamrulhasan0077/flood-prediction-dataset'),('[2] Sakshi Satre. Flood Prediction Using Linear Regression. Kaggle notebook.','https://www.kaggle.com/code/sakshisatre/flood-prediction-using-linear-regression-0-84558/notebook'),('[3] broccoli beef. Sorting along the feature axis as feature engineering. Kaggle discussion.','https://www.kaggle.com/competitions/playground-series-s4e5/discussion/499579'),('[4] scikit-learn. Feature selection, official documentation.','https://scikit-learn.org/stable/modules/feature_selection.html'),('[5] Angelopoulos, A. N. & Bates, S. A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. 2021, revised 2022.','https://arxiv.org/abs/2107.07511')]
for title,url in refs:p(title,'SmallZH');p(url,'SmallZH');story.append(Spacer(1,7))
sub('來源可重現識別');p('下載日期：2026-10-16；版本：1；授權標示：CC0: Public Domain。','SmallZH');p('SHA-256: '+manifest['source']['sha256'],'SmallZH')
sub('執行環境');p(json.dumps(manifest['versions'],ensure_ascii=False),'SmallZH');p(f'本次主程式執行約 {manifest["runtime_seconds"]:.1f} 秒，threadpool 限制 1 執行緒。實際分數保存於 CSV／JSON，不受報告四捨五入影響。','SmallZH')
sub('重要輸出');p('outputs/run_manifest.json：切分、設定與結果；cv_folds.csv：逐折分數；selected_features.json：選定欄位；test_predictions.csv：測試預測與區間；diagnostics.json 與 reduced_diagnostics.json：診斷適用限制；data/source.json 與 kaggle_metadata.json：來源。','SmallZH')
def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c7dcdc'));canvas.line(44,39,A4[0]-44,39);canvas.setFont(FONT,8);canvas.setFillColor(colors.HexColor('#63777d'))
    canvas.drawString(44,25,'5115056023 | HW2 | '+('證據齊備，請本人核對' if status['ready_to_submit'] else '待補：完整 GPT 對話 PDF'))
    canvas.drawRightString(A4[0]-44,25,str(doc.page));canvas.restoreState()
base=ROOT/'tmp/pdfs/report_body.pdf';base.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(base),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=43,bottomMargin=53,title='5115056023 HW2 洪水機率多元線性回歸',author='5115056023 (GPT assisted)')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
final=PDF/'5115056023_hw2_report.pdf'
writer=PdfWriter();writer.append(str(base))
if conv_ok:
    for file in conv_files:writer.append(str(file))
writer.add_metadata({'/Title':'5115056023 HW2 Flood Regression Report','/Author':'5115056023; GPT assisted','/Subject':'CRISP-DM; student-provided NotebookLM summary; transcript status disclosed'})
with final.open('wb') as f:writer.write(f)
reader=PdfReader(str(final));assert len(reader.pages)>=10
text='\n'.join(page.extract_text() for page in reader.pages)
assert 'NotebookLM' in text and '5115056023' in text and 'FloodProbability' in text
print('PDF:',final);print('Pages:',len(reader.pages));print('Submission ready:',status['ready_to_submit'])
