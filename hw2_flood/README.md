# 5115056023 HW2：洪水機率多元線性回歸

新版獨立於原本 50 Startups；不覆蓋舊網站首頁。
網站部署路徑：https://simon-1224.github.io/AIOT_DOINCLASS/hw2/

## 目前繳交狀態
模型、五種特徵選擇、評估、預測區間、中文 PDF 和互動頁面已實作。
NotebookLM 研究回答已由學生貼回，保存閱讀版及 GPT 核對後摘要。
**仍缺完整原始 GPT 對話 PDF 與本人完整性確認，目前 ZIP 是待補版，不是完整繳交版。**
以 outputs/submission_status.json 為準；package_submission.py 會列出待辦。

## 執行環境與步驟
建議 Python 3.13；requirements.txt 記錄實際驗證的套件版本。
在本資料夾開啟終端機：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe 5115056023_hw2.py
.\.venv\Scripts\python.exe predict_sample.py --input example_input.json
.\.venv\Scripts\python.exe build_report.py
.\.venv\Scripts\python.exe package_submission.py
```

中文 PDF 優先使用 Windows 的 Microsoft JhengHei（msjh.ttc）；其他環境可設定 HW2_CJK_FONT 為本機可嵌入的中文字型路徑，沒有時使用 ReportLab CJK fallback。請視覺檢查字型。
不需要在系統安裝 LaTeX。圖表使用 Matplotlib，PDF 使用 ReportLab。
模型已附於本機 ZIP，重跑程式可再生；GitHub 不存放二進位模型。
如果資料檔未附，執行 python download_data.py，可由 Kaggle 官方 API 下載 version 1。

## CRISP-DM
1. Business Understanding：預測連續 FloodProbability，僅作教學。
2. Data Understanding：50,000 筆、20 個整數特徵、一個浮點目標；核對缺值、重複與來源。
3. Data Preparation：開發 30,000／校準 10,000／測試 10,000；種子 42，Pipeline 各折學習補值與標準化。
4. Modeling：OLS、Ridge、Lasso，五種特徵選擇各比較 14/16/18/20 特徵。
5. Evaluation：開發五折 CV 排名，測試不挑模型；五種指標、殘差、回歸診斷、95% split-conformal 區間。
6. Deployment：predict_sample.py、prediction_bundle.joblib、web/index.html。

## 重要實驗發現
此下載版本在開發集顯示 FloodProbability 幾乎等於 20 個特徵總和乘以 0.005；完整 OLS 的 R² 接近 1，誤差約浮點精度。
不要把它宣稱為真實洪災預測能力。來源上傳者承認外部來源、蒐集方式與地理覆蓋未知，合成性質也未被獨立驗證。
完整模型退化的區間之外，另提供固定 14 特徵模型的非退化區間示範。實際數字以 outputs JSON/CSV 為準。
完整模型殘差為浮點誤差，正式殘差檢定標示不適用，另附14特徵模型探索診斷。無時間地理索引，獨立性未證實。

## 特徵選擇與評估
SelectKBest、RFE、Forward SFS、Lasso coefficient ranking、Permutation Importance。
每個 CV 訓練折內，選擇器固定抽樣最多 2,500 筆；模型回歸使用完整訓練折。
SFS 使用訓練內三折；Permutation 使用子樣本內70/30再分割。最後模型以CV平均MSE決定。
34 個設定包含平均值基準與固定參數 HistGradientBoosting 對照；不對20特徵窮舉。
k=20時五種選擇方法與OLS full是相同方案，不將同分解讀為不同模型的優勝。
Adjusted R² 對懲罰/非線性模型只是名目特徵數版本，非有效自由度推論。

## 95% 預測區間
使用獨立校準的絕對殘差，第 ceil((10000+1)*0.95)=9501 小值為半寬 q。
模型固定使用開發資料訓練，沒有把模型全資料重訓後繼續沿用舊半寬。
交換性假設下為邊際涵蓋，不是個別地區的條件保證；未在真實時空資料上驗證。

## 資料與授權
Kaggle：https://www.kaggle.com/datasets/kamrulhasan0077/flood-prediction-dataset
Version 1；下載日期2026-10-16；頁面標示 CC0: Public Domain。
完整 metadata、SHA-256、欄位清單見 data/source.json 與 data/kaggle_metadata.json。
本作業依上傳頁授權標示保存及附上資料，不宣稱已驗證上傳者完整權利鏈。

## NotebookLM 與 GPT 輔助
學生貼回的 NotebookLM 回答：evidence/notebooklm_raw.txt（Markdown與公式正規化閱讀版）。
編修摘要：evidence/notebooklm_summary.txt。修正目標 dtype、選特徵不保證改善、交換性與邊際涵蓋。
未取得可點選引用標記或筆記本連結，這項證據限制已在報告標示。
GPT 協助程式、研究、圖表、報告；實驗分數來自真實程式執行。
完整對話PDF仍需本人匯出，依 CONVERSATION_EXPORT_GUIDE.md 完成。
**不要將完整私人對話上傳公開 GitHub。** 本地報告可附原始對話，公開網站只發佈分析與互動結果。

## 檔案
- 5115056023_hw2.py：分析入口。
- hw2_pipeline.py：可序列化的自訂選擇器。
- outputs/：分數、逐折結果、切分、特徵與圖表。
- output/pdf/5115056023_hw2_report.pdf：中文報告及實際存在的對話附錄。
- web/index.html：離線互動介面，Python 匯出模型係數，非伺服器即時重訓。
- NOTEBOOKLM_GUIDE.md、CONVERSATION_EXPORT_GUIDE.md：證據指引。
- package_submission.py：驗證後打包 5115056023_hw2.zip，缺附件時明確標示待補。

沒有提交 Kaggle，沒有名次。外部同題方法分數不直接與本地不同資料版本比較。
