# 完整 GPT 對話 PDF 匯出（尚待完成）

請保留原始完整作業相關對話，不能以摘要代替。
1. 若對話在 ChatGPT 網頁，開啟完整對話並確認前面內容已載入。
2. 使用瀏覽器列印（Ctrl+P）並選擇「另存為 PDF」，或使用你已信任的匯出工具。先查看列印預覽，確認首尾對話與程式區塊未缺漏。
3. 若是 Codex 桌面聊天沒有直接列印選項，將完整可見對話逐段複製到 Word / Google Docs，保留使用者與助理標示及原始文字，然後匯出 PDF。這是轉存原始文字，不是重寫摘要。
4. 若舊聊天與新聊天分開，分別匯出，勿漏掉舊作業過程。
5. 將 PDF 放入 evidence/，檔名使用 gpt_conversation_01.pdf、gpt_conversation_02.pdf ...。
6. 本人逐份確認內容完整後建立 evidence/conversation_manifest.json，格式如下：
{"complete": true, "confirmed_by_student": true, "scope": "包含所有 HW2 作業相關舊聊天與新聊天；本人已核對首尾與程式區塊", "files": ["gpt_conversation_01.pdf", "gpt_conversation_02.pdf"]}
7. 重新執行 build_report.py，程式會將這些真實 PDF 依檔名順序合併到報告附錄；再執行 package_submission.py 更新壓縮檔。

尚未有這些證據時，報告與 ZIP 會明確標為待補，不能當作完整繳交版。
如有帳密、API token 或其他敏感資訊，請先確認沒有出現在匯出 PDF，再繳交給老師。
