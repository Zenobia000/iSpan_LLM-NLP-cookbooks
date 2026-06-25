# M3 結構收斂 — 投影片生成提示詞

> 視覺風格：深色系精緻簡報 × 插圖手繪混搭 ｜ 比例 16:9 ｜ 模組主色：霧霾鴨綠 muted teal #5E948C（收斂層主色，低飽和、無霓虹） ｜ 工具：gpt-image-2（draw skill）
> 文字：標題/內文皆為乾淨平整向量級無襯線字、字寬自然不壓縮；手繪感只用於插圖與標註。
> 用法：逐頁複製「生圖提示詞」餵入生圖（建議 `--size 1536x1024 --quality medium` 以求中文清晰）；每頁文字精簡，中文為主、術語保留英文。
> 所屬：收斂層

## 全模組共用設計系統 token

**視覺/配色（6 條）**
- 底色：深炭灰到墨黑平滑細微漸層（#101010 → #1E1E1E），疊極淡幾乎不可見紋理，保持平整專業，不要厚重黑板/粗紙紋。
- 主體：暗底＋粉筆白 #ECE8E0 為絕對主體；副文字暖灰 #9A958C。
- 主色點綴：霧霾鴨綠 muted teal #5E948C 僅作節制重點（圈選、關鍵詞、引線、節點高亮），低飽和霧面。
- 手繪只用於插圖與標註層：粉筆/蠟筆/低飽和 marker 質感的箭頭、圈選、波浪底線、隨手框、便利貼、★、↗ 引線；線條刻意不工整、霧面不發光。
- 嚴禁：霓虹、螢光、發光 glow/光暈、高彩度電光色、俗氣 3D、stock photo 人像、彩虹漸層、浮水印。
- 標題/內文一律乾淨向量感，不可手寫化、不可壓縮。

**版面/字體/字級（7 條）**
- 16:9 橫向；四周安全邊距約 7–9%，內容不貼邊、不侵入四角。
- 上下各留一條乾淨空白帶供後製：不畫模組編號徽章、不畫課程名、不畫層別標、不畫頁碼、不畫頁首頁尾列。
- 兩帶結構：標題帶（上）＝小型 kicker/eyebrow 段落歸屬（含 muted teal 小圓點）＋ H1 頁標題，皆左對齊；主視覺帶（中）＝最大區塊放圖解/流程/對比/卡片，垂直置中。
- 字體：中文思源黑體 / Noto Sans TC（標題 Bold、內文 Regular），向量級乾淨平整、字寬自然不壓縮不擠窄、字距均勻；英數術語 Inter；程式碼 JetBrains Mono；手寫標註才用 marker/chalk 感。
- 字級階層：封面 Display 特大（畫面高約 1/6–1/5）；內頁 H1 大而醒目（約 Display 60%）；kicker 小、字距略寬；body 中等、行距約 1.5；caption 小。
- 標題與內文大小至少差 2–2.5 倍，層級一眼可辨；同層級字級全頁一致。
- 單頁聚焦一觀念，主視覺元素群組 ≤5，寧缺勿擠；左對齊基線網格、統一間距、留白呼吸感。

---

## 投影片 1 — 封面：結構收斂
- 段落：封面
- 版面槽位：Display 主標題｜一句副標｜主視覺（不含徽章/頁碼，上下留後製空白帶）
- 頁面文字（精準短中文）：「結構收斂」「把『輸出長什麼樣』的決定權，從模型自由發揮收回到你定義的 schema」「Structured Output」
- 生圖提示詞：
一張精緻深色系 Keynote 封面，表達「把輸出格式的控制權從模型收回到 schema」。底色 #101010→#1E1E1E 平滑細微漸層、極淡紋理保持平整。16:9，安全邊距 8%，上下各留乾淨空白帶供後製，不畫任何模組編號徽章、課程名、層別、頁碼、頁首頁尾列。版面左對齊：左中放 Display 特大粗體無襯線中文主標題「結構收斂」（思源黑體 Bold，暖白 #ECE8E0，字寬自然不壓縮），其下一句中等副標「把『輸出長什麼樣』的決定權，從模型自由發揮收回到你定義的 schema」（暖灰 #9A958C），副標下小字英文「Structured Output」（Inter，muted teal #5E948C）。右側主視覺：一個自由散漫的雲狀手繪輪廓（代表模型自由發揮）被一個乾淨的方角「{ }」大括號框收束成整齊卡片，括號用 muted teal #5E948C marker 手繪質感。手繪層：從雲到括號畫一條 muted teal ↗ 收束箭頭、卡片角落一個粉筆白★。暗底＋粉筆白主體＋鴨綠節制點綴，無霓虹無發光。需渲染中文：結構收斂／把『輸出長什麼樣』的決定權，從模型自由發揮收回到你定義的 schema；英文：Structured Output。

---

## 投影片 2 — 開場：這一層要收回什麼控制權
- 段落：開場
- 版面槽位：kicker「開場」｜H1「從『希望它是 JSON』到『保證符合 schema』」｜主視覺：請求 vs 保證對照
- 頁面文字（精準短中文）：H1「從『希望它是 JSON』到『保證符合 schema』」；關鍵詞「請用 JSON＝一句請求」「會炸 pipeline 的地雷」「把約束從 prompt 層搬到 API／型別層」
- 生圖提示詞：
一張深色系教學簡報，對比「prompt 裡寫請用 JSON 只是請求」與「保證符合 schema」。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker 小字「開場」前綴一個 muted teal #5E948C 小圓點；其下 H1 粗體無襯線大字「從『希望它是 JSON』到『保證符合 schema』」（思源黑體 Bold 暖白，字寬自然）。主視覺帶分左右兩張卡：左卡標「請用 JSON」內含一段亂掉的輸出示意（多一層 markdown fence、漏一個欄位、數字變字串），手繪粉筆白叉叉與波浪底線標出三處錯；右卡標「保證符合 schema」內含整齊對齊的鍵值列、每行一個 muted teal 勾。卡間一條 muted teal ↗ 升級箭頭，箭頭旁手寫便利貼「把約束從 prompt 層搬到 API／型別層」。手繪層：左卡三個錯誤處圈粉筆白圈、右卡綠勾為鴨綠。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光。需渲染中文：開場／從『希望它是 JSON』到『保證符合 schema』／請用 JSON／保證符合 schema／把約束從 prompt 層搬到 API／型別層；英文：JSON、schema。

---

## 投影片 3 — 學習目標
- 段落：學習目標
- 版面槽位：kicker「學習目標」｜H1「這一章你會帶走的四件事」｜主視覺：四張編號卡
- 頁面文字（精準短中文）：H1「這一章你會帶走的四件事」；四卡「① 三層保證選對約束手段」「② responses.parse 取型別化結果」「③ 走完 function calling 三步協定」「④ Pydantic 在邊界驗證」
- 生圖提示詞：
一張深色系簡報，列出四個學習目標。底色 #101010→#1E1E1E 平滑漸層、極淡紋理。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「學習目標」含 muted teal #5E948C 小圓點；H1 粗體大字「這一章你會帶走的四件事」（暖白、字寬自然不壓縮）。主視覺帶：四張等寬卡片橫向排列、間距均勻，每卡左上一個 muted teal 大號碼 ①②③④，卡內中等內文：①「三層保證選對約束手段」②「responses.parse 取型別化結果」③「走完 function calling 三步協定」④「Pydantic 在邊界驗證」。卡片以細霧面框線、留白充足。手繪層：在第①卡與第④卡角落各畫一個粉筆白★，第②卡的 responses.parse 下畫 muted teal 波浪底線。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光，元素群組僅四卡。需渲染中文：學習目標／這一章你會帶走的四件事／三層保證選對約束手段／取型別化結果／走完 function calling 三步協定／在邊界驗證；英文：responses.parse、function calling、Pydantic。

---

## 投影片 4 — 核心①：三層保證階梯
- 段落：核心概念脈絡（3.1）
- 版面槽位：kicker「核心概念 ①」｜H1「三層保證：從『請求』到『強制』」｜主視覺：三階梯
- 頁面文字（精準短中文）：H1「三層保證：從『請求』到『強制』」；階梯「Prompt 要求｜無約束」「JSON mode｜弱：保證可解析」「Structured Outputs｜強：保證符合 schema」
- 生圖提示詞：
一張深色系簡報，用上升階梯表達三層約束力。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「核心概念 ①」含 muted teal #5E948C 小圓點；H1 粗體大字「三層保證：從『請求』到『強制』」（暖白、字寬自然）。主視覺帶：由左下往右上的三級階梯，每級一張卡：第一級「Prompt 要求」標「無約束」、第二級「JSON mode」標「弱：保證可解析」、第三級（最高、以 muted teal 高亮）「Structured Outputs」標「強：保證符合 schema」。階梯左側一條向上的箭頭標「約束力遞增」。手繪層：第三級頂端畫一個 muted teal 隨手框圈住、旁邊便利貼「約束在哪一層，差別就在這」、底部一條粉筆白波浪底線。暗底粉筆白主體、鴨綠只點綴最高級，無霓虹無發光，元素群組三卡＋一箭頭。需渲染中文：核心概念 ①／三層保證：從『請求』到『強制』／無約束／弱：保證可解析／強：保證符合 schema／約束力遞增；英文：Prompt、JSON mode、Structured Outputs。

---

## 投影片 5 — 核心②：約束發生在哪一層
- 段落：核心概念脈絡（3.1 關鍵差異）
- 版面槽位：kicker「核心概念 ②」｜H1「事後檢查 vs 事前約束」｜主視覺：解碼過程對比
- 頁面文字（精準短中文）：H1「事後檢查 vs 事前約束」；對比「prompt：拜託模型自律」「schema：每個 token 生成時就被限制走向合法結構」；心法「不要用 prompt 防漏欄位，要用 schema 讓漏欄位變成不可能」
- 生圖提示詞：
一張深色系簡報，對比「事後檢查壞輸出」與「在解碼時就限制只能合法」。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「核心概念 ②」含 muted teal #5E948C 小圓點；H1 粗體大字「事後檢查 vs 事前約束」（暖白、字寬自然）。主視覺帶上下兩條 token 生成流程：上排「事後檢查」——一串方塊 token 自由生成後尾端才被一個檢查放大鏡攔下、旁標「prompt：拜託模型自律」；下排「事前約束」——同一串 token 每一格上方都有 muted teal 護欄/閘門限制走向、旁標「每個 token 生成時就被限制走向合法結構」。底部置中一條心法橫幅。手繪層：上排放大鏡旁粉筆白問號、下排每個閘門用 muted teal marker 描邊、心法句下波浪底線。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光。需渲染中文：核心概念 ②／事後檢查 vs 事前約束／prompt：拜託模型自律／每個 token 生成時就被限制走向合法結構／不要用 prompt 防漏欄位，要用 schema 讓漏欄位變成不可能；英文：token、prompt、schema。

---

## 投影片 6 — 核心③：Structured Outputs 拿型別化結果
- 段落：核心概念脈絡（3.2）
- 版面槽位：kicker「核心概念 ③」｜H1「Pydantic model → schema → 型別化物件」｜主視覺：三段流程
- 頁面文字（精準短中文）：H1「Pydantic model → schema → 型別化物件」；流程「定義 BaseModel」「SDK 轉 JSON Schema 約束解碼」「output_parsed 已是物件，不用再 json.loads()」
- 生圖提示詞：
一張深色系簡報，呈現 Structured Outputs 的三段資料流。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「核心概念 ③」含 muted teal #5E948C 小圓點；H1 粗體大字「Pydantic model → schema → 型別化物件」（暖白、字寬自然）。主視覺帶：三個節點由左到右、以 muted teal 連接箭頭串起。節點一卡片「定義 BaseModel」內含等寬 JetBrains Mono 三行欄位示意 name/age/is_subscriber；節點二齒輪圖示「SDK 轉 JSON Schema 約束解碼」；節點三整齊物件卡「output_parsed 已是物件」並標「不用再 json.loads()」。手繪層：節點三用 muted teal 隨手框圈住、旁便利貼「保證欄位齊、型別對」、json.loads() 上畫粉筆白刪除斜線表「省去」。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光，元素群組三節點。需渲染中文：核心概念 ③／定義／約束解碼／已是物件／不用再／保證欄位齊、型別對；英文：Pydantic、BaseModel、JSON Schema、output_parsed、json.loads()。

---

## 投影片 7 — 核心④：Function calling 三步協定
- 段落：核心概念脈絡（3.3）
- 版面槽位：kicker「核心概念 ④」｜H1「Function calling：請求—執行—回填—再請求」｜主視覺：循環流程
- 頁面文字（精準短中文）：H1「請求—執行—回填—再請求」；四步「① 帶 tools 發請求」「② 讀 function_call：name／arguments／call_id」「③ 執行本地函式並回填 function_call_output」「④ 帶結果再請求出最終答案」
- 生圖提示詞：
一張深色系簡報，呈現 function calling 的四步迴圈協定。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「核心概念 ④」含 muted teal #5E948C 小圓點；H1 粗體大字「請求—執行—回填—再請求」（暖白、字寬自然）。主視覺帶：一個由四個節點構成的環形/弧形流程，順時針以 muted teal 箭頭連接。節點①「帶 tools 發請求」、②「讀 function_call：name／arguments／call_id」、③「執行本地函式並回填 function_call_output」、④「帶結果再請求出最終答案」回到模型。中央放一個小圖示：左邊「Tool schema（模型看見的 contract）」與右邊「Python 函式（真正執行）」用一條虛線對齊相連。手繪層：在 call_id 上畫 muted teal 圈並引線標「不可省：對應哪一次呼叫」、中央虛線旁便利貼「兩者要對得上」。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光。需渲染中文：核心概念 ④／請求—執行—回填—再請求／帶 tools 發請求／讀／執行本地函式並回填／帶結果再請求出最終答案／不可省：對應哪一次呼叫／兩者要對得上；英文：function_call、name、arguments、call_id、function_call_output、Tool schema。

---

## 投影片 8 — 核心⑤：鎖死工具參數
- 段落：核心概念脈絡（3.4）
- 版面槽位：kicker「核心概念 ⑤」｜H1「strict 與 tool_choice：兩個鎖」｜主視覺：兩把鎖卡片
- 頁面文字（精準短中文）：H1「strict 與 tool_choice：兩個鎖」；卡一「strict=True + additionalProperties=False：鎖死參數、禁止自創欄位」；卡二「tool_choice：auto 自行判斷 vs 強制呼叫特定工具」；分界線「只要結構化資料 → 用 Structured Outputs，別騙假工具」
- 生圖提示詞：
一張深色系簡報，講兩個讓 function calling 可信賴的開關。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「核心概念 ⑤」含 muted teal #5E948C 小圓點；H1 粗體大字「strict 與 tool_choice：兩個鎖」（暖白、字寬自然）。主視覺帶兩張並排卡片各配一個手繪鎖圖示：左卡「strict=True + additionalProperties=False」內文「鎖死參數、禁止自創欄位」；右卡「tool_choice」內文「auto 自行判斷 vs 強制呼叫特定工具」。卡片下方一條置中分界橫幅「只要結構化資料 → 用 Structured Outputs，別騙假工具」。手繪層：兩個鎖圖示用 muted teal marker 描邊、分界橫幅左端畫一個粉筆白分隔線與★。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光，元素群組兩卡＋一橫幅。需渲染中文：核心概念 ⑤／兩個鎖／鎖死參數、禁止自創欄位／自行判斷／強制呼叫特定工具／只要結構化資料／別騙假工具；英文：strict=True、additionalProperties=False、tool_choice、auto、Structured Outputs。

---

## 投影片 9 — 核心⑥：Pydantic 是 schema 單一真實來源
- 段落：核心概念脈絡（3.5）
- 版面槽位：kicker「核心概念 ⑥」｜H1「一個 model，同時是 schema 與驗證器」｜主視覺：model 中心輻射五件
- 頁面文字（精準短中文）：H1「一個 model，同時是 schema 與驗證器」；五件「Field(description) 給模型提示」「Literal 鎖死值域」「巢狀 BaseModel」「model_validate_json 最後一道閘」「ValidationError 接壞資料」；技巧「放 reasoning 欄位先推理再結論」
- 生圖提示詞：
一張深色系簡報，呈現 Pydantic model 作為 schema 與驗證器的中心輻射圖。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「核心概念 ⑥」含 muted teal #5E948C 小圓點；H1 粗體大字「一個 model，同時是 schema 與驗證器」（暖白、字寬自然）。主視覺帶：中央一張 muted teal 高亮卡「Pydantic model」，向外輻射五個節點：「Field(description) 給模型提示」「Literal 鎖死值域」「巢狀 BaseModel」「model_validate_json 最後一道閘」「ValidationError 接壞資料」。節點用細霧面引線連回中央。手繪層：在 model_validate_json 節點畫 muted teal 隨手框圈住、旁便利貼「reasoning 欄位先推理再結論」、ValidationError 旁畫粉筆白盾牌小圖。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光，輻射節點五個。需渲染中文：核心概念 ⑥／同時是 schema 與驗證器／給模型提示／鎖死值域／巢狀／最後一道閘／接壞資料／先推理再結論；英文：Pydantic model、Field(description)、Literal、BaseModel、model_validate_json、ValidationError、reasoning。

---

## 投影片 10 — 核心⑦：應用整合（分類 + 串流 + Gradio）
- 段落：核心概念脈絡（3.6）
- 版面槽位：kicker「核心概念 ⑦」｜H1「先分類成結構化 category，再路由」｜主視覺：分類路由＋串流
- 頁面文字（精準短中文）：H1「先分類成結構化 category，再路由」；要點「Streaming delta：邊生成邊顯示」「Gradio 6 ChatInterface(type=messages)」「先分類 → 再路由到工具／知識庫」
- 生圖提示詞：
一張深色系簡報，呈現分類客服機器人的整合流程。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「核心概念 ⑦」含 muted teal #5E948C 小圓點；H1 粗體大字「先分類成結構化 category，再路由」（暖白、字寬自然）。主視覺帶：左側使用者訊息卡 → 中間一個「分類器」節點吐出 muted teal 結構化 category 標籤 → 右側分岔成兩條路由「工具」與「知識庫」。下方一條串流示意：一串逐漸點亮的文字 delta 方塊配標「Streaming delta：邊生成邊顯示」；右下角一個簡潔聊天視窗外框標「Gradio 6 ChatInterface(type=messages)」。手繪層：分類器節點用 muted teal 隨手框圈、分岔箭頭旁便利貼「先分類 → 再路由」、串流方塊末端粉筆白↗。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光。需渲染中文：核心概念 ⑦／先分類成結構化 category，再路由／邊生成邊顯示／先分類／再路由到工具／知識庫；英文：category、Streaming delta、Gradio 6、ChatInterface(type=messages)。

---

## 投影片 11 — 程式碼導讀①：三層保證對照
- 段落：程式碼導讀（01-json-mode）
- 版面槽位：kicker「程式碼導讀 ①」｜H1「01-json-mode：三段對比一眼看穿」｜主視覺：三欄程式概念對照
- 頁面文字（精準短中文）：H1「三段對比一眼看穿」；三欄「prompt-only：要自己 json.loads，可能漏欄位」「JSON mode：可解析，欄位它說了算」「parse：output_parsed 保證齊全」
- 生圖提示詞：
一張深色系程式碼導讀簡報，用三欄概念卡對比三層做法（不貼整段 code，只放關鍵 API 行）。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「程式碼導讀 ①」含 muted teal #5E948C 小圓點；H1 粗體大字「三段對比一眼看穿」（暖白、字寬自然）。主視覺帶三欄等寬卡片，每卡頂部一行 JetBrains Mono 等寬程式片段、下方一句結論：欄一頂行「responses.create(input=...)」結論「要自己 json.loads，可能漏欄位」；欄二頂行「text={format:json_object}」結論「可解析，欄位它說了算」；欄三（muted teal 高亮）頂行「responses.parse(text_format=Users)」結論「output_parsed 保證齊全」。手繪層：欄一結論旁畫粉筆白叉、欄三結論旁畫 muted teal 勾並隨手框圈整欄。暗底粉筆白主體、鴨綠只點綴第三欄，無霓虹無發光，三欄。需渲染中文：程式碼導讀 ①／三段對比一眼看穿／要自己／可能漏欄位／可解析，欄位它說了算／保證齊全；英文：responses.create、json.loads、json_object、responses.parse、output_parsed。

---

## 投影片 12 — 程式碼導讀②：三步協定的 code 落點
- 段落：程式碼導讀（02-function-calling-basics）
- 版面槽位：kicker「程式碼導讀 ②」｜H1「02-function-calling：三步落在哪幾行」｜主視覺：步驟對應 API
- 頁面文字（精準短中文）：H1「三步落在哪幾行」；三步「create(tools=…, strict=True)」「篩 item.type==function_call 讀 arguments/call_id」「json.loads 執行後回填 function_call_output」
- 生圖提示詞：
一張深色系程式碼導讀簡報，把 function calling 三步對應到關鍵 API（不貼整段 code）。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「程式碼導讀 ②」含 muted teal #5E948C 小圓點；H1 粗體大字「三步落在哪幾行」（暖白、字寬自然）。主視覺帶：垂直三段，每段左側一個 muted teal 大號碼、右側一行 JetBrains Mono 等寬關鍵程式：① 「responses.create(tools=[...], strict=True)」② 「[i for i in output if i.type=='function_call']」③ 「json.loads(args) → function_call_output(call_id=...)」。三段以細引線串接，末端回到「再 create() 出最終答案」。手繪層：在 call_id 上畫 muted teal 圈並引線「回填要對齊」、step① 的 strict=True 下畫粉筆白波浪底線。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光，三段。需渲染中文：程式碼導讀 ②／三步落在哪幾行／再／出最終答案／回填要對齊；英文：responses.create、tools、strict=True、type=='function_call'、json.loads、function_call_output、call_id。

---

## 投影片 13 — 程式碼導讀③：Pydantic 驅動擷取
- 段落：程式碼導讀（03-structured-extraction）
- 版面槽位：kicker「程式碼導讀 ③」｜H1「03-extraction：非結構文字 → 型別化物件」｜主視覺：擷取管線
- 頁面文字（精準短中文）：H1「非結構文字 → 型別化物件」；管線「Report(BaseModel)+Field」「model_validate_json 驗證成物件」「Literal 鎖值域、list[Speaker] 巢狀」「except ValidationError 接壞資料」
- 生圖提示詞：
一張深色系程式碼導讀簡報，呈現用 Pydantic 把逐字稿擷取成物件的管線（不貼整段 code）。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「程式碼導讀 ③」含 muted teal #5E948C 小圓點；H1 粗體大字「非結構文字 → 型別化物件」（暖白、字寬自然）。主視覺帶左到右管線：左「一段凌亂逐字稿」文件圖示 → 中「Report(BaseModel) + Field(description)」schema 卡（JetBrains Mono 欄位示意，含 Literal[...] 與 list[Speaker]）→ 右「整齊型別化物件」卡，連接箭頭中段標「model_validate_json」。右下一個小分支「except ValidationError」接到一個粉筆白盾牌。手繪層：Literal 旁畫 muted teal 圈標「鎖值域」、list[Speaker] 旁便利貼「巢狀」、validate 箭頭下波浪底線「最後一道閘」。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光。需渲染中文：程式碼導讀 ③／非結構文字 → 型別化物件／鎖值域／巢狀／最後一道閘；英文：Report、BaseModel、Field(description)、Literal、list[Speaker]、model_validate_json、ValidationError。

---

## 投影片 14 — 練習與驗收
- 段落：練習與驗收
- 版面槽位：kicker「練習與驗收」｜H1「交出一個 schema 即契約的分類器」｜主視覺：練習清單＋驗收檢核
- 頁面文字（精準短中文）：H1「交出一個 schema 即契約的分類器」；練習「三層保證各跑一次記錄壞輸出」「走完 function calling 三步」「Literal+model_validate_json 處理壞資料」；驗收「100% 通過 model_validate_json，無 ValidationError、無越界類別」
- 生圖提示詞：
一張深色系簡報，呈現練習任務與驗收檢核。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「練習與驗收」含 muted teal #5E948C 小圓點；H1 粗體大字「交出一個 schema 即契約的分類器」（暖白、字寬自然）。主視覺帶分左右：左欄「練習」三條手繪 checkbox 列「三層保證各跑一次記錄壞輸出」「走完 function calling 三步」「Literal + model_validate_json 處理壞資料」；右欄「驗收」一張 muted teal 高亮卡「100% 通過 model_validate_json」下標「無 ValidationError、無越界類別」。手繪層：右欄驗收卡角落畫粉筆白★、100% 下畫 muted teal 波浪底線、左欄三個 checkbox 其一打粉筆白勾。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光。需渲染中文：練習與驗收／交出一個 schema 即契約的分類器／練習／走完 function calling 三步／處理壞資料／驗收／無越界類別；英文：Literal、model_validate_json、ValidationError、100%。

---

## 投影片 15 — 驗收心法：不再需要防禦性解析
- 段落：練習與驗收（通過標準）
- 版面槽位：kicker「驗收心法」｜H1「真正收斂：schema 本身就是保證」｜主視覺：before/after 對照
- 頁面文字（精準短中文）：H1「真正收斂：schema 本身就是保證」；before「下游還要寫一堆 if 'key' in data 防漏欄位」；after「別人只看你的 Pydantic model 就知道輸出長什麼樣」
- 生圖提示詞：
一張深色系簡報，對比「需要防禦性解析」與「schema 即保證」。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「驗收心法」含 muted teal #5E948C 小圓點；H1 粗體大字「真正收斂：schema 本身就是保證」（暖白、字寬自然）。主視覺帶左右對照兩卡：左卡「未收斂」內含一堆雜亂的 if 'key' in data 防禦判斷（JetBrains Mono 示意、被粉筆白叉劃掉）標「下游還要防漏欄位」；右卡（muted teal 高亮）「已收斂」內含一張乾淨 Pydantic model 卡標「別人只看你的 model 就知道輸出長什麼樣」。兩卡間一個 muted teal ↗ 箭頭。手繪層：左卡亂判斷打粉筆白叉、右卡 model 用 muted teal 隨手框圈、箭頭旁便利貼「schema 即契約」。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光。需渲染中文：驗收心法／真正收斂：schema 本身就是保證／下游還要防漏欄位／別人只看你的 model 就知道輸出長什麼樣／schema 即契約；英文：if 'key' in data、Pydantic model。

---

## 投影片 16 — 重點回顧／承先啟後
- 段落：回顧
- 版面槽位：kicker「重點回顧」｜H1「結構收斂：把資料變成可被程式信賴」｜主視覺：三柱回顧＋承接箭頭
- 頁面文字（精準短中文）：H1「把資料變成可被程式信賴」；三柱「三層保證選對手段」「Structured Outputs + function calling」「Pydantic 在邊界驗證」；承接「這份可信資料 → 餵進 M4 檢索／M5 工具參數／M7 評估判準」
- 生圖提示詞：
一張深色系收束簡報，回顧 M3 三大支柱並承接下一層。底色 #101010→#1E1E1E 平滑漸層。16:9、安全邊距 8%、上下留乾淨後製空白帶，不畫徽章/頁碼/頁首頁尾。標題帶左對齊：kicker「重點回顧」含 muted teal #5E948C 小圓點；H1 粗體大字「把資料變成可被程式信賴」（暖白、字寬自然）。主視覺帶：三根並列的回顧柱卡「三層保證選對手段」「Structured Outputs + function calling」「Pydantic 在邊界驗證」，三柱下方匯流成一條 muted teal 箭頭向右指出，末端三個小標籤「M4 檢索」「M5 工具參數」「M7 評估判準」。手繪層：三柱頂端各一個粉筆白★、匯流箭頭旁便利貼「可被程式信賴的資料」、承接標籤用 muted teal 圈。暗底粉筆白主體、鴨綠節制點綴，無霓虹無發光，群組三柱＋承接。需渲染中文：重點回顧／把資料變成可被程式信賴／三層保證選對手段／在邊界驗證／可被程式信賴的資料／檢索／工具參數／評估判準；英文：Structured Outputs、function calling、Pydantic、M4、M5、M7。
