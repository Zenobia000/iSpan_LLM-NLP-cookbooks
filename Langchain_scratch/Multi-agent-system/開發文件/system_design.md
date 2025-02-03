LLM 推理策略流程框架設計文件

本設計文件描述一個以 __大型語言模型 \(LLM\) 推理策略__ 為核心的系統框架，採用__金字塔結構__呈現，從頂層應用服務 \(Application Layer\) 到中間的領域層 \(Domain Layer\)，再到底層基礎 \(Prompt Toolbox\)，全面遵循 __SOLID__ 設計原則，便於開發人員（PG）進行實作與維運。

## 目錄

1. [設計背景與目標](#設計背景與目標)
2. [系統整體金字塔架構](#系統整體金字塔架構)
   - [2.1 金字塔層級說明](#21-金字塔層級說明)
   - [2.2 八大應用流程 (C1 ~ C8)](#22-八大應用流程-c1--c8)
   - [2.3 Mermaid 架構圖示](#23-mermaid-架構圖示)
3. [設計細節與功能擴展](#設計細節與功能擴展)
   - [3.1 應用層 (Application Layer)](#31-應用層-application-layer)
   - [3.2 領域層 (Domain Layer)](#32-領域層-domain-layer)
     - [3.2.1 Domain Services](#321-domain-services)
     - [3.2.2 Aggregates & Entities](#322-aggregates--entities)
   - [3.3 底層基礎 (Prompt Toolbox)](#33-底層基礎-prompt-toolbox)
4. [SOLID 原則的應用](#solid-原則的應用)
   - [4.1 單一職責原則（SRP）](#41-單一職責原則-srp)
   - [4.2 開放封閉原則（OCP）](#42-開放封閉原則-ocp)
   - [4.3 里氏替換原則（LSP）](#43-里氏替換原則-lsp)
   - [4.4 介面隔離原則（ISP）](#44-介面隔離原則-isp)
   - [4.5 依賴反轉原則（DIP）](#45-依賴反轉原則-dip)
5. [實作範例與提示](#實作範例與提示)
   - [5.1 範例類別與方法](#51-範例類別與方法)
   - [5.2 領域服務與工廠方法](#52-領域服務與工廠方法)
   - [5.3 應用層整合](#53-應用層整合)
6. [總結與開發建議](#總結與開發建議)
   - [總結](#總結)
   - [開發建議](#開發建議)

---

__設計背景與目標__

__背景__

現代軟體開發日益依賴 LLM（如 GPT 系列、Claude 等）進行各類推理、輔助決策與內容生成。然而，缺乏系統化設計可能導致提示詞散亂、邏輯耦合度高，進而增加維護困難度。

__目標__

1. __建立清晰分層的金字塔架構__：應用層 \(Application Layer\) → 領域層 \(Domain Layer\) → 底層基礎 \(Prompt Toolbox\)。
2. 結合 __提示詞工程__（Prompt Engineering）與 __DDD \+ SOLID__ 思維，實現高可維護、可擴展的 LLM 解決方案。
3. 提供 __SA/SD 分析設計__ 八大流程 \(C1 ~ C8\) 及實作參考，協助 PG 快速應對複雜需求。

__系統整體金字塔架構__

__2\.1 金字塔層級說明__

系統架構採用金字塔結構，分為三個層級：

1. __應用層 \(Application Layer\)__  
位於金字塔頂端，負責處理用戶請求與業務流程，調度下層服務。
2. __領域層 \(Domain Layer\)__  
中間層，承載核心業務邏輯與資料管理，處理具體的推理策略。
3. __底層基礎 \(Prompt Toolbox\)__  
底層，提供最基本的提示詞策略與函式庫，支持上層業務需求。

__2\.2 八大應用流程 \(C1 ~ C8\)__

系統設計遵循 SA/SD 的八大流程，依序處理從需求界定到知識內化的全流程：

1. __C1 問題界定 \(ProblemDefinitionService\)__
2. __C2 理解全貌 \(HolisticUnderstandingService\)__
3. __C3 建立系統模型 \(ModelingService\)__
4. __C4 假設與驗證 \(HypothesisService\)__
5. __C5 全方位檢視 \(ComprehensiveAppraisalService\)__
6. __C6 創造與調整 \(IdeationRefinementService\)__
7. __C7 決策與行動 \(DecisionImplementationService\)__
8. __C8 知識內化與傳達 \(InternalizationService\)__

__2\.3 Mermaid 架構圖示__

mermaid

複製

flowchart TB

    %% ============================

    %% = Application Layer

    %% ============================

    subgraph APP\["應用層 \(Application Layer\)"\]

    direction TB

    C1\["C1 問題界定"\] \-\-> C2\["C2 理解全貌"\]

    C2 \-\-> C3\["C3 建立系統模型"\]

    C3 \-\-> C4\["C4 假設與驗證"\]

    C4 \-\-> C5\["C5 全方位檢視"\]

    C5 \-\-> C6\["C6 創造與調整"\]

    C6 \-\-> C7\["C7 決策與行動"\]

    C7 \-\-> C8\["C8 知識內化與傳達"\]

    end

    %% ============================

    %% = Domain Layer

    %% ============================

    subgraph DM\["領域層 \(Domain Layer\)"\]

    direction TB

    subgraph DomainServices\["Domain Services"\]

    DS1\["PromptService<br/>\- 組合 / 呼叫提示詞策略 \(PHP/HSP/GoT\)"\]

    DS2\["HypothesisService<br/>\- 假設檢驗與結果評估"\]

    DS3\["GraphService<br/>\- 管理 GoT 多重推理路徑"\]

    end

    subgraph DomainAggregates\["Aggregates & Entities"\]

    AG1\["PromptAggregate<br/>聚合提示上下文"\]

    E1\["PromptEntity / VO"\]

    AG2\["HypothesisAggregate<br/>管理假設與驗證結果"\]

    E2\["HypothesisEntity / VO"\]

    AG3\["GraphAggregate<br/>思維圖 \(GoT\) 結構"\]

    E3\["GraphNode / VO"\]

    end

    end

    %% ============================

    %% = Toolbox \(Unit Layer\)

    %% ============================

    subgraph PT\["底層基礎 \(Prompt Toolbox\)"\]

    direction TB

    T1\["Progressive\-Hint Prompting \(PHP\)"\]

    T2\["Hint\-before\-Solving Prompting \(HSP\)"\]

    T3\["Hypothesis Testing Prompting"\]

    T4\["Graph of Thought \(GoT\)"\]

    end

    %% ============================

    %% Edges: Application \-> Domain

    %% ============================

    C1 \-\->|"\(呼叫\) defineProblem\(\)"| DS1

    C2 \-\->|"\(呼叫\) gatherContext\(\)"| DS1

    C3 \-\->|"\(呼叫\) buildModel\(\)"| DS1

    C4 \-\->|"\(呼叫\) verifyHypothesis\(\)"| DS2

    C5 \-\->|"\(呼叫\) appraisePlan\(\)"| DS1

    C5 \-\->|"\(呼叫\) appraisePlan\(\)"| DS2

    C6 \-\->|"\(呼叫\) refineIdeas\(\)"| DS1

    C6 \-\->|"\(呼叫\) refineIdeas\(\)"| DS2

    C6 \-\->|"\(呼叫\) refineIdeas\(\)"| DS3

    C7 \-\->|"\(呼叫\) finalizeDecision\(\)"| DS1

    C7 \-\->|"\(呼叫\) finalizeDecision\(\)"| DS2

    C7 \-\->|"\(呼叫\) finalizeDecision\(\)"| DS3

    C8 \-\->|"\(呼叫\) internalizeOutcome\(\)"| DS1

    C8 \-\->|"\(呼叫\) internalizeOutcome\(\)"| DS3

    %% ============================

    %% Edges: Domain Services \-> Aggregates

    %% ============================

    DS1 \-\->|"\(讀/寫\) 提示詞上下文"| AG1

    DS1 \-\->|"\(更新\) PromptEntity"| E1

    DS2 \-\->|"\(讀/寫\) 假設結果"| AG2

    DS2 \-\->|"\(更新\) HypothesisEntity"| E2

    DS3 \-\->|"\(讀/寫\) 思維圖資訊"| AG3

    DS3 \-\->|"\(更新\) GraphNode"| E3

    %% ============================

    %% Edges: Domain Services \-> Toolbox

    %% ============================

    DS1 \-\->|"\(使用\) PHP/HSP/GoT"| T1

    DS1 \-\->|"\(使用\) HSP"| T2

    DS2 \-\->|"\(使用\) Hypothesis Testing"| T3

    DS3 \-\->|"\(使用\) GoT 函式"| T4

__設計細節與功能擴展__

__3\.1 應用層 \(Application Layer\)__

__主要角色__：C1 ~ C8 八大 Service，各自對應 SA/SD 一個階段的需求。

__責任__：

- 接收使用者或外部系統的請求。
- 串接並調度領域層服務進行處理。
- 管理業務流程的順序與邏輯。

__不做什麼__：

- 不直接處理提示詞組合或假設檢驗的邏輯。
- 不管理資料儲存。

__範例流程 \(C4\) HypothesisService__

pseudo

複製

// Application Layer pseudo\-code

class HypothesisOrchestrator \{

    public Result executeHypothesisCheck\(HypothesisData input\) \{

        return domain\.hypothesisService\.verifyHypothesis\(input\);

    \}

\}

其中 domain\.hypothesisService 為領域層對應的物件。

__3\.2 領域層 \(Domain Layer\)__

__3\.2\.1 Domain Services__

- __PromptService__
	- 負責組合與呼叫底層提示策略（PHP, HSP, GoT）。
	- 方法範例：generateInitialPrompt\(\), refinePromptContext\(\), buildGraphFromPrompt\(\)。
	- 與 PromptAggregate / PromptEntity 互動，保存或更新提示詞上下文。
- __HypothesisService__
	- 透過 Hypothesis Testing Prompting 驗證多重假設，蒐集測試數據。
	- 更新 HypothesisAggregate / HypothesisEntity 狀態，供後續分析與決策引用。
- __GraphService__
	- 管理 Graph of Thought \(GoT\)，建立、分支、合併思維路徑。
	- 操控 GraphAggregate / GraphNode，支持複雜推理的視覺化。

__3\.2\.2 Aggregates & Entities__

- __PromptAggregate__
	- 包含多個 PromptEntity，記錄提示詞及上下文歷史（如先前解答、用戶輸入等）。
	- 方法範例：addNewPrompt\(\), combineContext\(\)。
- __HypothesisAggregate__
	- 聚合所有假設（多個 HypothesisEntity），包含其驗證結果、依賴前提、成功與否。
	- 方法範例：recordHypothesisOutcome\(hypoId, outcome\)。
- __GraphAggregate__
	- 維護完整的思維圖（GoT），包含多個 GraphNode。
	- 方法範例：addNode\(\), linkNodes\(\), markBranchAsResolved\(\)。

__重點__：

- Aggregates / Entities 僅負責資料與業務狀態，不直接呼叫底層 Toolbox。
- 任何提示詞或運算需求，皆由 Domain Service 協調並更新。

__3\.3 底層基礎 \(Prompt Toolbox\)__

提供最小粒度的 LLM 提示詞策略與函式庫，支援領域層的業務需求。

- __Progressive\-Hint Prompting \(PHP\)__
	- 漸進式引導 LLM 修正或細化答案，避免一次過度複雜指令造成誤差。
	- 函式範例：php\_generateHint\(context\), php\_refineHint\(hintList\)。
- __Hint\-before\-Solving Prompting \(HSP\)__
	- 先提供提示後再請 LLM 解題，搭配解題關鍵點或領域知識先行輸入。
	- 常用於需求定義或總結，例如：hsp\_prepareContext\(domainKeywords\)。
- __Hypothesis Testing Prompting__
	- 多重假設與反向推理，提升推理完整度。
	- 函式範例：testHypothesis\(hypo, data\)，輸出假設為真或假的關鍵跡象。
- __Graph of Thought \(GoT\)__
	- 以圖狀結構維持多條推理路徑，便於可視化或交互分析。
	- 基礎 API：createNode\(content\), createEdge\(nodeA, nodeB\), mergePaths\(\.\.\.\)。

__特性__：

- 這些函式屬於最底層的小單元，不含特定業務邏輯。
- 如何運用在業務上，由 Domain Services 負責協調。

__SOLID 原則的應用__

__4\.1 單一職責原則（SRP）__

- __Application Services \(C1~C8\)__：
	- 只關心 SA/SD 流程，不處理提示詞細節。
- __Domain Services__：
	- 各自針對不同領域職責（Prompt / Hypothesis / Graph）。
- __Aggregates / Entities__：
	- 僅維護資料與狀態，不承擔運算邏輯。

__4\.2 開放封閉原則（OCP）__

- 引入新提示策略（如 Chain\-of\-Thought, Self\-Consistency）：
	- 只需擴充 Toolbox 或 Domain Services 相應方法，不需修改現有服務邏輯。
- 擴充新 Domain Service（如 EvaluationService）：
	- 不破壞原有結構。

__4\.3 里氏替換原則（LSP）__

- 底層 PHP, HSP, HypothesisTesting 均實作相同介面 IPromptStrategy 的不同實作。
- __Domain Service \(PromptService\)__ 可以無縫替換策略，便於測試階段使用 Mock 物件替代真實 LLM 呼叫。

__4\.4 介面隔離原則（ISP）__

- __Application Layer__：
	- 呼叫 Domain Services 時，只需關注所需的方法，避免非必要方法干擾。
- __Domain Services__：
	- 呼叫 Toolbox 時，根據實際需求隔離在各自的接口或抽象類別中。

__4\.5 依賴反轉原則（DIP）__

- __高層 \(Application Layer\)__ 不依賴於低層 \(Prompt Toolbox\)。
- __依賴於抽象介面__：
	- 高層依賴於 Domain Services 的抽象介面。
	- Domain Services 反向依賴於抽象的 IPromptStrategy 或工具函式，而非具體實作。

__實作範例與提示__

__5\.1 範例類別與方法__

php

複製

// PromptService\.php

interface IPromptStrategy \{

    public function generateHint\($context\);

    public function refineHint\($hints\);

\}

class PHPStrategy implements IPromptStrategy \{

    public function generateHint\($context\) \{

        // PHP 相關提示生成邏輯

    \}

    public function refineHint\($hints\) \{

        // PHP 提示細化邏輯

    \}

\}

class PromptService \{

    private $promptStrategy;

    public function \_\_construct\(IPromptStrategy $strategy\) \{

        $this\->promptStrategy = $strategy;

    \}

    public function generateInitialPrompt\($context\) \{

        return $this\->promptStrategy\->generateHint\($context\);

    \}

    public function refinePromptContext\($hints\) \{

        return $this\->promptStrategy\->refineHint\($hints\);

    \}

\}

__5\.2 領域服務與工廠方法__

php

複製

// DomainServiceFactory\.php

class DomainServiceFactory \{

    public static function createPromptService\($strategyType\) \{

        switch \($strategyType\) \{

            case 'PHP':

                $strategy = new PHPStrategy\(\);

                break;

            case 'HSP':

                $strategy = new HSPStrategy\(\);

                break;

            // 其他策略

            default:

                throw new Exception\("Unknown strategy type"\);

        \}

        return new PromptService\($strategy\);

    \}

\}

__5\.3 應用層整合__

php

複製

// ApplicationLayer\.php

class HypothesisOrchestrator \{

    private $hypothesisService;

    public function \_\_construct\(HypothesisService $service\) \{

        $this\->hypothesisService = $service;

    \}

    public function executeHypothesisCheck\(HypothesisData $input\) \{

        return $this\->hypothesisService\->verifyHypothesis\($input\);

    \}

\}

__總結與開發建議__

__總結__

- __金字塔結構架構__：
	- __應用層 \(C1~C8\)__：負責流程順序與業務情境串接。
	- __領域層 \(Domain Services, Aggregates/Entities\)__：內聚核心邏輯與狀態管理。
	- __底層基礎 \(Prompt Toolbox\)__：實作基本提示詞功能模組。
- __LLM 推理策略__：
	- 結合 Progressive\-Hint Prompting \(PHP\)、Hint\-before\-Solving Prompting \(HSP\)、Hypothesis Testing Prompting、Graph of Thought \(GoT\) 等策略，確保模型具備可迭代性、反向驗證與複雜路徑思維。
- __SOLID 原則__：
	- 完全落實單一職責、開放封閉、里氏替換、介面隔離、依賴反轉，降低耦合度，提升可維護性與可測試性。

__開發建議__

1. __持續整合與持續部署 \(CI/CD\)__：
	- 搭配自動化測試框架，確保每層邏輯變更能被快速驗證，減少回歸錯誤。
2. __介面抽象化__：
	- 在 Domain Services 實現介面抽象，讓底層提示策略可隨時替換或升級，無需影響上層。
3. __模組化設計__：
	- 各層級模組化設計，便於獨立開發與維護，提升開發效率。
4. __詳細文檔與範例__：
	- 提供豐富的技術文檔與實作範例，協助開發人員快速上手並遵循設計原則。
5. __持續優化提示策略__：
	- 根據實際應用需求，持續優化和擴展提示策略，提升 LLM 推理效果與應用覆蓋範圍。

通過以上設計與實作，系統將具備高度的可維護性、可擴展性與靈活性，能夠有效支援各類複雜的 LLM 推理需求。

