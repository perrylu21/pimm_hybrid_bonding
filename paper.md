# 論文撰寫工作流程（paper.md）

> 題目（暫定）：**Physics-Informed Learning for Die-Shift Prediction in Data-Scarce Die-to-Wafer Hybrid Bonding**
> 目標：arXiv（cs.CE 或 physics.app-ph，cross-list cs.LG），之後投 IEEE TCPMT

---

## 1. 工具分工

| 工作 | 用什麼 | 原因 |
|---|---|---|
| 寫初稿（Method、Results、Experiments） | **VS Code + Claude Code** | 可以直接讀 CSV、程式和 `main.tex`，數字與實作一致；能編譯 LaTeX；有 git diff |
| 產生數字巨集、修 LaTeX 編譯錯誤 | **VS Code + Claude Code** | 跟程式綁在一起 |
| 腦力激盪、題目與賣點定位、找文獻 | **網頁版 Claude** | 有網頁搜尋，適合長篇討論 |
| 審稿（每節寫完後） | **網頁版 Claude**（開新對話） | 沒看過寫作過程，比較客觀 |
| 英文潤飾 | 網頁版 Claude（最後再做） | 定稿後一次潤飾 |

原則：**跟程式和數據綁在一起的工作用 VS Code，跟想法和文字有關的用網頁版。**

---

## 2. 開始前的準備（一次性）

### 2.1 在專案根目錄建立 `CLAUDE.md`，寫入以下寫論文規則

```markdown
## Paper-writing rules
- Every number in the paper must come from results/benchmark/*.csv (via docs/paper/numbers.tex). Never estimate or round by hand.
- Cite only entries already in docs/paper/refs.bib. If a new reference is needed, ask me first; never invent a citation.
- Write one section per request. Do not rewrite sections I did not ask about.
- Report the PINN result honestly: the soft physics loss gives no measurable gain (ablation at N=64). Do not spin it.
- Claims about real fabs are out of scope: all validation is against the 3D transient FEM (state this, do not hide it).
- Figures are produced by hbsim/paperfigs.py; change the code, never edit figures by hand.
- After every edit, compile docs/paper/main.tex with latexmk and fix errors.
```

### 2.2 讓數字自動產生（`numbers.tex`）

請 Claude Code 寫一個 `scripts/make_numbers.py`：讀 `results/benchmark/*.csv`，輸出 `docs/paper/numbers.tex`，例如：

```latex
\newcommand{\RzeroKL}{99.1}        % R0 membrane+bending captured [%]
\newcommand{\RzeroGeneric}{96.2}
\newcommand{\RzeroMembrane}{45.6}
\newcommand{\NdmaxOurs}{64}         % FEM runs to reach R2(d_max) >= 0.95
\newcommand{\NdmaxBaseline}{512}
\newcommand{\EtwoOursFour}{2.5}     % nm, 4 marks, prior 256 runs, 2 nm noise
\newcommand{\EtwoAffineFour}{8.1}
```

在 `main.tex` 開頭 `\input{numbers}`，正文只用巨集。這樣重跑實驗後，論文數字會自動跟著更新。

### 2.3 確認編譯環境

- 安裝 MacTeX（或 BasicTeX + `latexmk`）
- `cd docs/paper && latexmk -pdf main.tex` 能成功編譯
- 第一次編譯後，請 Claude Code 修掉所有 warning（引用、圖檔路徑）

---

## 3. 寫作順序（一節一節來）

| 順序 | 章節 | 主要素材 | 完成標準 |
|---|---|---|---|
| 1 | **Method**（簡化物理、兩種實作、稀疏 mark 重建） | `hbsim/physbasis.py`、`hbsim/pinn.py`、`docs/superpowers/specs/` | 每條方程式都對得到程式 |
| 2 | **High-fidelity reference model**（3D FEM） | `hbsim/physics.py`、README 的參數表、網格收斂 | 16 個參數的表、0.2% 收斂 |
| 3 | **Experiments**（R0、E1–E4 的設定） | `hbsim/benchmark.py` | 讀者能照著重現 |
| 4 | **Results** | `results/benchmark/*.csv`、`figures/` | 每張圖一段解讀，數字全用巨集 |
| 5 | **Discussion & Limitations** | E3、PINN ablation | 誠實交代沒有實測資料、PINN 為何沒贏 |
| 6 | **Introduction** | 前面全部 | 貢獻條列與結果一致 |
| 7 | **Related work** | `refs.bib` + 網頁版找的新文獻 | 每篇都親自核對過 |
| 8 | **Abstract** | 最後寫 | 只放最強的 3–4 個數字 |

每節的流程：

1. 在 VS Code 請 Claude Code 寫該節 → 看 diff → 自己改寫關鍵句
2. `latexmk` 編譯 → 匯出 PDF
3. 把該節 PDF 丟給網頁版 Claude 審稿（見第 4 節）
4. 根據審稿意見回 VS Code 修改 → git commit

---

## 4. 審稿用的提示詞（網頁版，開新對話）

```text
You are a strict reviewer for IEEE Transactions on Components, Packaging and Manufacturing Technology.
Review the attached section of a methodology paper on die-shift prediction in D2W hybrid bonding.
Focus on:
1. Is the lack of experimental data stated clearly and handled credibly (no inverse crime, model-form gap quantified)?
2. Are the baseline comparisons fair (same data, same budget, analytic prior given to baselines too)?
3. Is every claim supported by a number, figure or table?
4. Would the PINN negative result raise doubts about the method name or framing?
5. Unclear notation, missing definitions, overclaiming words ("novel", "first", "significantly").
List issues by severity (major / minor) with a concrete fix for each.
```

---

## 5. 論文要講的故事（核心數字，來源：`results/benchmark/`）

| 實驗 | 結果 | 檔案 |
|---|---|---|
| **R0** 物理表示能力 | Membrane + bending 23 維表示 **99.1%** 非 affine overlay；一般多項式 24 維 96.2%；只有 membrane 45.6% → **非均勻 die shift 由彎曲主導** | `r0_representation.csv`、`fig_r0_representation` |
| **E1** 資料效率 | d_max R² ≥ 0.95：本法 **64 runs**，最佳 baseline（解析式 + GP）**512 runs**（少 8 倍）；N=32 時 0.925 vs 0.791 | `e1_learning_curve.csv`、`fig_e1_learning_curves` |
| **E2** 稀疏 mark 重建 | 4 個 mark：本法 **2.5 nm**（256-run 先驗）/ 5.0 nm（64-run）vs 業界 affine **8.1 nm** | `e2_sparse_marks.csv`、`fig_e2_sparse_marks` |
| **E3** 穩健性 | 外插 dT_head > 40：d_max R² **0.96** vs 0.82–0.83；訓練雜訊 5 nm：0.99 vs ≤ 0.90 | `e3_robustness.csv`、`fig_e3_robustness` |
| **E4** Ablation | 拿掉解析先驗在小 N 明顯變差；**PINN 的 physics loss 無效益**（N=64：92 vs 88 nm） | `pinn_ablations/`、`fig_e4_ablations` |

要誠實寫的兩點：

1. **PINN 沒有贏。** 原因：eigenstrain 是自由的 closure 場時，PDE 殘差只約束形狀；真正帶來效益的是解析先驗和低維物理基底，而 hard-constrained 版本已經精確編碼了它們。
2. **m_x 不是優勢。** 本法和「解析式 + GP」打平；贏在 d_max 和非均勻分量。

---

## 6. 弱點（沒有實測資料）的處理方式

- 驗證對象是**物理更完整的獨立模型**（3D 暫態 FEM，含 bond-wave locking），簡化模型是 2D，**不是 inverse crime**
- 模型落差已量化（R0：殘差中位數 0.58 nm）
- 量級與公開數據一致（imec：die corner scaling 80–150 nm，Bex 2024）
- E3 模擬真實資料的不完美：雜訊、外插
- 程式、資料、Docker image digest 全部公開
- Limitations 一節寫出「實機驗證流程」：每顆 die 量幾個 mark、量幾片 wafer、要記錄哪些製程參數

---

## 7. 投稿前檢查清單

- [ ] `main.tex` 裡沒有 `\todo`
- [ ] 所有數字都來自 `numbers.tex`
- [ ] `refs.bib` 每一篇都用 DOI / arXiv 編號親自核對過（AI 可能編造引用）
- [ ] 關鍵論述（物理發現、PINN 分析）用自己的話重寫過
- [ ] 圖由 `python cli.py paper-figs --bench results/benchmark` 重新產生，與 CSV 一致
- [ ] `python -m pytest` 全部通過
- [ ] 程式網址（GitHub）已填入 Code availability
- [ ] 作者、單位、致謝
- [ ] arXiv 分類：cs.CE 或 physics.app-ph，cross-list cs.LG

---

## 8. 重要檔案位置

| 用途 | 路徑 |
|---|---|
| 論文骨架 | `docs/paper/main.tex`、`docs/paper/refs.bib` |
| 實驗結果 | `results/benchmark/*.csv`（PINN：`pinn/`、`pinn_ablations/`） |
| 論文圖表 | `results/benchmark/figures/`（PDF + PNG、`table_e1_summary.tex`） |
| 方法程式 | `hbsim/physbasis.py`、`hbsim/pinn.py` |
| 實驗程式 | `hbsim/benchmark.py`、`hbsim/paperfigs.py` |
| 重跑實驗 | README「Physics-informed models and data-scarce benchmarks」一節 |
