# refs_notes — P1 文献核实记录

> 阶段：P1 文献｜日期：2026-08-07｜目的：核实基座规格 + FME 竞争方法划界叙事

## 1. Alin (2026) — 基座方法规格（已锁定）

**正式方法名**：EnsPSRWSIMPLS（Ensemble Penalized Smoothed Robust Weighted SIMPLS），基学习器 PSRWSIMPLS。
**来源**：Alin, A. (2026). Ensemble robust SIMPLS with block-penalized smoothing for scalar-on-function regression. *Chemometrics and Intelligent Laboratory Systems*, 269, 105626. DOI 10.1016/j.chemolab.2025.105626.
（期刊页 18 页；AVESİS / core.ac.uk / NSTL 均有书目与摘要记录）

**四要素（逐条对照 plan.md §1）**：
1. **Blockwise B-spline 平滑**：对高维离散函数预测子做块对角 B-spline 基平滑；"supports multiple functional predictors through a block-diagonal construction" → **分块 = 按函数预测子**（与 plan.md §1 一致）。**核验结论：Tecator/S5 仅单函数预测子 → 退化为单 B-spline 块（惩罚 B-spline）**，多预测子块对角为自然推广（审稿意见 12 成立）。
2. **Robust weighted SIMPLS（RWSIMPLS）**：摘要称 "disparity-based reweighted SIMPLS" 获取潜方向/潜得分，抗异常值。"RWSIMPLS" 名称在摘要中即为 robust weighted SIMPLS。
3. **潜空间惩罚回归**：二次惩罚恰为系数函数的**块对角粗糙度惩罚** → 光滑可解释 β(t)。与 Reiss & Ogden (2007) approach 2 同源（样条平滑预测子 + 潜得分回归惩罚化，无基表示假设）。
4. **Bootstrap 集成**：classical/sufficient bagging 两版 + **mean/median 聚合** + **OOB / .632+ 诊断**。
5. **调参**：k 折 CV 同时调潜成分数 A 与粗糙度惩罚。

**对照对象（原文）**：PSRWSIMPLS（基学习器）、robust RFPLS、robust RWSIMPLS、非 PLS 惩罚函数回归（pfr）。真实光谱案例 + 模拟；在共线性与污染下稳健/光滑/稳定兼顾。

**与相关方法对比定位**：adapts Reiss & Ogden (2007) approach 2；区别于 Lin & Zi（EEMD+Lasso，无 PLSR/无粗糙度/集成在分解层而非 β 聚合）、Sun et al.（部分函数线性回归 + GP 先验，无 PLSR）。

**SSRN 全文**：审稿人给出 abstract_id 5593333 的 27 页预印本；本环境 WebFetch 返回 403（反爬）。**因贡献已降级为 Inspired by，摘要级方法细节已足够锁定基座规格**；如需逐行细节可在人工浏览器打开 SSRN 读取。

## 2. FME / iFME — 竞争方法（P1 已精读，用于划界叙事）

**FME**：Chamroukhi, F., Pham, N. T., Hoang, V. H., & McLachlan, G. J. (2024). Functional mixtures-of-experts. *Statistics and Computing*, 34(3), 98. DOI 10.1007/s11222-023-10379-0.
**iFME**：同文的可解释稀疏版（对专家系数 β(t) 与门控函数 α(t) 的导数加 Lasso 稀疏正则，EM-Lasso 算法）。

**模型形式（关键）**：
- 专家：Y_i = β_{z,0} + ∫X_i(t)β_z(t)dt + ε_i（每个专家一个函数型回归）
- 门控：**函数型多分类 logistic（softmax）**：π_z(X) = exp{α_{z,0} + ∫X(t)α_z(t)dt} / Σ_{z'=1..K} exp{...}
- **多专家概率混合** ŷ = Σ_z π_z(x)ŷ_z(x)；EM / EM-Lasso 估计
- 应用：Canadian weather、DTI 多发性硬化数据

**与 AHFE 的边界（写进 §2.4 / 知识库）**：
- FME = 在**多个完整专家**之间做**概率混合**（softmax 门控），每个专家都是独立完整的函数型回归。
- AHFE = **一个线性稳健基座 + 一个残差校正器**的残差加性混合，sigmoid-link 门控控制**残差校正强度 w(x)**，非多专家概率分配。
- 数学对照：FME ŷ=Σ_k π_k ŷ_k；AHFE ŷ=(1−w)ŷ_A + w·ŷ_B^full（ŷ_B^full=ŷ_A+r̂_B）。**门控参数空间维度、语义、专家结构均不同**——AHFE 不声称"首次做函数型门控"，而是"残差校正强度门控"。

## 3. 数据集可获取性（P1 核查）
- Tecator：`skfda.datasets.fetch_tecator` 返回 X=FDataGrid（`.data_matrix` 形状 (215,100,1) → squeeze→(215,100)），y=(215,3) [fat/moisture/protein]。**已验证可获取**。
- Gasoline octane（S4 补充）：fda.usc 公开 CSV / skfda 内置，备用。

## 4. 待用户侧动作（人工，非阻塞）
- 如需 Alin 2026 逐行算法细节：在浏览器打开 SSRN 5593333 读取全文（本环境 403）。

## 结论
- 基座规格**已锁定**（EnsPSRWSIMPLS 四要素与 plan.md §1 一致）。
- FME 竞争叙事**已锁定**（多专家概率混合 vs 残差校正强度门控，见 §2.4）。
