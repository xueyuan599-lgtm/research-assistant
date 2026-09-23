# -*- coding: utf-8 -*-
"""生成 AHFE 研究约 4000 字论文级 PDF（reportlab，内置 CJK 字体）。"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, HRFlowable)
from reportlab.lib.styles import ParagraphStyle

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
FONT = 'STSong-Light'

body = ParagraphStyle('body', fontName=FONT, fontSize=10.5, leading=17,
                      spaceAfter=6, firstLineIndent=21)
h2 = ParagraphStyle('h2', fontName=FONT, fontSize=13.5, leading=18,
                    spaceBefore=10, spaceAfter=6, textColor=colors.HexColor('#2a5aa0'))
h3 = ParagraphStyle('h3', fontName=FONT, fontSize=11.5, leading=16,
                    spaceBefore=6, spaceAfter=4, textColor=colors.HexColor('#444444'))
title = ParagraphStyle('title', fontName=FONT, fontSize=19, leading=25,
                       spaceAfter=6, textColor=colors.HexColor('#10233f'), alignment=1)
sub = ParagraphStyle('sub', fontName=FONT, fontSize=10.5, leading=15,
                     spaceAfter=14, textColor=colors.grey, alignment=1)
cap = ParagraphStyle('cap', fontName=FONT, fontSize=9, leading=12,
                     spaceAfter=8, textColor=colors.HexColor('#555555'), alignment=1)
ref = ParagraphStyle('ref', fontName=FONT, fontSize=9, leading=12,
                     spaceAfter=3, textColor=colors.HexColor('#333333'))


def P(t): return Paragraph(t, body)
def H(t): return Paragraph(t, h2)
def H3(t): return Paragraph(t, h3)
def C(t): return Paragraph(t, cap)
def R(t): return Paragraph(t, ref)


HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, 'outputs', 'figures')

story = []
story.append(Paragraph('函数型数据集成回归：残差校正混合框架（AHFE）', title))
story.append(Paragraph('AHFE — Adaptive Hybrid Functional Ensemble：可解释线性基座上的残差梯度提升校正与门控', sub))

# ===== 摘要 =====
story.append(H('摘要'))
story.append(P('针对函数型数据（scalar-on-function，简称 SoFR）回归中线性潜空间模型对非线性功能效应系统性欠拟合的问题，'
               '本文提出一种自适应混合函数型集成方法 AHFE。方法以稳健加权 SIMPLS 集成（受 Alin 等 2026 年工作启发）为可解释线性基座，'
               '在残差上训练梯度提升树（GBDT）作为非线性校正器，并以 sigmoid 链接门控按预测时可计算的函数型特征调制校正强度，'
               '得到融合式 $\\hat y_{AHFE} = \\hat y_A + w(x)\\hat r_B$。门控在交叉拟合（cross-fitting）生成的样本外（OOF）预测上训练，'
               '全程不接触测试集目标，从而杜绝泄漏与过拟合；同时给出最优门控的 oracle 命题，作为方法学的理论锚。'
               '在合成非线性、Tecator 真实光谱以及两类污染场景的确证基准（独立种子）上，'
               'AHFE 相对 SIMPLS 基座在非线性任务上提升 $\\Delta R^2 \\approx +0.39$，配对 bootstrap 的 95% 置信区间排除 0；'
               '线性数据不致损，函数型离群下保持稳健。核心结论经受诚实检验：逐样本自适应门控与常数校正 $\\gamma$ 无显著差异，'
               '因此本方法的价值定位于"残差校正混合"整体增益，而非逐样本门控的自适应机制。'))

# ===== 1 引言 =====
story.append(H('1 引言与问题背景'))
story.append(P('函数型数据以整条曲线为基本观测单元，广泛存在于近红外光谱、工业过程信号、生理轨迹与气象监测等场景。'
               'scalar-on-function 回归旨在建立函数预测子 $X(t)$ 与标量响应 $Y$ 之间的映射，其经典形式为 '
               '$Y=\\alpha+\\int_{\\mathcal{T}} X(t)\\beta(t)dt+\\varepsilon$。由于曲线在离散网格上观测，'
               '直接对每个网格点做高维回归会遇到维数灾难与多重共线性。偏最小二乘（PLS）及其高效变体 SIMPLS（de Jong, 1993）'
               '通过将曲线投影到低维线性潜空间来估计系数函数 $\\beta(t)$，兼顾光滑可解释与计算高效，'
               '已成为化学计量学、过程分析等领域的事实标准工具。Alin 等（2026）进一步引入分块 B 样条平滑、'
               '稳健加权与自助聚合，构建集成稳健 SIMPLS，提升了系数光滑性与对异常值的稳健性。'))
story.append(P('然而，PLS 潜空间本质上是 $X$ 的线性投影，其假设函数效应在潜变量空间中可线性表示。'
               '当响应对曲线呈非线性功能效应时——例如中心化二次交互 $\\lambda(q^2-\\mathbb{E}q^2)$，'
               '其中 $q=\\int X\\varphi dt$，又或阈值、分段、高阶交互等结构——线性潜空间将系统性欠拟合，'
               '表现为残差中残留可解释的非线性信号。函数型领域的非线性替代方案各有代价：'
               '深度函数型网络与纯树集成虽拟合能力强，却失去可解释的系数函数 $\\beta(t)$；'
               '广义函数型线性模型需预指定链接函数，灵活性受限；函数型混合专家（FME）需在多个完整专家间做概率混合，'
               '参数规模与识别性负担较大。本文的目标是在保留线性基座可解释性的前提下，'
               '以数据驱动方式补上被潜空间遗漏的非线性结构，并对增益做严格且诚实的统计检验。'))

# ===== 2 方法基础 =====
story.append(H('2 方法基础'))
story.append(H3('2.1 稳健 SIMPLS 集成（线性基座）'))
story.append(P('线性组件采用稳健加权 SIMPLS（RWSIMPLS）。其核心是以迭代重加权估计抑制响应污染：'
               '先用 MAD 尺度标准化残差，再以 Tukey bisquare 权重对离群样本降权，迭代至权重稳定。'
               '加权后的 SIMPLS 对成分数 $A$ 个潜变量分解，得到光滑、可解释的系数函数 $\\hat\\beta(t)$。'
               '为强化稳健性并刻画不确定性，并行拟合 $B$ 个自助样本并对预测取平均，构成集成基座 M3，'
               '同时作为 AHFE 的组件 A。该组件提供低方差、稳健、可解释的线性预测 $\\hat y_A$。'))
story.append(H3('2.2 残差梯度提升（非线性校正器）'))
story.append(P('非线性组件 B 在残差 $r=y-\\hat y_A$ 上训练标准梯度提升树（本研究采用 XGBoost）。'
               '输入特征为从曲线导出的函数型特征：FPCA 评分、B 样条系数与一阶导数幅值统计。'
               '树模型自带特征选择，能刻画 PLS 潜空间（线性投影）遗漏的非线性功能结构，输出 $\\hat r_B(x)$。'
               '将校正器置于残差上而非响应本身，保证了与线性基座的解耦：基座已解释的部分不会被重复拟合。'))

# ===== 3 AHFE =====
story.append(H('3 AHFE 方法'))
story.append(H3('3.1 残差校正门控融合'))
story.append(P('AHFE 以门控权重融合线性基座与残差校正器：'
               '$$\\hat y_{AHFE}(x)=\\hat y_A(x)+w(x)\\hat r_B(x),\\quad w(x)=\\sigma(\\alpha+z^\\top\\theta)\\in[0,1].$$'
               '门控特征 $z=[\\text{FPCA 评分},|\\hat r_B|,Q\\text{-残差},\\text{leverage}]$ 均为预测时可计算量，'
               '不含真实残差幅值——预测新样本时 $y$ 尚未观测。语义上，$w\\to0$ 处信任线性基座、保持可解释，'
               '$w\\to1$ 处启用非线性校正。与函数型混合专家不同，AHFE 只含一个线性基座与一个校正器，'
               '$w$ 调制的是"修正强度"而非多专家概率分配；等价地可写为 '
               '$\\hat y=(1-w)\\hat y_A+w(\\hat y_A+\\hat r_B)$，与经典混合专家形式同构但只有一个非线性"专家"。'))
story.append(H3('3.2 交叉拟合门控训练（防泄漏）'))
story.append(P('门控若在训练集上直接学习会接触目标 $y$，引起泄漏与过拟合，使门控的测试增益虚高。'
               '为此在外层训练折内再做内层 $K$ 折交叉拟合：每个内折在其余数据上拟合组件 A、B 与特征投影器，'
               '对保留折生成真正的 OOF 预测 $(\\hat y_A^{OOF},\\hat r_B^{OOF},z^{OOF})$；'
               '随后在 OOF 数据上最小化 $\\sum_i(y_i-\\hat y_{A,i}-w(z_i)\\hat r_{B,i})^2+\\lambda_g\\|\\theta\\|^2$ 训练门控。'
               '训练完成后，A、B 与特征投影器在整个外层训练集上重训，并一次性预测测试集。'
               '门控参数仅在 OOF 上学习，测试折的 $y$ 从未参与，从而保证门控训练无泄漏。'))
story.append(H3('3.3 Oracle 门控理论'))
story.append(P('记 $e_A=Y-\\hat y_A$、$q=\\hat r_B$，AHFE 的条件风险 $R(w|z)=\\mathbb{E}[(e_A-wq)^2|z]$ '
               '对 $w$ 求最优，得 oracle 门控 $w^*(z)=\\Pi_{[0,1]}\\mathbb{E}[e_A q|z]/\\mathbb{E}[q^2|z]$，'
               '其中 $\\Pi_{[0,1]}$ 为截断算子。AHFE-fixed 是它的特例：$\\gamma^*=\\Pi_{[0,1]}\\mathbb{E}[e_A q]/\\mathbb{E}[q^2]$。'
               '由此形成方法学链条：SIMPLS $\\rightarrow$ 固定残差校正 $\\rightarrow$ 条件最优残差校正。'
               '该命题提供理论语义：若残差模型在某类样本上确实解释了基座的系统残差，门控自动抬高；'
               '若树组件只是在拟合噪声，$w^*(z)\\approx0$，AHFE 退化为 SIMPLS，自带防劣化机制。'))

# ===== 4 实验 =====
story.append(H('4 实验与结果'))
story.append(P('确证基准在方案冻结后以全新种子 {100,101,102,103,104} 与 5 折交叉验证独立运行，'
               '所得结果未回用于任何模型修改，以避免"看结果改模型"的自适应过拟合。'
               '方法集包括：M1 函数线性模型（FLM）、M2 函数特征加 GBDT（与 AHFE 组件 B 使用完全相同的特征库，'
               '保证公平对照）、M3 SIMPLS 集成（基座）、AHFE-fixed 与 AHFE-adaptive。'
               '场景覆盖 S1 Tecator 脂肪含量（真实光谱）、S5 合成非线性（中心化二次项）、'
               'S5b 线性阳性对照、S6a Y 离群污染与 S6b 函数型离群。'))
story.append(Spacer(1, 4))
data = [
    ['场景', 'M1 FLM', 'M2 特征+GBDT', 'M3 SIMPLS', 'AHFE-fixed', 'AHFE-adaptive'],
    ['S1 Tecator', '0.920', '0.940', '0.882', '0.974', '0.970'],
    ['S5 非线性', '0.420', '0.796', '0.424', '0.823', '0.815'],
    ['S5b 线性', '0.892', '0.859', '0.894', '0.893', '0.891'],
    ['S6b 函数型离群', '0.916', '0.932', '0.833', '0.971', '0.970'],
    ['S6a Y 离群', '0.259', '-0.030', '0.198', '0.199', '0.186'],
]
tbl = Table(data, colWidths=[3.4*cm, 2.3*cm, 2.6*cm, 2.3*cm, 2.3*cm, 2.5*cm])
tbl.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5aa0')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, -1), FONT),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#eef3fa')]),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
]))
story.append(tbl)
story.append(C('表 1 各场景种子均值 R²。AHFE 在非线性与真实数据上显著高于 SIMPLS 基座，线性下持平。'))
story.append(P('核心结果：非线性 S5 上 AHFE-fixed 达到 0.823，相对 M3 基座 0.424 提升约 +0.39；'
               '配对逐样本 bootstrap 给出 $\\Delta\\text{RMSE}=0.162$，95% 置信区间 [0.119, 0.209] 排除 0，'
               '统计上显著。真实 Tecator 上 AHFE 0.974 相对 M3 0.882 提升 +0.09。'
               '线性 S5b 上 AHFE 与 M3 持平（0.893），说明门控在无非线性时无需强制开启，混合结构不致损。'
               'S6b 函数型离群下 AHFE 0.971 相对 M3 0.833 保持稳健优势；S6a 强 Y 离群下所有方法大幅退化，'
               'AHFE 无增益，此局限如实报告。'))
story.append(Image(os.path.join(FIG, 'fig3_r2_scenarios.png'), width=15.5*cm, height=7.6*cm))
story.append(C('图 1 各场景 R² 分组对比（确证种子均值）。'))
story.append(H3('4.1 门控机制与可解释性'))
story.append(P('门控机制如实报告：自适应门控与固定门控在各场景几乎相同（S5 为 0.815 对 0.823），'
               '可行的 oracle 亦接近固定门控；逐样本真实非线性强度与门控权重的 Spearman 相关仅 +0.036 且不显著，'
               'NL 分位数对平均权重的曲线非单调。因此本研究不宣称逐样本门控优于常数校正。'
               '线性组件的系数函数与真值相关系数达 0.937，说明非线性校正并未污染线性解释。'
               '对校正器的 SHAP 分析显示，非线性残差主要由 FPCA3 投影与一阶导数幅值驱动，B 样条系数贡献较小。'))
story.append(Image(os.path.join(FIG, 'fig5_gate_mechanism.png'), width=10*cm, height=6.4*cm))
story.append(C('图 2 门控机制：NL 分位数对平均门控权重（如实报告其非单调，Spearman≈0）。'))

# ===== 5 对比 =====
story.append(H('5 与其他方法对比'))
story.append(P('实证上，AHFE 相对基座 M3 的增益最大（非线性 +0.39），相对使用相同特征库的 M2 也有约 +0.03 的优势，'
               '说明增益源于"残差校正线性基座"的结构设计而非特征工程；相对线性基线 FLM 在非线性下提升 +0.40，'
               '在线性下持平。文献定位上，AHFE 处于"线性可解释"（FLM、SIMPLS 家族）与"非线性拟合"'
               '（树集成、深度函数型网络）两类方法之间：它保留系数函数 $\\beta(t)$ 的可解释性，'
               '同时以数据驱动的残差校正补充非线性，数据需求与算力成本适中，并继承稳健 SIMPLS 的抗污染能力。'))
story.append(P('与函数型混合专家（FME）相比，AHFE 表达力更弱但更简单、可解释，且无需在多个专家间做概率混合，'
               '规避了混合模型常见的识别性负担；与函数型高斯过程相比，AHFE 计算代价低一个量级且协方差设计简单；'
               '与深度函数型网络相比，AHFE 数据需求更小并保有线性解释。需要指出的是，'
               '上述文献层面的对比多为概念与性质定位，FME、函数型高斯过程等并未在本基准中实际运行，'
               '若作为正式竞争方法尚需补充硬性实证，这是本研究可继续加强之处。'))

# ===== 6 创新与诚实评价 =====
story.append(H('6 创新点与诚实评价'))
story.append(P('本研究的核心创新是残差校正混合结构：在可解释线性基座上叠加残差梯度提升校正，'
               '以"保解释、补非线性"的折中方式提升函数型回归的非线性拟合能力，并在独立确证基准上验证了其显著增益。'
               '交叉拟合门控与 oracle 门控命题提供了方法学上的严谨性与理论锚，'
               '但前者属于对半参数机器学习中交叉拟合良好实践的复用，后者为条件最小二乘的初等最优条件，'
               '二者均不构成深层原理创新。就方法新颖度而言，AHFE 属于"已有组件的合理组合加严格实证"，'
               '更贴合方法应用类研究，而非机制性算法理论创新。'))
story.append(P('诚实边界必须声明：原设想的逐样本自适应门控优势在合成与真实数据上均未兑现——'
               '自适应与常数校正几乎等价，门控未能干净追踪逐样本非线性位置。'
               '这一负面结果被如实报告并写入知识库，说明本方法的价值定位是"残差校正混合"整体，'
               '而非门控的自适应机制。研究仍有两点局限：其一，非线性结构测试仅含中心化二次一种，覆盖面偏窄；'
               '其二，系统实证仅运行 XGBoost 一种梯度提升实现。补足多类非线性 DGP 与多基学习器（LightGBM、CatBoost），'
               '并实跑 FME 与函数型高斯过程做硬对比，将进一步提高结论的说服力与普适性。'))

# ===== 7 结论 =====
story.append(H('7 结论'))
story.append(P('本文提出并严格验证了残差校正混合函数型集成 AHFE：以稳健 SIMPLS 为可解释线性基座、'
               '残差梯度提升补非线性、交叉拟合门控控制修正强度。在非线性 scalar-on-function 任务上，'
               'AHFE 相对基座提升约 $\\Delta R^2=+0.39$，置信区间排除 0，线性数据无伤，函数型离群下稳健。'
               '方法贡献可信且诚实，创新核心是"可解释基座 + 残差校正"的折中结构；'
               '自适应门控的增益经实证否定并如实披露。作为方法应用与实证研究，'
               'AHFE 定位明确、可复现，可为函数型非线性回归的后续应用提供参考。'))

story.append(Spacer(1, 6))
story.append(HRFlowable(width='100%', thickness=0.6, color=colors.HexColor('#cccccc')))
story.append(H('参考文献'))
for i, ref in enumerate([
    'Alin, A., et al. (2026). Ensemble robust SIMPLS with block-penalized smoothing for scalar-on-function regression. Chemometrics and Intelligent Laboratory Systems, 269.',
    'Chamroukhi, F., et al. (2024). Functional mixtures-of-experts for scalar-on-function regression. Statistics and Computing, 34(3):98.',
    'de Jong, S. (1993). SIMPLS: An alternative approach to partial least squares regression. Chemometrics and Intelligent Laboratory Systems, 18(3):251-263.',
    'Hubert, M., & Vanden Branden, K. (2003). Robust methods for partial least squares regression. Journal of Chemometrics, 17(10):537-549.',
    'Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. Statistical Science, 11(2):89-121.',
    'Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. ACM SIGKDD.',
    'Ramsay, J. O., & Silverman, B. W. (2005). Functional Data Analysis (2nd ed.). Springer.',
    'Jacobs, R. A., et al. (1991). Adaptive mixtures of local experts. Neural Computation, 3(1):79-87.',
], 1):
    story.append(R(f'[{i}] {ref}'))

out = os.path.join(HERE, 'ahfe_report.pdf')
doc = SimpleDocTemplate(out, pagesize=A4,
                        leftMargin=2.2*cm, rightMargin=2.2*cm,
                        topMargin=2.0*cm, bottomMargin=2.0*cm,
                        title='函数型数据集成回归：残差校正混合框架（AHFE）')
doc.build(story)
print('Wrote', out)
