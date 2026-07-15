# Titanic V4 — 激进特征工程方案

> 设计目标：平衡特征丰富度与泛化能力，刷榜 LB 0.78-0.80+
> 历史教训：V1（54特征 + TargetEnc）严重过拟合（CV 0.853 -> LB 0.775）；V3（11特征）反而更优（CV 0.834 -> LB 0.773）
> 核心理念：**激进在方法论严格性，不在特征数量**。每个特征必须通过 5 阶段筛选，宁缺毋滥。

---

## 0. 前置原则：为什么这次不同

| V1 失败原因 | V4 对策 |
|---|---|
| 54 特征无筛选，全扔进模型 | 5 阶段筛选：MI -> Spearman -> Boruta -> Stability -> Permutation |
| Target Encoding 用全训练集均值（严重泄漏） | 所有统计编码必须在 CV fold 内计算，加平滑先验 |
| 稀有类别未合并 | 频次 < 5 一律合并为 `Rare` |
| 特征高度冗余（Spearman > 0.9 不处理） | 相关性去重保留 MI 更高的 |
| 交互特征无理论依据，暴力叉乘 | 仅保留文献/EDA 确认有意义的交互，且通过筛选 |

---

## 1. 缺失值处理策略

### 1.1 Age（177 缺失，19.9%）

**推荐：方案 A（分组中位数 + 缺失指示器）**

| 方案 | 方法 | 优点 | 缺点 | 判定 |
|------|------|------|------|------|
| A | Pclass x Sex 分组中位数填补 | 最简单、最稳健、无泄漏风险 | 损失组内年龄分布信息 | **采纳** |
| B | MICE 多重插补（5 imputations, pooled） | 统计上更严谨，保留不确定性 | 891 样本太小，插补模型不稳定；多重数据集增加工程复杂度 | 不采纳 |
| C | RF 回归插补 + 加噪声 | 捕捉非线性关系 | 891 样本上 RF 插补模型本身过拟合；噪声级别难以校准 | 不采纳 |

**实施细节：**
- 按 `Pclass x Sex` 6 个分组计算中位数填入
- 生成 `AgeMissing` 二值指示特征（缺失可能本身有预测意义：三等舱乘客年龄记录不全的比例更高）
- 部分年龄值以 `.5` 结尾（推定值），不特殊处理（信息不足）

**为什么不选 MICE/RF：**
891 样本上插补模型比预测模型更复杂是反直觉的——你在用一个 50 棵树的 RF 去预测 177 个缺失值，然后用另一个 RF 做最终分类。插补误差传播到下游，在小样本上不可控。分组中位数的 bias 高于复杂方法，但 variance 远低于它们——在小样本上，bias-variance tradeoff 偏向低 variance。

### 1.2 Cabin（687 缺失，77.1%）

**策略：HasCabin + Deck(Missing 作为单独类别)**

- `HasCabin`：二值，有/无客舱记录（V3 已验证有效）
- `Deck`：提取 Cabin 首字母，A/B/C/D/E/F/G/T + `M`(Missing)。77% 的 Missing 比例意味着 Deck 的大部分信号来自 "有没有记录" 而非 "具体在哪个甲板"
- 不填补具体客舱号（缺乏信息基础）

### 1.3 Embarked（2 缺失）

填 `S`（Southampton，众数，占比约 72%）。

### 1.4 Fare（test 中 1 缺失）

按 `Pclass` 分组中位数填补（三等舱中位数）。

---

## 2. 特征构造清单（44 候选，目标筛选至 25-35）

### 2.1 A 类：基础特征（5 个）

| # | 特征名 | 类型 | 构造方式 | 说明 |
|---|--------|------|---------|------|
| A1 | Pclass | ordinal | 保留 1/2/3 | 社会经济地位代理变量，与 Survived 强相关 |
| A2 | Sex | binary | male=0, female=1 | 最强单一预测特征（女性存活率 74% vs 男性 19%） |
| A3 | Age | continuous | 分组中位数填补后 StandardScaler | 连续年龄效应（儿童优先上救生艇） |
| A4 | Fare_log | continuous | log1p(Fare) | 极度右偏（max=512 vs median=14），log 变换后接近正态 |
| A5 | Embarked | categorical | OneHot(3): C/Q/S | 登船港口，与 Pclass 相关（Cherbourg 多一等舱） |

**设计说明：**
- Pclass 使用 Ordinal(1/2/3) 而非 OneHot。虽然 OneHot 理论上更灵活（允许非线性效应），但 891 样本上节省 2 维 > 增加一点灵活性。LightGBM/CatBoost 对 ordinal 和 OneHot 几乎无差别，树模型自动学习切分点。
- Age 保留连续值 + StandardScaler。等于同时保留数值和分箱（logistic regression 可通过分箱获得非线性，树模型自动处理）。

### 2.2 B 类：家庭特征（7 个）

| # | 特征名 | 类型 | 构造方式 | 说明 |
|---|--------|------|---------|------|
| B1 | FamilySize | continuous | SibSp + Parch + 1 | 家庭规模，非线性效应：独自一人和超大家庭存活率均低 |
| B2 | IsAlone | binary | FamilySize == 1 | 独行乘客标记（存活率约 30%，低于有家庭者） |
| B3 | FamilySize_bin | categorical | 1 / 2-3 / 4-5 / 6+ | 分箱揭示非线性：1人（约30%）、2-3人（约55%）、4-5人（约40%）、6+人（约16%） |
| B4 | SibSp_bin | categorical | 0 / 1 / 2+ | 0个（约35%）、1个（约55%）、2+个（约25%），非线性 |
| B5 | Parch_bin | categorical | 0 / 1 / 2+ | 0个（约35%）、1个（约55%）、2+个（约42%） |
| B6 | HasSpouse | binary | 已婚女性（Mrs）或 Title=Mrs 且 Parch>0 | 通过 Title 推断婚姻状态，比 SibSp>0 更精确。已婚女性存活率约 79%（救生艇优先家庭） |
| B7 | HasChild | binary | Parch > 0 | 有父母或子女同行。有子女者存活率更高（女性 + 儿童优先） |

**设计说明：**
- `HasSpouse` 不直接用 SibSp > 0（SibSp 含兄弟姐妹，不精确）。判别逻辑：Title=Mrs 即已婚女性；Title=Mr 且 FamilySize>1 且有女性同姓 → 已婚男性。在 891 样本中这个识别准确率足够。
- B3/B4/B5 的分箱阈值基于训练集 survival rate 分布，不引入测试集信息。
- B2/B4/B5 存在结构性相关（FamilySize 由 SibSp+Parch 构成），将在阶段 2 相关性筛选中去重。
- B6/B7 也可从 Title 直接推断，与 C 类特征部分冗余，由筛选器决定取舍。

### 2.3 C 类：Title 特征（6 个）

| # | 特征名 | 类型 | 构造方式 | 说明 |
|---|--------|------|---------|------|
| C1 | Title | categorical | Mr/Mrs/Miss/Master/Dr/Rev/Military/Other | 主类别。Military 合并 Col/Major/Capt；Other 合并其余低频 |
| C2 | Title_Rare | binary | Title 是否属于稀有类别（频次<10） | 稀有头衔者可能特殊（贵族、神职人员等），存活模式不同 |
| C3 | Title_Pclass | interaction | Title × Pclass 交叉编码 | 头等舱 Mrs(存活97%) vs 三等舱 Mrs(存活46%)。拉平后的 Title 掩盖了舱位差异 |
| C4 | Title_Age | interaction | Title × Age_bin 交叉 | Master=男童(<13)，Miss=年轻未婚女性(<30)、年长未婚女性(30+)存活模式不同 |
| C5 | IsMaster | binary | Title == 'Master' | 男童是 Titanic 上除女性外存活率最高的群体（~58%，接近三等舱女性） |
| C6 | TitleGroup | categorical | 女性尊称(Mrs/Miss)/男性尊称(Mr)/儿童(Master)/职业(Dr/Rev)/军事/其他 | 更粗粒度的 Title 分组，作为 C1 的稳健替代（合并后减少稀疏性） |

**Title 映射表：**

| 原始 Title | 归类 | 训练集频次 | 存活率 |
|---|---|---|---|
| Mr | Mr | 517 | 15.7% |
| Miss | Miss | 182 | 69.7% |
| Mrs | Mrs | 125 | 79.2% |
| Master | Master | 40 | 57.5% |
| Dr | Dr | 7 | 42.9% |
| Rev | Rev | 6 | 0.0% |
| Col | Military | 2 | 50.0% |
| Major | Military | 2 | 50.0% |
| Capt | Military | 1 | 0.0% |
| Mlle | Miss | 2 | 100% |
| Ms | Miss | 1 | 100% |
| Mme | Mrs | 1 | 100% |
| Lady/Countess/Sir/Jonkheer/Don/Dona | Other | 5 | — |

**设计说明：**
- C1 是 V3 已验证有效的特征（LabelEncoded）。
- C3 是核心交互：Title 和 Pclass 的联合效应远大于单独 Title。一个三等舱 Mrs 的处境和一个头等舱 Mrs 完全不同。但注意，这是交互而非独立特征——筛选器可能更倾向保留单独 Title 和 Pclass，让模型自己学习交互。**C3 作为候选，在筛选阶段与 A1+C1 竞争。**
- C5（IsMaster）是二值即够用的特征：Master 是信号，非 Master 已由 Sex+Title 覆盖。
- C6 是为 OneHot 友好的粗分组。

### 2.4 D 类：Deck/Cabin 特征（5 个）

| # | 特征名 | 类型 | 构造方式 | 说明 |
|---|--------|------|---------|------|
| D1 | Deck | categorical | Cabin 首字母：A/B/C/D/E/F/G/T + M(Missing) | 甲板级别。A=顶层(最贵)，G=底层。T 仅有1人（测试集可能无）。M=缺失(687人) |
| D2 | HasCabin | binary | Cabin.notna() | V3 已验证有效。有客舱记录者存活率约 67%，无记录约 30% |
| D3 | CabinNum | categorical | Cabin 数字部分分箱：0-30/31-60/61-100/100+/Missing | 同甲板不同区域。数字越小越靠近船头（更危险，先沉没）。信号较弱，候选 |
| D4 | CabinShared | binary | 同一 Cabin 号出现多次 | 多人同客舱 = 家庭/团体出行。可能已由 FamilySize 覆盖 |
| D5 | Deck_Pclass | interaction | Deck × Pclass | 甲板在不同舱位的含义不同。一等舱的 C-Deck 和三等舱的(无Cabin)需交叉 |

**Deck 分布：**
| Deck | 训练集频次 | 存活率 | 备注 |
|------|----------|--------|------|
| A | 15 | 46.7% | 顶层，最贵 |
| B | 47 | 74.5% | 一等舱核心甲板 |
| C | 59 | 59.3% | 一等舱 |
| D | 33 | 75.8% | 一二等舱交界 |
| E | 32 | 75.0% | 一二等舱 |
| F | 13 | 61.5% | 二三等舱 |
| G | 4 | 50.0% | 三等舱最底层 |
| T | 1 | 0.0% | 仅1人（测试集可能无） |
| M(Missing) | 687 | 29.8% | 大部分三等舱乘客 |

**设计说明：**
- D1 (Deck) 75% 的值是 Missing，分类数 9。OneHot 会加 8 维但大部分样本落在 Missing 列。Count Encoding 或 Target Encoding(LOOCV+平滑) 更合适的编码问题留给 Agent C。
- D3 (CabinNum) 信号弱且与 Deck 高度相关，**低优先级**，预期在筛选器阶段被淘汰。
- D4 (CabinShared) 可能已由 Ticket 特征群组覆盖，同样低优先级。

### 2.5 E 类：Ticket 特征（7 个）

| # | 特征名 | 类型 | 构造方式 | 说明 |
|---|--------|------|---------|------|
| E1 | TicketPrefix | categorical | Ticket 字符串中的字母部分（无字母则 `NUM`） | 约 200+ 唯一值，>80% 频次 <5。需激进合并。可能是售票代理/批次标识 |
| E2 | TicketLen | continuous | len(Ticket) | 票号长度。短票号（如 "PC 17608" vs "1601"）可能是电话预订/特殊渠道 |
| E3 | TicketNumLen | continuous | 票号中数字部分长度 | 纯数字票 vs 字母+数字票的差异性 |
| E4 | TicketGroupSize | continuous | 同一 Ticket 号的乘客数 | 同票同行人数。可能比 FamilySize 更准确（家庭+仆人共用票）。核心特征 |
| E5 | TicketGroupSurvival | continuous | **LOOCV Target Encoding**：对每个训练样本，用其他所有样本中同 Ticket 群体的存活率估计 | 极度危险特征。在 CV fold 内计算，smoothing=20 |
| E6 | TicketSurvivedAll | binary | 同票群体全部存活（0/1/-1=独行） | E4 和 E5 的组合信号。全存活票 = 家庭/团体全活 |
| E7 | TicketSurvivedNone | binary | 同票群体全部遇难（0/1/-1=独行） | 同上，反向信号 |

**设计说明（重要）：**
- E5/E6/E7 是本方案中最容易过拟合的特征组。E5 本质上是用 Survived target 编码的，尽管用 LOOCV 避免了全量泄漏，但在 891 样本上，同票群体通常 2-4 人，存活率估计的方差极大。
- **E5 设置高平滑参数（smoothing=20）**：global mean 权重更大，group mean 仅在 ticket group size 大时占优。这相当于"相信全局先验，除非同票群体有压倒性证据"。
- E4 是 E 类中最安全的特征——它是纯粹的数值计数，不涉及 target。
- 如果筛选器发现 E5/E6/E7 的 permutation importance 方差过大 → 标记为不稳定 → 淘汰。

### 2.6 F 类：交互特征（9 个）

| # | 特征名 | 类型 | 构造方式 | 说明 |
|---|--------|------|---------|------|
| F1 | Sex_Pclass | categorical | Sex × Pclass (6类) | **Titanic 最强交互**。女性一等舱存活率 97%，男性三等舱存活率 14% |
| F2 | Sex_Age | continuous | Sex × Age（数值相乘） | 女性和男性的年龄效应不同：女性各年龄段存活率均高且平坦，男性年龄梯度明显（男童 > 青年男性 > 老年男性） |
| F3 | Pclass_Age | continuous | Pclass × Age | 年龄效应在舱位间的差异：一等舱老年人存活率仍高（财富），三等舱儿童存活率低于一等舱成人 |
| F4 | Pclass_Fare | continuous | Pclass × Fare_log | 不同舱位内票价差异的含义不同：一等舱 Fare 方差巨大（富人贫富差距），三等舱 Fare 集中 |
| F5 | Sex_FamilySize | categorical | Sex × FamilySize_bin | 单身女性和单身男性的存活模式差异远大于有家庭者 |
| F6 | Pclass_FamilySize | categorical | Pclass × FamilySize_bin | 三等舱大家庭 vs 一等舱独行：社会网络和资源的交叉 |
| F7 | Pclass_Embarked | categorical | Pclass × Embarked (8类，Q无一等舱) | 同舱位不同港口登船的乘客群体差异（Cherbourg 登船的国际旅客多） |
| F8 | Sex_Embarked | categorical | Sex × Embarked (6类) | 不同港口登船者的性别比例和存活率差异 |
| F9 | Age_Pclass_Sex | categorical | Age_bin × Pclass × Sex 三路交互 | 激进特征。如"三等舱老年男性"vs"一等舱女童"。高度稀疏，候选优先级最低 |

**设计说明：**
- F1 是本方案中优先级最高的交互特征。它在 V1/V3 中都未被显式构造（V3 让树模型自己学，但显式编码可能在 LR 模型中释放额外信号，在树模型中也可能通过减小搜索空间来改善小样本拟合）。
- F9 是激进尝试。25 类以上，每类平均 < 36 个样本。**低优先级，预期在筛选器阶段被淘汰**，除非群组信号极强。
- F5-F8 作为候选项，让数据决定是否有新增信息（已有 Sex/Pclass/Embarked/FamilySize 的主效应，交互可能冗余）。
- 所有交互特征的编码：低基数（<8类）用 OneHot，高基数用 Count Encoding 或 Target Encoding(LOOCV)。

### 2.7 G 类：聚合统计特征（5 个）

| # | 特征名 | 类型 | 构造方式 | 说明 |
|---|--------|------|---------|------|
| G1 | Pclass_FareMean | continuous | 训练集中各 Pclass 的 Fare 均值（在 CV fold 内计算） | 乘客票价相对于该舱位平均票价的偏离程度 |
| G2 | Pclass_FareRank | continuous | 乘客 Fare 在该 Pclass 内的分位数（0-1） | 同舱内的相对财富。G1 和 G2 在不同尺度表达同一概念 |
| G3 | Title_SurvivalRate | continuous | **LOOCV Target Encoding**，smoothing=10 | Title 的全局存活率编码。比 C1 LabelEncoded 多了 target 信息 |
| G4 | Deck_SurvivalRate | continuous | **LOOCV Target Encoding**，smoothing=10 | Deck 的存活率编码 |
| G5 | FamilyID | categorical | Surname 首词 + FamilySize 组合 | 同一家庭标识。可能用于识别家庭存活一致性。极高基数（~400+唯一值），仅作为 Count Encoding 或弃用 |

**设计说明（关键）：**
- G3/G4 是 V1 中 Target Encoding 的改良版——区别在于 LOOCV + 强平滑。但必须警示：**历史 LESSONS.md 明确指出 Target Encoding 是小样本毒药**。G3/G4 作为候选进入筛选器，如果 Permutation Importance 方差 > 均值的 50% 或 Stability Selection 概率 < 0.5 → 标记为高危 → 淘汰。
- G1 和 G2 是安全的聚合特征——不涉及 Survived target。G1 表达绝对偏离（"这个人的票比同舱位平均贵多少"），G2 表达排位（"在同舱中排前 20%"）。树模型可能偏好 G2（分位数归一化后的排序信号），线性模型可能偏好 G1（自然尺度）。
- G5 基数极高，不适合直接编码。如果使用，仅用 `FamilyID_Count`（同一 FamilyID 出现次数）作为 Count Encoding，不参加 Target Encoding。

---

## 3. 编码策略总表

| 特征组 | 特征 | 推荐编码 | 理由 |
|--------|------|---------|------|
| Sex | A2 | Binary 0/1 | — |
| Pclass | A1 | Ordinal 1/2/3 | 节省维度，树模型可自动学习非线性切分 |
| Embarked | A5 | OneHot (3 维) | 无序 3 类，OneHot 代价小 |
| Title (C1) | C1 | Label Encoded (0-7) | 8 类，树模型可直接用 ordinal；LR 需 OneHot 但增加 7 维 |
| Title_Rare | C2 | Binary | — |
| Title_Pclass | C3 | Count Encoding 或 OneHot(~16类) | 16 类 OneHot 尚可接受 |
| Deck | D1 | Count Encoding 或 Target Encoding(smoothing=10) | 9 类，75% Missing，OneHot 太稀疏 |
| HasCabin | D2 | Binary | — |
| CabinNum | D3 | Ordinal 分箱 | 与 D1 高度相关，可能不值得编码 |
| TicketPrefix | E1 | Frequency Encoding（合并后约 20 类） | 200+ 类合并后约 20 类，Count Encoding 最佳 |
| TicketGroupSize | E4 | 保留数值（StandardScaler） | 连续值，有小部分的非线性效应 |
| TicketGroupSurvival | E5 | 保留连续值（已是概率） | 0-1 范围，已平滑 |
| FamilySize | B1 | 保留数值 + 分箱 OneHot | 双向保留：数值给线性信号，分箱给非线性 |
| FamilySize_bin | B3 | OneHot (4 维) | 4 类，代价小 |
| IsAlone | B2 | Binary | — |
| Sex_Pclass | F1 | OneHot (6 维) | 6 类 OneHot 代价小 |
| Sex_Age | F2 | 保留数值 | 连续交互 |
| Pclass_Age | F3 | 保留数值 | 连续交互 |
| Age | A3 | StandardScaler | 标准化消除量纲 |
| Fare_log | A4 | StandardScaler | 已做 log 变换 |

**编码顺序约定（给 Agent C）：**
1. 先完成所有特征构造（Section 2 的 44 个候选）
2. 稀有类别合并（频次 < 5 → `Rare`）
3. OneHot 编码先做（低基数特征）
4. Target Encoding / Count Encoding 后做（高基数特征）
5. 最后 StandardScaler（数值列）

---

## 4. 特征筛选协议（5 阶段，防过拟合护栏）

这是 V4 方案最核心的模块。V1 的 54 个特征未经任何筛选直接进入模型是 CB->LB 崩塌的根本原因。V4 必须在模型训练前完成特征筛选。

### 阶段 1：互信息（Mutual Information）筛选

- **方法**：`sklearn.feature_selection.mutual_info_classif`
- **参数**：`n_neighbors=3, random_state=42`
- **阈值**：MI < 0.01 -> 删除
- **目的**：剔除与 Survived 完全无关的噪声特征
- **预期淘汰**：TicketLen (E2)、TicketNumLen (E3) 等可能落在淘汰区

### 阶段 2：Spearman 相关性去重

- **方法**：计算所有数值特征对 Spearman 秩相关系数
- **阈值**：|r| > 0.85
- **去重规则**：对每对高相关特征，保留 MI 更高者
- **特殊处理**：OneHot 生成的列组（如 Embarked_C/Q/S）互斥，不参与去重（它们的相关性是结构性的，不是冗余信号）
- **预期淘汰示例**：
  - `FamilySize` vs `FamilySize_bin` 高度相关 → 保留 MI 更高者
  - `IsAlone` vs `FamilySize=1` → 保留一个
  - `G1(Pclass_FareMean)` vs `A1(Pclass)` 可能高相关 → 保留 Pclass

### 阶段 3：Boruta（Shadow Features）

- **方法**：`boruta.BorutaPy`（基于 RF 的 all-relevant feature selection）
- **参数**：`n_estimators=200, max_depth=5, perc=100, max_iter=100, random_state=42`
- **原理**：为每个真实特征创建随机打乱的 shadow 副本，只保留比最佳 shadow 表现更好的特征
- **输出**：Accepted / Tentative / Rejected 三分类
- **策略**：Accepted -> 保留；Tentative -> 候选池（视后续筛选结果）；Rejected -> 淘汰

### 阶段 4：Stability Selection

- **方法**：L1-LogisticRegression 在 100 次 bootstrap 子样本上拟合
- **参数**：`n_bootstrap=100, sample_fraction=0.7, C=0.5, random_state=42`
- **阈值**：特征被选中（系数非零）的概率 > 0.6 -> 保留
- **原理**：L1 正则化的特征选择不稳定（数据微扰导致特征集变化），Stability Selection 通过 bootstrap 稳定化，只保留"反复被选中"的稳健特征
- **优势**：严格控制假阳性率（False Discovery Rate），特别适合小样本场景

### 阶段 5：Permutation Importance

- **方法**：RF（`n_estimators=200, max_depth=5, random_state=42`）拟合后打乱每列
- **参数**：`n_repeats=20, random_state=42`
- **阈值**：`mean_importance - 2 * std_importance > 0` -> 保留
- **原理**：如果打乱某列对模型预测无影响，该列没有提供独特信息

### 最终特征集决策规则

```
最终特征集 = (Boruta Accepted ∪ 阶段 4 概率>0.6 ∪ 阶段 5 mean-2std>0) 三路交集
              ∪ (三路中至少两路通过的"边缘特征"作为候选池，人工审查后决定)
```

**三路交集**：三个筛选器都同意的特征是最可信的核心特征集（预计 15-20 个）。
**候选池（两路通过）**：预计 5-10 个，人工审查后择优保留或不保留（最终控制在 25-35 总特征）。

### 筛选协议的可视化流程

```
44 候选特征
    │
    ▼ 阶段 1: MI < 0.01 淘汰
~38 特征
    │
    ▼ 阶段 2: |r| > 0.85 去重
~30 特征
    │
    ├──► 阶段 3: Boruta ──► Accepted/Tentative
    ├──► 阶段 4: Stability ──► 概率>0.6
    └──► 阶段 5: Permutation ──► mean-2std>0
         │
         ▼ 三路交集
    最终 25-35 特征
```

---

## 5. 小样本特殊防护措施

### 5.1 稀有类别合并
- **阈值**：频次 < 5 的类别合并为 `Rare`
- **适用范围**：Title(已做)、Deck(T=1)、TicketPrefix(约 150 个低频)、FamilyID
- **实施时机**：编码之前、仅在训练集上计算频次（测试集不引入新类别信息）

### 5.2 Target Encoding 的安全实施
- **LOOCV 编码**：对每个样本 i，用除 i 以外的所有训练样本计算该类的 target mean
- **平滑先验**：`encoded_value = (n_class * class_mean + smoothing * global_mean) / (n_class + smoothing)`
- **平滑参数**：Title smoothing=10, Deck smoothing=10, TicketGroup smoothing=20（最保守）
- **实施时机**：在 CV 的每个 fold 内部独立计算（`fit_transform` 的逻辑在 5-fold CV 内完成）
- **不进入测试集**：测试集的 target encoding 值用训练集全量统计量（已平滑）计算

### 5.3 零方差特征
- 交互特征构造后检查是否产生零方差列（如某个 Sex_Pclass 组合在数据中不存在）
- 如有，删除该列

### 5.4 信息泄漏红线
| 禁止 | 允许 |
|------|------|
| 用测试集 Survived 做任何计算 | 用训练集的 Survived 在 CV fold 内做编码 |
| 用全量训练集 Survived 均值做编码（V1 的错误） | LOOCV 或 CV fold 内编码 |
| 稀有类别合并使用测试集频次 | 仅用训练集频次判定稀有度 |
| 在特征筛选中使用测试集 | 仅用训练集和 CV 过程 |

---

## 6. 特征优先级与降级路径

如果最终筛选出的特征超过 35 个（三路交集太宽），按以下优先级降级：

| 优先级 | 特征组 | 理由 |
|--------|--------|------|
| P0（必保留） | A(除A5) + B1+B2 + C1 + D1+D2 + E4 + F1 | 核心信号，V3 已验证 + 关键交互 |
| P1（高优先） | B3+B4+B5+B6 + C2 + D2 + E5 | 增强信号，预期稳健 |
| P2（中优先） | C3+C4+C5 + E1+E6+E7 + F2+F3+F4 + G1+G2 | 补充信号，需筛选验证 |
| P3（实验性） | B7 + C6 + D3+D4+D5 + F5-F9 + G3+G4+G5 | 高风险/实验性特征 |

如果 P0+P1 已达 20+，P2+P3 从严筛选；如果 P0+P1 < 15，P2 放宽两路即可。

---

## 7. 预期风险与对策

| 风险 | 概率 | 影响 | 对策 |
|------|------|------|------|
| 过拟合（CV-LB gap 扩大） | 中 | 高 | 5 阶段筛选严格控制特征数；三路交集的严格标准 |
| Target Encoding 毒化 | 中 | 高 | LOOCV + 强平滑(smoothing >= 10)；Permutation Importance 检测后淘汰 |
| 交互特征稀疏导致模型不稳定 | 低 | 中 | 稀有类别合并（<5 → Rare）；OneHot 仅用于低基数交互 |
| 筛选器间不一致导致特征集过小 | 低 | 低 | 两路通过候选池兜底；人工审查补充 |
| Ticket 特征群泄漏 | 中 | 高 | E5 smoothing=20; E6/E7 标记为高风险候选 |

---

## 8. 输出物清单（交付给 Agent C）

Agent C 收到本文档后应产出：

1. `v4/features.py` — 特征工程模块（包含所有构造函数和筛选协议）
2. `v4/preprocessing.py` — 缺失值处理模块
3. `v4/encoding.py` — 编码策略实现（含 LOOCV Target Encoding）
4. `v4/selection.py` — 5 阶段筛选器实现
5. `v4/feature_report.txt` — 筛选结果汇总（各阶段存活特征列表 + 最终特征集）
6. `v4/X_train_processed.csv` — 最终特征矩阵（保存供模型训练使用）
7. `v4/X_test_processed.csv` — 测试集特征矩阵

---

## 附录：数据诊断号

| 指标 | 数值 | 说明 |
|------|------|------|
| 训练集样本 | 891 | 二分类 |
| 测试集样本 | 418 | 无标签 |
| 存活率 | 38.4% (342/891) | 轻度不平衡，不需要 SMOTE 或 class_weight |
| 缺失最严重列 | Cabin (77.1%) | 用 HasCabin + Deck(Missing) 策略 |
| 数值列偏度 | Fare: skew≈4.8（极端右偏） | log1p 后 skew≈0.5 |
| Age 分布 | 均值 29.7, 中位数 28.0, skew≈0.4 | 轻微右偏，不需要变换 |
| 类别分布 | Survived=0: 549 (61.6%), Survived=1: 342 (38.4%) | — |

---

*V4 方案 v1.0 — Agent B (Feature Engineering Designer)*
*待 Agent C 实现，Agent E (Critic) 审查*
