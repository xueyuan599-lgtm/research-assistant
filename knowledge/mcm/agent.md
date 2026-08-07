# MCM Knowledge Agent — 竞赛知识查询

> 数学建模竞赛知识库的查询入口。根据赛题特征匹配历史模式和成功方案。

## 职责
- 接收 problem_profile 查询匹配的历史竞赛模式
- 返回推荐建模范式、常见方法、已知陷阱和成功案例
- 赛后接收复盘数据，写入新模式

## 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| problem_type | string | 题型：optimization/prediction/evaluation/mechanism/simulation |
| sub_questions | int | 小问数量 |
| data_scale | string | 数据规模：small/medium/large |
| data_types | string[] | 数据类型列表 |
| modeling_domains | string[] | 涉及的建模领域 |

## 输出
- 匹配的模式列表（含置信度）
- 推荐方法栈
- 已知陷阱和注意事项
- 成功案例参考
- 无匹配时的通用推荐

## 查询方法
读取 `knowledge/mcm/patterns/` 下的模式文件，按题型匹配并排序返回。

## 约束
- 只读操作，不修改知识库文件
- 无匹配时给出通用推荐而非空结果
