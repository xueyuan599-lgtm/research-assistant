#!/d/py/Python3/python
# -*- coding: utf-8 -*-
import json, subprocess, os

commands = []

def title(text, size='22pt'):
    commands.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"align":"center","bold":True,"size":size,"font":"黑体","font.ea":"黑体","spaceBefore":"24pt","spaceAfter":"6pt","lineSpacing":"1.5x"}})
def subtitle(text):
    commands.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"align":"center","size":"14pt","font":"宋体","font.ea":"宋体","spaceAfter":"18pt","lineSpacing":"1.5x"}})
def h1(text):
    commands.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"align":"center","bold":True,"size":"16pt","font":"黑体","font.ea":"黑体","spaceBefore":"18pt","spaceAfter":"6pt","lineSpacing":"1.5x"}})
def h2(text):
    commands.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"align":"left","bold":True,"size":"14pt","font":"黑体","font.ea":"黑体","spaceBefore":"12pt","spaceAfter":"3pt","lineSpacing":"1.5x"}})
def h3(text):
    commands.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"align":"left","bold":True,"size":"12pt","font":"黑体","font.ea":"黑体","spaceBefore":"6pt","spaceAfter":"3pt","lineSpacing":"1.5x"}})
def body(text):
    commands.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"font":"宋体","font.ea":"宋体","size":"12pt","firstLineIndent":"0.74cm","lineSpacing":"1.5x","spaceAfter":"2pt"}})
def ref(text):
    commands.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"font":"宋体","font.ea":"宋体","size":"10.5pt","lineSpacing":"1.5x","spaceBefore":"0pt","spaceAfter":"1pt"}})
def blank():
    commands.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":"","size":"6pt","spaceBefore":"0pt","spaceAfter":"0pt"}})

def run_cli(args_list):
    """Run officecli and return stdout as utf-8 string"""
    proc = subprocess.run(args_list, capture_output=True)
    out = proc.stdout.decode('utf-8', errors='replace')
    err = proc.stderr.decode('utf-8', errors='replace')
    if err.strip():
        print(f'  [stderr] {err[:200]}')
    return out.strip()

# ===== CONTENT =====
title('应用统计硕士（025200）考研择校分析报告', '22pt')
subtitle('—— 含参考书目与数据科学方向 基于2025年招录数据')
body('报告说明：本报告基于2025年全国硕士研究生招生录取数据，系统梳理了长三角、武汉、北京地区主要高校应用统计专业（025200）的招录信息及参考书目，并补充了数据科学/大数据相关专业方向。数据来源为各高校研究生院官网、统计学院公告及权威考研资讯平台。部分数据以"预估"或"未公布"标注，建议以各校最新招生简章为准。')

h1('第一部分 长三角地区（上海/杭州/南京/苏州）')

h2('1.1 复旦大学')
body('招生单位：数学科学学院、上海数学与交叉学科研究院、上海数学中心、大数据学院。2025年复试线345分（较2024年395分降50分），统考计划约147人，录取约144人，报录比约3:1。大数据学院录取最低分369分，最高分451分。2026年计划统招198人。')
h3('参考书目')
body('茆诗松《概率论与数理统计教程》（第三版），高等教育出版社【核心】')
body('李贤平《概率论基础》（第三版），高等教育出版社【拔高】')
body('应坚刚《概率论》，复旦大学出版社')
body('韦来生《数理统计》，科学出版社')
body('备考提示：复旦432以概率论难度大著称，李贤平《概率论基础》是拉开差距的关键。学制2-3年，学费21.8-25万。')

h2('1.2 上海交通大学')
body('招生单位：数学科学学院。2025年复试线375分，统考录取61人（含扩招），复录比1.21:1，报录比4.7:1。专业课平均分137+。初试考英语（一）。')
h3('参考书目（官方指定）')
body('何书元《概率论》，北京大学出版社')
body('韦来生《数理统计》，科学出版社')
body('贾俊平《统计学》（第7/8版），中国人民大学出版社')
body('辅助：茆诗松《概率论与数理统计教程》。学制2年，学费约20万。')

h2('1.3 华东师范大学')
body('招生单位：统计学院。2025年复试线360分，统考录取65人，报录比3.4:1（近5年最低）。录取最低分360分，最高分430分。专业课平均分130分。')
h3('参考书目')
body('贾俊平《统计学》（第7/8版），中国人民大学出版社【核心】')
body('茆诗松《概率论与数理统计教程》（第三版）【核心】')
body('刘剑平《应用数理统计》，华东理工大学出版社')
body('何晓群《应用回归分析》（第5版）；王燕《应用时间序列分析》')
body('备考提示：华师大432大纲涵盖概率论16条+统计学16条。学费10万/年，学制2年。')

h2('1.4 上海财经大学')
body('招生单位：统计与数据科学学院、数学学院。2025年复试线323分（国家线），统考约93人。')
h3('参考书目（无官方指定，经验总结）')
body('贾俊平《统计学》（第8版），中国人民大学出版社【核心】')
body('茆诗松《概率论与数理统计教程》（第三版）【核心】')
body('何晓群《应用回归分析》；王燕《应用时间序列分析》')
body('韦来生《数理统计》，科学出版社。学制2年，学费20万。')

h2('1.5 南京大学')
body('招生单位：数学学院。2025年复试线355分，统考14人，复录比1.21:1。录取最低分359分。')
h3('参考书目（官方指定）')
body('孙荣恒《应用数理统计》（第三版），科学出版社')
body('高惠璇《应用多元统计分析》，北京大学出版社')
body('李贤平《概率论基础》（第三版），高等教育出版社')
body('Cryer \x26 Chan《时间序列分析及应用（R语言）》（第二版），机械工业出版社')
body('备考提示：南大2025年参考书目有变动。学费约1万/年，学制2年。')

h2('1.6 东南大学')
body('招生单位：数学学院。2025年复试线351分，统考录取15人，仅淘汰1人。')
h3('参考书目')
body('贾俊平《统计学》（第7版），中国人民大学出版社')
body('茆诗松《概率论与数理统计教程》，高等教育出版社')

h2('1.7 苏州大学')
body('招生单位：数学科学学院。2025年复试线370分，统考15人，报录比9.87:1。录取最低分380分。学费仅1万/年。')
h3('参考书目（2025年大更新）')
body('贾俊平《统计学》，中国人民大学出版社')
body('魏宗舒《概率论与数理统计》，高等教育出版社')
body('何晓群《应用回归分析》（第2版 R语言版），电子工业出版社')
body('王燕《应用时间序列分析》，中国人民大学出版社')
body('王学民《应用多元统计分析》，上海财经大学出版社')
body('备考提示：苏大2025年参考书从2本扩至5本，备考范围明显扩大。')

h2('1.8 浙江财经大学（本校）')
body('招生单位：数据科学学院。2025年复试线345分，录取约90人，进复试125人。学费1万/年。')
h3('参考书目')
body('李金昌《统计学》（核心指定教材）')
body('配套：李金昌《统计学》复习笔记及题库')
body('备考提示：浙财使用李金昌版教材，与多数院校不同，本校考生有课程基础优势。')

h1('第二部分 武汉地区')

h2('2.1 武汉大学')
body('招生单位：数学与统计学院。2025年复试线380分，统考仅5人，报录比9.17:1。录取最低分382分。有396分、394分考生因笔试低被淘汰。学费涨至6万/年。')
h3('参考书目（无官方指定）')
body('茆诗松《概率论与数理统计教程》（第三版）【核心】')
body('李贤平《概率论基础》；韦来生《数理统计》')
body('贾俊平《统计学》（第7版）')

h2('2.2 华中科技大学')
body('招生单位：数学与统计学院。2025年复试线365分，统考录取31人，复录比约1.37:1。')
h3('参考书目')
body('刘次华、万建平《概率论与数理统计》（第三版），华中科技大学出版社【华科本校教材】')
body('茆诗松《概率论与数理统计教程》')
body('贾俊平《统计学》（第7版）')
body('备考提示：有官方考试大纲，概率论60分+统计学90分，不允许使用计算器。')

h2('2.3 中南财经政法大学')
body('招生单位：统计与数学学院。2025年复试线323分（国家线），统考约99人。2026年计划扩招至177人。')
h3('参考书目')
body('茆诗松《概率论与数理统计教程》')
body('贾俊平《统计学》。学制2年，学费3.5万/年。')

h1('第三部分 北京地区')

h2('3.1 北京大学（大数据专硕/数据科学）')
body('原应用统计已停招。现对应大数据专硕（1453S1数据科学），数学科学学院招生。2025年复试线360分，录取22人，录取最低分380分。')
h3('参考书目（822数据科学基础）')
body('统计学部分（90分）：茆诗松《概率论与数理统计教程》')
body('算法与数据结构部分（60分）：《算法导论》或数据结构教材')
body('初试考822数据科学基础（非432统计学）。学制2年，学费20万。')

h2('3.2 中国人民大学')
body('统计学院、统计与大数据研究院。2025年复试线355分，合计录取约48人。2026年统考仅4人，竞争将极其惨烈。')
h3('参考书目')
body('贾俊平《统计学》（第7版），中国人民大学出版社【核心】')
body('茆诗松《概率论与数理统计教程》（第三版）【核心】')
body('何晓群《应用回归分析》《多元统计分析》')
body('王燕《应用时间序列分析》；金勇进《抽样技术》')
body('备考提示：人大参考书较多，需全面准备回归、多元统计、时间序列、抽样技术。')

h2('3.3 北京师范大学')
body('统计学院（珠海校区）。2025年复试线345分，录取约237人（全国最多），复录比1.17:1。')
h3('参考书目')
body('茆诗松《概率论与数理统计教程》（第三版）【核心】')
body('贾俊平《统计学》（第7/8版），中国人民大学出版社')
body('何晓群《应用回归分析》《多元统计分析》')
body('王燕《应用时间序列分析》')
body('备考提示：北师大432侧重统计学100分+概率论50分。招生全国最多，上岸率最高。')

h2('3.4 中央财经大学')
body('统计与数学学院。2025年复试线342分，统考59人，报录比4.8:1。学费仅2.5万/年。')
h3('参考书目')
body('刘扬《统计学》（核心教材，中财本校用书）')
body('贾俊平《统计学》（辅助补充）')
body('茆诗松《概率论与数理统计教程》（近年概率论占比上升）')

h2('3.5 对外经济贸易大学')
body('统计学院。2025年复试线360分，统考约69人（扩招），复试仅面试无笔试。')
h3('参考书目')
body('贾俊平《统计学》（官方指定唯一参考书）')
body('茆诗松《概率论与数理统计教程》（近年概率论计算题增加）')

h1('第四部分 数据科学方向补充')

h2('4.1 数据科学考研路径')
body('路径一：应用统计（025200）下设数据科学方向。最常见路径，考432统计学。代表：华东师范大学、北京师范大学、浙江财经大学。')
body('路径二：大数据专硕（1453/1453S1）。交叉学科门类。代表：北京大学（1453S1，考822数据科学基础）、武汉大学（1453S1，考899数据科学）。')
body('路径三：统计学下交叉方向（0714J1等）。理学学硕，招生较少。')

h2('4.2 推荐路径对比')
body('偏统计功底 -> 应用统计（025200），考432统计学，院校选择最多，核心参考书茆诗松+贾俊平。')
body('偏计算机/算法 -> 大数据专硕（1453），考数据科学基础，院校选择少但竞争相对缓和。')

h1('第五部分 综合推荐与备考建议')

h2('5.1 针对浙财本科的冲刺推荐')
body('冲刺推荐：华东师范大学（985中报录比最低3.4:1，统考65人）；中央财经大学（211性价比最高，报录比4.8:1，学费2.5万/年）；华中科技大学（统考31人，复试线稳定365分）。')
body('稳妥推荐：北京师范大学（珠海）招生237人；中南财经政法大学扩招至177人，复试线323分；浙江财经大学本校，学费1万/年。')

h2('5.2 核心参考书总览')
body('【通读必读】茆诗松《概率论与数理统计教程》（第三版）—— 90%以上院校的核心参考书。')
body('【通读必读】贾俊平《统计学》（第7/8版）—— 绝大多数院校的统计学基础教材。')
body('【进阶选读】李贤平《概率论基础》—— 复旦、南大等概率论深度要求高的院校必备。')
body('【进阶选读】韦来生《数理统计》—— 多所985推荐补充。')
body('【专题拓展】何晓群《应用回归分析》、王燕《应用时间序列分析》、何晓群《多元统计分析》。')

h2('5.3 考试科目注意事项')
body('注意区分英语一/英语二：上交、人大、央财、北大考英语（一）；复旦、华师大、上财、武大、华科、北师大考英语（二）。英语一难度明显高于英语二。')
body('注意区分432/822/899：应用统计绝大多数考432统计学；北大考822数据科学基础（统计学+算法）；武大数据科学方向考899数据科学。')

# ===== REFERENCES =====
h1('参考文献')
ref('[1] 复旦大学研究生院. 2025年招生复试基本分数线')
ref('[2] 南京大学数学学院. 2025年应用统计参考书目')
ref('[3] 华东师范大学统计学院. 2025年复试录取实施细则')
ref('[4] 武汉大学数学与统计学院. 2025年复试录取工作细则')
ref('[5] 华中科技大学. 2025年研究生招生考试统计学考试大纲')
ref('[6] 中国人民大学统计学院. 2025年复试录取工作方案')
ref('[7] 浙江财经大学研究生院. 2025年复试分数线通知')
ref('[8] 苏州大学数学科学学院. 2025年复试分数线公告')
ref('[9] 各考研资讯平台（启航考研、新东方考研、应统联盟等）')

# ===== EXECUTE =====
json_path = 'E:/wuyi/数学建模半自动/research-assistant/outputs/考研report_batch.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(commands, f, ensure_ascii=False, indent=2)
print(f'JSON saved: {len(commands)} commands')

officecli = 'C:/Users/lenovo/AppData/Local/OfficeCLI/officecli.exe'
docx_path = 'E:/wuyi/数学建模半自动/research-assistant/outputs/应用统计考研择校报告.docx'

# Create fresh document
print('Creating document...')
subprocess.run([officecli, 'close', docx_path], capture_output=True)
r1 = subprocess.run([officecli, 'create', docx_path], capture_output=True)
print(r1.stdout.decode('utf-8', errors='replace').strip())

# Page setup
subprocess.run([officecli, 'set', docx_path, '/', '--prop', 'pageWidth=21cm', '--prop', 'pageHeight=29.7cm',
    '--prop', 'marginTop=2.54cm', '--prop', 'marginBottom=2.54cm',
    '--prop', 'marginLeft=3.18cm', '--prop', 'marginRight=3.18cm',
    '--prop', 'marginHeader=1.5cm', '--prop', 'marginFooter=1.75cm'], capture_output=True)
subprocess.run([officecli, 'set', docx_path, '/', '--prop', 'docDefaults.font=宋体', '--prop', 'docDefaults.fontSize=12pt'], capture_output=True)

# Header/Footer
subprocess.run([officecli, 'add', docx_path, '/', '--type', 'header', '--prop', 'type=default',
    '--prop', 'text=应用统计硕士考研择校分析报告', '--prop', 'align=center', '--prop', 'size=9pt', '--prop', 'font=宋体', '--prop', 'bold=true'], capture_output=True)
subprocess.run([officecli, 'add', docx_path, '/', '--type', 'footer', '--prop', 'type=default',
    '--prop', 'text=— ', '--prop', 'align=center', '--prop', 'size=9pt', '--prop', 'font=宋体'], capture_output=True)
subprocess.run([officecli, 'add', docx_path, '/footer[1]/p[1]', '--type', 'field', '--prop', 'fieldType=page', '--prop', 'size=9pt', '--prop', 'font=宋体'], capture_output=True)
subprocess.run([officecli, 'add', docx_path, '/footer[1]/p[1]', '--type', 'run', '--after', 'r[6]', '--prop', 'text= —', '--prop', 'font=宋体', '--prop', 'size=9pt'], capture_output=True)

# Batch write content
print('Writing content...')
r2 = subprocess.run([officecli, 'batch', docx_path, '--input', json_path, '--json'], capture_output=True)
out2 = r2.stdout.decode('utf-8', errors='replace').strip()
print(out2[:500])

# Close
subprocess.run([officecli, 'close', docx_path], capture_output=True)
print('Done!')
