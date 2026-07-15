#!/d/py/Python3/python
# -*- coding: utf-8 -*-
import json, subprocess

C = []
def title(text, s='22pt'): C.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"align":"center","bold":True,"size":s,"font":"黑体","font.ea":"黑体","spaceBefore":"24pt","spaceAfter":"6pt","lineSpacing":"1.5x"}})
def subt(text): C.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":text,"align":"center","size":"14pt","font":"宋体","font.ea":"宋体","spaceAfter":"18pt","lineSpacing":"1.5x"}})
def h1(t): C.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":t,"align":"center","bold":True,"size":"16pt","font":"黑体","font.ea":"黑体","spaceBefore":"18pt","spaceAfter":"6pt","lineSpacing":"1.5x"}})
def h2(t): C.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":t,"align":"left","bold":True,"size":"14pt","font":"黑体","font.ea":"黑体","spaceBefore":"12pt","spaceAfter":"3pt","lineSpacing":"1.5x"}})
def h3(t): C.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":t,"align":"left","bold":True,"size":"12pt","font":"黑体","font.ea":"黑体","spaceBefore":"6pt","spaceAfter":"3pt","lineSpacing":"1.5x"}})
def p(t): C.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":t,"font":"宋体","font.ea":"宋体","size":"12pt","firstLineIndent":"0.74cm","lineSpacing":"1.5x","spaceAfter":"2pt"}})
def r(t): C.append({"command":"add","parent":"/body","type":"paragraph","props":{"text":t,"font":"宋体","font.ea":"宋体","size":"10.5pt","lineSpacing":"1.5x","spaceBefore":"0pt","spaceAfter":"1pt"}})

title('应用统计与数据科学硕士项目趋势分析（2026-2027）')
subt('新设项目 · 政策趋势 · 报考机遇')

h1('一、2026年新增/调整的应用统计硕士项目')

h2('1.1 继续大幅扩招的院校')
p('复旦大学：2026年计划统招198人（较2025年144人增长37.5%），大数据学院扩至110人。')
p('中南财经政法大学：2026年计划扩招至177人（较2025年99人增长78.8%），复试线预计回升至355分左右。')
p('华东师范大学：招生规模稳定在65人左右，2026年计划与2025年持平。')
p('浙江财经大学：2026年预计继续维持80-90人的招生规模。')

h2('1.2 2026年统考名额严重缩水的院校（避坑预警）')
p('中国人民大学：2026年统计学院仅2人，统计与大数据研究院仅2人，合计统考4人（2025年为48人），缩水超90%。')
p('武汉大学：2025年统考仅5人，2026年有望回升至20人（官方计划），但竞争仍将激烈。')

h1('二、数据科学专业硕士（1453）2026年招生情况')

h2('2.1 已确认招生的院校')
p('武汉大学 1453S1 数据科学：中部首个数据科学专业硕士，2026年首批招生。学制2年，学费8万/年。培养方向：大数据技术、人工智能理论与应用、大模型方法与应用。依托数学与统计学院、计算机学院等6大院系，专任教师69人（含院士1人）。')
p('北京大学 1453S1 数据科学（大数据专硕）：数学科学学院招生，2026年复试线360分，录取22人。初试考822数据科学基础（统计学90分+算法与数据结构60分）。学制2年，学费20万。')

h2('2.2 新增招生预测')
p('华东师范大学：2026年拟新增"数据科学专业硕士"（专业学位硕士点），已进入新增硕士点名单，等待教育部正式批复。若获批，将是长三角地区首个独立设置的数据科学专硕。')
p('清华大学深圳国际研究生院：0812J3数据科学和信息技术（学硕）招生30人；085411大数据技术与工程（专硕）招生16人。')

h1('三、2027年新增硕士点预测——数据科学方向')

h2('3.1 已确认新增')
p('北京大学 "数据科学"硕士点（交叉学科）：数学科学学院+计算机学院共建，2027年开始招生。考试科目二选一：①数学分析+高等代数（数学方向）；②数学一+801计算机专业基础（计算机方向）。招生约27人（含推免17人）。')

h2('3.2 高概率新增的院校')
p('预测一：华东师范大学数据科学专硕。如2026年获批，将在2027年正式招生。基于其数据科学与工程学院多年的交叉学科积累（2022年已升级为一级交叉学科），招生概率极高。')
p('预测二：复旦大学大数据学院或将申请数据科学独立专硕。目前其大数据学院已有成熟的大数据方向，若获批复，将在应统之外开辟新赛道。')
p('预测三：上海财经大学统计学院可能在统计学下增设数据科学交叉方向，发挥其财经数据优势。')
p('预测四：浙江大学数学科学学院或申请数据科学交叉学科项目。浙大已有大数据方向，且杭州是数据科学产业重镇。')

h1('四、交叉学科（第14大门类）整体趋势')

h2('4.1 核心数据')
p('交叉学科硕士招生年复合增长率达43%，远超第二名工学（4.3%）。招生人数从2022年约1.2万人增至2026年约5.1万人。专硕招生占比已达69.7%，学硕持续缩招。')

h2('4.2 政策驱动')
p('2025年8月，中央教育工作领导小组印发《高等教育学科专业设置调整优化行动方案（2025-2027年）》，强调新兴学科和交叉学科的孵化。')
p('2025年11月，国家发改委、国家数据局、教育部、科技部、中组部五部门联合发文，明确支持数据科学与工程、数字经济与管理等数据要素相关学科专业建设。')

h2('4.3 2027年新增交叉学科预测方向')
p('数据科学/数据要素：受五部门联合发文驱动，预计更多理工类和财经类院校增设数据科学交叉学科。')
p('人工智能+X：AI赋能传统学科（金融、医学、农业等），预计30+所院校新增AI交叉方向。')
p('数字经济：经济学与数据科学交叉，适合应统和数据科学背景考生。')
p('金融科技：金融+统计+计算机，复旦等校已开设，更多院校跟进。')

h1('五、对你（浙财本科）的启示与建议')

h2('5.1 2026-2027年报考策略')
p('策略一：关注新开设的数据科学专硕（1453）。新项目头一两年通常竞争较小，如北大2027年数据科学新增、武大1453S1首年招生、华师大数据科学专硕（若获批）。但需注意考试科目可能不同于432统计学。')
p('策略二：应用统计（025200）依然是基本盘。扩招院校（复旦、中南财、浙财本部）上岸概率更高。需警惕人大等统考名额骤降的院校。')
p('策略三：交叉学科是蓝海。第14大门类年增长43%，数据科学交叉方向是政策红利区，建议重点关注。')

h2('5.2 备考建议')
p('（1）数学三（303）是应统考研的核心拉分科目，目标130+。')
p('（2）432统计学：茆诗松《概率论与数理统计教程》+贾俊平《统计学》是90%院校的核心。如果要考数据科学方向，需额外准备算法与数据结构。')
p('（3）英语一 vs 英语二：数据科学方向多考英语一，难度高于英语二。')
p('（4）关注9月招生简章：各校2027年招生简章通常在2026年9-10月发布，重点关注新设硕士点的考试科目。')

h1('参考文献')
r('[1] 新东方. 2027中国大学生考研白皮书[EB/OL]. https://kaoyan.xdf.cn')
r('[2] 新东方网. 这些985院校2027年新增硕士点，值得关注[EB/OL]. 2026-04.')
r('[3] 新东方网. 交叉学科硕士招生年复合增长43%[EB/OL]. 2026-04.')
r('[4] 搜狐教育. 交叉学科年增长43% 2027年考生迎来新赛道机遇[EB/OL]. 2026-04.')
r('[5] 复旦大学研究生院. 2026年硕士研究生招生计划.')
r('[6] 武汉大学数据科学专硕招生公告[EB/OL]. 武汉大学数学与统计学院.')
r('[7] 中国人民大学统计学院. 2026年硕士研究生招生简章.')
r('[8] 国家发改委等五部门. 关于加强数据要素学科专业建设的意见[EB/OL]. 2025-11.')

# ===== EXECUTE =====
json_path = 'E:/wuyi/数学建模半自动/research-assistant/outputs/new_programs_batch.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(C, f, ensure_ascii=False, indent=2)
print(f'JSON saved: {len(C)} commands')

oc = 'C:/Users/lenovo/AppData/Local/OfficeCLI/officecli.exe'
dp = 'E:/wuyi/数学建模半自动/research-assistant/outputs/应统与数据科学趋势分析.docx'

def run(a):
    r = subprocess.run(a, capture_output=True)
    return r.stdout.decode('utf-8', errors='replace').strip()

run([oc, 'close', dp])
run([oc, 'create', dp])
run([oc, 'set', dp, '/', '--prop', 'pageWidth=21cm', '--prop', 'pageHeight=29.7cm',
    '--prop', 'marginTop=2.54cm', '--prop', 'marginBottom=2.54cm',
    '--prop', 'marginLeft=3.18cm', '--prop', 'marginRight=3.18cm',
    '--prop', 'marginHeader=1.5cm', '--prop', 'marginFooter=1.75cm'])
run([oc, 'set', dp, '/', '--prop', 'docDefaults.font=宋体', '--prop', 'docDefaults.fontSize=12pt'])
run([oc, 'add', dp, '/', '--type', 'header', '--prop', 'type=default',
    '--prop', 'text=应用统计与数据科学硕士趋势分析（2026-2027）', '--prop', 'align=center', '--prop', 'size=9pt', '--prop', 'font=宋体', '--prop', 'bold=true'])
run([oc, 'add', dp, '/', '--type', 'footer', '--prop', 'type=default', '--prop', 'text=— ', '--prop', 'align=center', '--prop', 'size=9pt', '--prop', 'font=宋体'])
run([oc, 'add', dp, '/footer[1]/p[1]', '--type', 'field', '--prop', 'fieldType=page', '--prop', 'size=9pt', '--prop', 'font=宋体'])
run([oc, 'add', dp, '/footer[1]/p[1]', '--type', 'run', '--after', 'r[6]', '--prop', 'text= —', '--prop', 'font=宋体', '--prop', 'size=9pt'])

r2 = run([oc, 'batch', dp, '--input', json_path, '--json'])
print(r2[:800])
run([oc, 'close', dp])
print('Done!')
