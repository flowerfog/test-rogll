from pathlib import Path
from datetime import date
import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "ROGII_PrefixRidge_SGraph_local_validation_review_2026-07-19.pdf"

FONT_REGULAR_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
FONT_REGULAR = "SimHei"
pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH)))


# Palette deliberately mixes neutral, teal, amber and red status colors.
INK = colors.HexColor("#20262D")
MUTED = colors.HexColor("#66717D")
LIGHT = colors.HexColor("#F3F5F6")
LINE_COLOR = colors.HexColor("#D7DDE1")
TEAL = colors.HexColor("#147D75")
TEAL_LIGHT = colors.HexColor("#E7F4F2")
AMBER = colors.HexColor("#B26A00")
AMBER_LIGHT = colors.HexColor("#FFF3DE")
RED = colors.HexColor("#B23A3A")
RED_LIGHT = colors.HexColor("#FBEAEA")
GREEN = colors.HexColor("#3F7D4A")
GREEN_LIGHT = colors.HexColor("#EAF5EC")
BLUE = colors.HexColor("#3B6293")
WHITE = colors.white

PAGE_W, PAGE_H = A4
LEFT = 18 * mm
RIGHT = 18 * mm
TOP = 18 * mm
BOTTOM = 17 * mm
CONTENT_W = PAGE_W - LEFT - RIGHT


styles = getSampleStyleSheet()


def style(name, **kwargs):
    base = dict(
        fontName=FONT_REGULAR,
        textColor=INK,
        leading=14,
        fontSize=9.5,
        wordWrap="CJK",
        spaceAfter=4,
    )
    base.update(kwargs)
    return ParagraphStyle(name, **base)


S_BODY = style("Body", fontSize=9.2, leading=14.2)
S_SMALL = style("Small", fontSize=7.7, leading=11.2, textColor=MUTED)
S_NOTE = style("Note", fontSize=8.2, leading=12.2, textColor=MUTED)
S_H1 = style("H1", fontSize=19, leading=24, spaceBefore=2, spaceAfter=8)
S_H2 = style("H2", fontSize=13.2, leading=17, textColor=INK, spaceBefore=7, spaceAfter=5)
S_H3 = style("H3", fontSize=10.5, leading=14, textColor=TEAL, spaceBefore=5, spaceAfter=3)
S_TITLE = style("Title", fontSize=25, leading=31, textColor=INK, alignment=TA_LEFT)
S_SUBTITLE = style("Subtitle", fontSize=11, leading=17, textColor=MUTED)
S_LABEL = style("Label", fontSize=7.2, leading=9, textColor=MUTED)
S_METRIC = style("Metric", fontSize=16, leading=19, textColor=INK)
S_TABLE = style("Table", fontSize=7.3, leading=10.2)
S_TABLE_HEAD = style("TableHead", fontSize=7.4, leading=10, textColor=WHITE, alignment=TA_CENTER)
S_TABLE_RIGHT = style("TableRight", fontSize=7.3, leading=10.2, alignment=TA_RIGHT)
S_STATUS = style("Status", fontSize=10, leading=13, alignment=TA_CENTER, textColor=WHITE)
S_FOOTER = style("Footer", fontSize=6.8, leading=8, textColor=MUTED)


def P(text, sty=S_BODY):
    return Paragraph(text, sty)


def fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def status_chip(text, background):
    table = Table([[P(text, S_STATUS)]], colWidths=[34 * mm], rowHeights=[8 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.4, background),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def callout(title, body, color=TEAL, background=TEAL_LIGHT):
    table = Table(
        [[P(title, style("CallTitle" + title[:2], fontSize=9.8, leading=13, textColor=color)),
          P(body, style("CallBody" + title[:2], fontSize=8.6, leading=13))]],
        colWidths=[35 * mm, CONTENT_W - 35 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def metric_cards(items):
    cells = []
    for label, value, note, color in items:
        cells.append(
            [
                P(label, S_LABEL),
                P(value, style("Metric" + label[:2], fontSize=15.5, leading=19, textColor=color)),
                P(note, S_SMALL),
            ]
        )
    inner = []
    for label, value, note in cells:
        inner.append(
            Table(
                [[label], [value], [note]],
                colWidths=[(CONTENT_W - 8 * mm) / len(items)],
                rowHeights=[5 * mm, 8 * mm, 9 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                        ("BOX", (0, 0), (-1, -1), 0.5, LINE_COLOR),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ]
                ),
            )
        )
    table = Table([inner], colWidths=[(CONTENT_W - 8 * mm) / len(items)] * len(items), hAlign="LEFT")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("RIGHTPADDING", (0, 0), (-2, -1), 4)]))
    return table


def data_table(headers, rows, widths=None, align_right=None, row_colors=None):
    align_right = set(align_right or [])
    table_data = [[P(str(value), S_TABLE_HEAD) for value in headers]]
    for row in rows:
        rendered = []
        for index, value in enumerate(row):
            rendered.append(P(str(value), S_TABLE_RIGHT if index in align_right else S_TABLE))
        table_data.append(rendered)
    table = Table(table_data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(table_data)):
        background = colors.white if row_index % 2 else LIGHT
        if row_colors and row_index - 1 < len(row_colors) and row_colors[row_index - 1] is not None:
            background = row_colors[row_index - 1]
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), background))
    table.setStyle(TableStyle(commands))
    return table


def bullet(text, color=TEAL):
    return Table(
        [[P("●", style("Dot" + str(len(text)), fontSize=7.5, leading=12, textColor=color)), P(text, S_BODY)]],
        colWidths=[5 * mm, CONTENT_W - 5 * mm],
        style=TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        ),
    )


class HorizontalBars(Flowable):
    def __init__(self, items, width=CONTENT_W, height=54 * mm, max_value=None):
        super().__init__()
        self.items = items
        self.width = width
        self.height = height
        self.max_value = max_value or max(value for _, value, _ in items)

    def wrap(self, availWidth, availHeight):
        return min(self.width, availWidth), self.height

    def draw(self):
        canvas = self.canv
        label_w = 39 * mm
        value_w = 17 * mm
        bar_w = self.width - label_w - value_w - 4 * mm
        row_h = self.height / max(len(self.items), 1)
        for index, (label, value, color) in enumerate(self.items):
            y = self.height - (index + 0.72) * row_h
            canvas.setFont(FONT_REGULAR, 8)
            canvas.setFillColor(INK)
            canvas.drawString(0, y, label)
            canvas.setFillColor(LIGHT)
            canvas.roundRect(label_w, y - 1.5, bar_w, 5.5, 1.5, fill=1, stroke=0)
            fill_w = max(0.8, bar_w * min(value / self.max_value, 1.0))
            canvas.setFillColor(color)
            canvas.roundRect(label_w, y - 1.5, fill_w, 5.5, 1.5, fill=1, stroke=0)
            canvas.setFillColor(INK)
            canvas.drawRightString(self.width, y, f"{value:.3f} ft")


class DecisionFlow(Flowable):
    def __init__(self, width=CONTENT_W, height=42 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, availWidth, availHeight):
        return min(self.width, availWidth), self.height

    def draw(self):
        c = self.canv
        boxes = [
            ("P0a", "SGraph 表示", "PASS", GREEN, GREEN_LIGHT),
            ("P0.5", "PrefixRidge", "FAIL", RED, RED_LIGHT),
            ("P1", "冻结长边图", "LIMITED", AMBER, AMBER_LIGHT),
            ("P2a", "学习型边残差", "FAIL", RED, RED_LIGHT),
        ]
        gap = 5 * mm
        box_w = (self.width - gap * 3) / 4
        box_h = 27 * mm
        y = 7 * mm
        for index, (gate, label, result, color, background) in enumerate(boxes):
            x = index * (box_w + gap)
            c.setFillColor(background)
            c.setStrokeColor(color)
            c.roundRect(x, y, box_w, box_h, 2, fill=1, stroke=1)
            c.setFillColor(color)
            c.setFont(FONT_REGULAR, 8.5)
            c.drawString(x + 4 * mm, y + box_h - 7 * mm, gate)
            c.setFillColor(INK)
            c.setFont(FONT_REGULAR, 8.5)
            c.drawString(x + 4 * mm, y + box_h - 14 * mm, label)
            c.setFillColor(color)
            c.setFont(FONT_REGULAR, 7.5)
            c.drawString(x + 4 * mm, y + 4 * mm, result)
            if index < len(boxes) - 1:
                x1 = x + box_w + 1 * mm
                x2 = x + box_w + gap - 1 * mm
                mid_y = y + box_h / 2
                c.setStrokeColor(MUTED)
                c.line(x1, mid_y, x2, mid_y)
                c.line(x2 - 2, mid_y + 2, x2, mid_y)
                c.line(x2 - 2, mid_y - 2, x2, mid_y)


class ReviewDocTemplate(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="ROGII PrefixRidge-SGraph v1.1 本地验证复盘",
            author="Codex",
            subject="PrefixRidge-SGraph 本地实验结果与停止结论",
        )
        frame = Frame(LEFT, BOTTOM, CONTENT_W, PAGE_H - TOP - BOTTOM, id="normal")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._decorate))

    @staticmethod
    def _decorate(canvas, doc):
        page = canvas.getPageNumber()
        canvas.saveState()
        if page > 1:
            canvas.setStrokeColor(LINE_COLOR)
            canvas.setLineWidth(0.4)
            canvas.line(LEFT, PAGE_H - 11 * mm, PAGE_W - RIGHT, PAGE_H - 11 * mm)
            canvas.setFont(FONT_REGULAR, 6.8)
            canvas.setFillColor(MUTED)
            canvas.drawString(LEFT, PAGE_H - 8.5 * mm, "ROGII | PREFIXRIDGE-SGRAPH VALIDATION REVIEW")
        canvas.setStrokeColor(LINE_COLOR)
        canvas.line(LEFT, 10 * mm, PAGE_W - RIGHT, 10 * mm)
        canvas.setFont(FONT_REGULAR, 6.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(LEFT, 6.3 * mm, "2026-07-19 | fold-clean local evidence")
        canvas.drawRightString(PAGE_W - RIGHT, 6.3 * mm, f"{page}")
        canvas.restoreState()


story = []


# ---------------------------------------------------------------------------
# Cover
# ---------------------------------------------------------------------------

story += [Spacer(1, 12 * mm)]
story.append(P("R&D VALIDATION REVIEW / FINAL", style("Eyebrow", fontSize=8.5, leading=11, textColor=TEAL)))
story.append(Spacer(1, 3 * mm))
story.append(P("ROGII PrefixRidge-SGraph v1.1<br/>本地验证复盘", S_TITLE))
story.append(P("从同井 Prefix 随机效应、冻结长边图到学习型双参考边残差的完整证据链", S_SUBTITLE))
story.append(Spacer(1, 7 * mm))
story.append(callout(
    "最终裁决: STOP",
    "按原方案自己的决策树，P2a 因果门失败后应停止整条 PrefixRidge-SGraph 扩展。"
    "图表示与长边拓扑仍是有效研究资产，但 PrefixRidge 和当前学习型边残差均未形成可迁移的 GR 信息。",
    RED,
    RED_LIGHT,
))
story.append(Spacer(1, 7 * mm))
story.append(metric_cards([
    ("PrefixRidge late-prefix", "-55.5%", "relative MSE gain，实际为明显恶化", RED),
    ("冻结 SGraph Fold1", "13.998 ft", "较 anchor 改善 1.546 ft，仍远离目标", AMBER),
    ("P2a Fold1 path gain", "-0.0029 ft", "学习型残差在 untouched Fold1 轻微恶化", RED),
]))
story.append(Spacer(1, 8 * mm))
story.append(P("复盘范围", S_H2))
story.append(bullet("原始方案: 《ROGII 单模型优化策略 PrefixRidge-SGraph v1.1》，共 14 页，原裁决为 P-1 至 P3 的条件 GO。"))
story.append(bullet("本地实现: PrefixRidge 长程边网关、固定权重 global long-edge SGraph、P2a 双参考区间残差网络。"))
story.append(bullet("验证协议: physical-well folds；Folds 3/4/5 训练、Fold2 选择，Folds 2/3/4/5 重训，Fold1 一次评估。"))
story.append(bullet("本报告只总结本地 fold-clean 证据，不把生产链 LB 6.792、CP12、Gold 或 Neighbor 作为 standalone 结果。"))
story.append(Spacer(1, 7 * mm))
story.append(P("结论日期: 2026-07-19", S_NOTE))
story.append(PageBreak())


# ---------------------------------------------------------------------------
# 1. Original thesis and decision map
# ---------------------------------------------------------------------------

story.append(P("01  原方案的核心命题与预注册停止规则", S_H1))
story.append(P(
    "原方案试图同时解决三个已知问题: 每井只有一个全局监督、井间异质性高、误差随距 PS 距离持续累积。"
    "核心公式为 <b>ΔS_ij = μ0_ij + φ_ij^T a_w + ε_ij</b>；其中全局边均值 μ0 学跨井共性，"
    "同井可见 prefix 通过闭式 Ridge 求随机效应 a_w，最后由锚定 SGraph 解唯一 S 路径并使用已知 Z 转回 TVT。"
))
story.append(Spacer(1, 4 * mm))
story.append(DecisionFlow())
story.append(Spacer(1, 4 * mm))
story.append(data_table(
    ["Gate", "原实验", "原 PASS 门", "本地状态"],
    [
        ["P0a", "true S edge -> graph -> 1-ft", "pooled <=0.20；p95 <=0.50；0 failure", "PASS"],
        ["P0.5", "early-prefix fit -> late-prefix", "MSE >=10%；>=60% 井同向；优于 shuffle", "FAIL"],
        ["P1", "冻结特征 + analytic ridge -> suffix", "edge MSE >=8%；path >=0.15 ft", "Ridge FAIL；global edge 有限有效"],
        ["P2a", "learned edge trunk + ridge-only + controls", "path >=0.30；edge >=12%；因果对照成立", "FAIL"],
        ["P2b/P3", "token、precision、端到端 graph", "仅在 P2a PASS 后允许", "未执行"],
    ],
    widths=[16 * mm, 51 * mm, 62 * mm, CONTENT_W - 129 * mm],
    row_colors=[GREEN_LIGHT, RED_LIGHT, AMBER_LIGHT, RED_LIGHT, LIGHT],
))
story.append(Spacer(1, 5 * mm))
story.append(callout(
    "审计原则",
    "本次没有在看到 Fold1 后修改模型、剂量或图权重。P2a 的 Fold2 资源门只差 0.00268 ft，"
    "因此按预先声明的灰区规则完成一次 Fold1 诊断，但原 0.10 ft 门仍保留为失败。",
    BLUE,
    colors.HexColor("#EAF0F7"),
))
story.append(PageBreak())


# ---------------------------------------------------------------------------
# 2. P0.5 PrefixRidge
# ---------------------------------------------------------------------------

story.append(P("02  P0.5/P1: PrefixRidge 同井随机效应被直接证伪", S_H1))
story.append(P(
    "实现严格遵循原方案的低成本因果门: 全局 LightGBM 边均值只使用 outer-train；同井 prefix 分成 early context 和 late query；"
    "Ridge 使用固定 λ 网格、float64 解析解，并与 no-adapter、label shuffle 和 matched other-well 对照。"
))
story.append(Spacer(1, 4 * mm))
story.append(data_table(
    ["指标", "原门槛", "实测", "判决"],
    [
        ["late-prefix relative MSE gain", ">= +10%", "-55.53%", "明显恶化"],
        ["late-prefix 正向井比例", ">= 60%", "36.1%", "失败"],
        ["suffix >=1000 ft relative MSE gain", ">= +8%", "-3.86%", "恶化"],
        ["远端正向井比例", ">= 60%", "36.1%", "失败"],
        ["Fold2 选中 Ridge λ", "应形成非零适配", "1000", "几乎关闭适配"],
    ],
    widths=[59 * mm, 34 * mm, 31 * mm, CONTENT_W - 124 * mm],
    align_right=[1, 2],
    row_colors=[RED_LIGHT, RED_LIGHT, RED_LIGHT, RED_LIGHT, AMBER_LIGHT],
))
story.append(Spacer(1, 5 * mm))
story.append(P("为什么这个失败是结构性的", S_H2))
story.append(bullet("同一口井早期 prefix 学到的边残差不能迁移到晚 prefix，更不能迁移到 1000+ ft suffix。"))
story.append(bullet("最优 λ 自动收缩到 1000，说明解析器本身在用强先验把井级参数压回零，而不是发现稳定的井级随机效应。"))
story.append(bullet("same-well 虽比乱序/他井对照相对更合理，但仍显著差于 global no-adapter；这只能证明微弱身份结构，不能证明可部署收益。"))
story.append(Spacer(1, 4 * mm))
story.append(callout(
    "组件裁决: PrefixRidge STOP",
    "原方案第 13 页决策树规定 P0.5 失败即关闭同井适配假设。后续实验因此不再包含 Ridge；"
    "继续测试的只是独立的全局长边均值与 SGraph。",
    RED,
    RED_LIGHT,
))
story.append(PageBreak())


# ---------------------------------------------------------------------------
# 3. Global edge and SGraph
# ---------------------------------------------------------------------------

story.append(P("03  全局长边均值与 SGraph: 表示成立，预测能力有限", S_H1))
story.append(P(
    "PrefixRidge 失败后，保留 global edge mean，冻结 32 ft 节点、32/64/128/256/512/1024/2048 ft skip edges、"
    "anchor-star 与 Fold2 标定的 inverse-variance 权重。Fold2 只选择图拓扑，Fold1 一次应用。"
))
story.append(Spacer(1, 3 * mm))
story.append(data_table(
    ["边距离", "global edge balanced RMSE", "相对 zero-edge MSE 结论"],
    [
        ["0-500 ft", "2.458 ft", "强信号"],
        ["500-1000 ft", "4.651 ft", "有效但开始衰减"],
        ["1000-2000 ft", "6.790 ft", "仍有结构信号"],
        [">2000 ft", "9.436 ft", "误差显著，仍优于 zero"],
    ],
    widths=[39 * mm, 55 * mm, CONTENT_W - 94 * mm],
    align_right=[1],
))
story.append(Spacer(1, 5 * mm))
story.append(P("Fold1 路径结果", S_H2))
story.append(HorizontalBars([
    ("Graph oracle (truth edges)", 0.089683, GREEN),
    ("multiscale-star", 13.997579, AMBER),
    ("anchor carry-forward", 15.543113, BLUE),
    ("chain-only", 48.641002, RED),
], max_value=50.0))
story.append(Spacer(1, 3 * mm))
story.append(data_table(
    ["路径", "overall", "0-500 ft", ">2000 ft", "结论"],
    [
        ["anchor", "15.543", "5.864", "17.640", "基线"],
        ["chain-only", "48.641", "5.383", "60.935", "误差累积崩溃"],
        ["multiscale-star", "13.998", "3.433", "16.543", "较 anchor 改善 1.546 ft"],
        ["truth-edge oracle", "0.090", "0.044", "0.093", "图与插值实现正确"],
    ],
    widths=[36 * mm, 28 * mm, 28 * mm, 28 * mm, CONTENT_W - 120 * mm],
    align_right=[1, 2, 3],
    row_colors=[LIGHT, RED_LIGHT, AMBER_LIGHT, GREEN_LIGHT],
))
story.append(Spacer(1, 4 * mm))
story.append(callout(
    "保留结论",
    "长边与 anchor-star 是必要结构: 它们阻止 chain-only 在远端灾难性累积；truth-edge oracle 0.09 ft 证明图求解器没有问题。"
    "但 global edge 仍以几何特征为主，13.998 ft 只是一项有限改进，不能支持 P5 或 5.5 ft 目标。",
    AMBER,
    AMBER_LIGHT,
))
story.append(PageBreak())


# ---------------------------------------------------------------------------
# 4. P2a design and Fold2
# ---------------------------------------------------------------------------

story.append(P("04  P2a: 双参考学习型区间残差的设计与 Fold2 灰区", S_H1))
story.append(P(
    "P2a 试图补上 global edge 的输入缺口。完整井网络使用 native horizontal GR、合法 geometry/known mask、"
    "calibrated typewell GR 与 visible heel GR；约 800 ft 感受野的多膨胀 TCN 在 32 ft 图节点计算匹配摘要，"
    "边表示包含端点、区间均值及 25%/50%/75% 顺序采样。网络只学习 cross-fitted global edge 的 bounded residual，"
    "每条边修正硬帽为 ±24 ft，图保持完全冻结。"
))
story.append(Spacer(1, 4 * mm))
story.append(data_table(
    ["项目", "冻结设置"],
    [
        ["样本单位", "一口完整物理井；不使用 Neighbor / CP12 / Gold"],
        ["模型", "212,610 参数；horizontal pool=4；embedding=32"],
        ["Fold2 协议", "Folds 3/4/5 训练，Fold2 选 epoch；early stopping"],
        ["最终协议", "Folds 2/3/4/5 从零重训固定 epoch；Fold1 一次评估"],
        ["因果对照", "no_gr、shuffled_typewell、shuffled_heel"],
    ],
    widths=[42 * mm, CONTENT_W - 42 * mm],
))
story.append(Spacer(1, 5 * mm))
story.append(metric_cards([
    ("Fold2 base", "14.9087 ft", "冻结 multiscale-star", BLUE),
    ("Fold2 selected", "14.8113 ft", "epoch 3", TEAL),
    ("Fold2 gain", "+0.0973 ft", "原门 >=0.10，差 0.00268", AMBER),
]))
story.append(Spacer(1, 5 * mm))
story.append(callout(
    "为什么仍跑 Fold1",
    "0.0973 ft 是 0.10 ft 资源门的 97.3%，继续成本很低，且 epoch 已完全由 Fold2 冻结。"
    "因此追加一次不调参 Fold1 诊断，但保留原 Fold2 门为 FAIL；Fold1 不参与任何选择。",
    AMBER,
    AMBER_LIGHT,
))
story.append(PageBreak())


# ---------------------------------------------------------------------------
# 5. P2a Fold1
# ---------------------------------------------------------------------------

story.append(P("05  P2a untouched Fold1: 边级、路径级和因果门同时失败", S_H1))
story.append(P("Fold1 不是“增益被图吃掉”，而是神经支路在边级已经没有可迁移信号。"))
story.append(Spacer(1, 3 * mm))
story.append(data_table(
    ["边子集", "edges", "base RMSE", "candidate RMSE", "relative MSE gain"],
    [
        ["suffix_all", "169,147", "8.4097", "8.4113", "-0.038%"],
        ["length >=1000", "51,142", "13.3551", "13.3572", "-0.033%"],
        ["length >=2000", "27,716", "15.3613", "15.3637", "-0.030%"],
        ["end >2000", "111,782", "9.4364", "9.4384", "-0.043%"],
    ],
    widths=[37 * mm, 25 * mm, 31 * mm, 35 * mm, CONTENT_W - 128 * mm],
    align_right=[1, 2, 3, 4],
    row_colors=[RED_LIGHT, RED_LIGHT, RED_LIGHT, RED_LIGHT],
))
story.append(Spacer(1, 5 * mm))
story.append(data_table(
    ["路径指标", "冻结 base", "P2a full", "变化"],
    [
        ["overall RMSE", "13.99758", "14.00049", "-0.00291 ft gain"],
        ["near 0-500", "3.43271", "3.43254", "+0.00017 ft gain"],
        ["mid 1000-2000", "11.50760", "11.50860", "-0.00100 ft gain"],
        ["far >2000", "16.54270", "16.54667", "-0.00397 ft gain"],
        ["远端正向井比例", "-", "50.98%", "近似随机"],
    ],
    widths=[44 * mm, 35 * mm, 35 * mm, CONTENT_W - 114 * mm],
    align_right=[1, 2, 3],
    row_colors=[RED_LIGHT, GREEN_LIGHT, RED_LIGHT, RED_LIGHT, RED_LIGHT],
))
story.append(Spacer(1, 5 * mm))
story.append(P("因果对照", S_H2))
story.append(data_table(
    ["模式", "RMSE", "相对 full 退化", "mean |edge move|", "解释"],
    [
        ["full", "14.00049", "0", "0.01696 ft", "几乎零修正"],
        ["no_gr", "13.99964", "-0.00086", "0.01453 ft", "去掉 GR 反而略好"],
        ["shuffled_typewell", "14.00060", "+0.00010", "0.01694 ft", "基本无影响"],
        ["shuffled_heel", "14.00051", "+0.00001", "0.01696 ft", "完全无影响"],
    ],
    widths=[35 * mm, 27 * mm, 33 * mm, 34 * mm, CONTENT_W - 129 * mm],
    align_right=[1, 2, 3],
    row_colors=[LIGHT, RED_LIGHT, RED_LIGHT, RED_LIGHT],
))
story.append(Spacer(1, 4 * mm))
story.append(callout(
    "P2a 裁决: FAIL",
    "Fold1 边级 relative MSE 为负，路径轻微恶化，远端正井比例接近 50%，三个干预没有因果退化。"
    "网络实际退化为平均 0.017 ft 的近零修正；Fold2 的 +0.097 ft 是不可迁移的小偏差。",
    RED,
    RED_LIGHT,
))
story.append(PageBreak())


# ---------------------------------------------------------------------------
# 6. Component matrix and causal diagnosis
# ---------------------------------------------------------------------------

story.append(P("06  组件级裁决: 什么被证伪，什么仍然成立", S_H1))
story.append(data_table(
    ["组件/命题", "证据", "裁决", "后续处置"],
    [
        ["锚定 S 表示", "truth-edge graph oracle 0.090 ft", "成立", "保留"],
        ["32 ft multiscale-star 拓扑", "Fold1 13.998 vs anchor 15.543；chain 48.641", "有限成立", "可作固定结构资产"],
        ["global geometry edge mean", "各距离桶均优于 zero；>2000 RMSE 9.436", "成立但弱", "保留诊断基线"],
        ["PrefixRidge 同井随机效应", "late-prefix -55.5%；远端 -3.86%", "证伪", "永久关闭"],
        ["当前 P2a GR 残差学习", "Fold1 edge/path 负增益；controls 无退化", "证伪", "关闭，不做 P2b"],
        ["learned precision", "P2a 未通过，且无有效均值可校准", "未授权", "不执行"],
        ["可微 SGraph path loss", "输入残差无因果信号", "不值得", "不执行"],
        ["所有 GR matching", "P2a 没有显式 correspondence loss", "未被证伪", "转向显式检索监督"],
    ],
    widths=[42 * mm, 63 * mm, 25 * mm, CONTENT_W - 130 * mm],
    row_colors=[GREEN_LIGHT, AMBER_LIGHT, GREEN_LIGHT, RED_LIGHT, RED_LIGHT, LIGHT, LIGHT, TEAL_LIGHT],
))
story.append(Spacer(1, 5 * mm))
story.append(P("根因诊断", S_H2))
story.append(bullet("边残差目标存在一条更容易的几何捷径。global 模型的重要性主要由 ΔZ、edge length、方向与端点几何主导。"))
story.append(bullet("P2a 虽然输入了 typewell/heel，但没有直接要求编码器识别正确 TVT 对应；三轮训练后，最安全的解是把残差头压到零。"))
story.append(bullet("因此当前失败不是图求解错误，也不能推出 GR 本身没有信息；它证明的是“仅靠最终边残差监督，网络不会自动学出可迁移 GR matching”。"))
story.append(bullet("在输入没有形成因果匹配信号前，增加参数、token attention、learned precision 或可微图损失只会放大噪声。"))
story.append(PageBreak())


# ---------------------------------------------------------------------------
# 7. Final decision and next stage
# ---------------------------------------------------------------------------

story.append(P("07  最终结论与下一阶段边界", S_H1))
story.append(callout(
    "路线级结论",
    "PrefixRidge-SGraph v1.1 按原预注册规则完成了低成本证伪。P0a 证明图表示正确；P0.5 否定同井 Ridge；"
    "冻结长边图只提供有限几何改善；P2a 在 untouched Fold1 的边、路径与因果三层全部失败。"
    "因此不进入 P2b、learned precision、端到端 SGraph 或更大 backbone。",
    RED,
    RED_LIGHT,
))
story.append(Spacer(1, 6 * mm))
story.append(P("对 5.5 ft standalone 目标的含义", S_H2))
story.append(bullet("当前最强证据不是“图不够强”，而是 reference matching 没有被最终轨迹损失强制学会。"))
story.append(bullet("13.998 ft 的 global SGraph 仍远高于旧 S3 9.687 ft 和 5.5 ft 目标；它不能作为新主干继续扩张。"))
story.append(bullet("下一实验必须先把 GR 对应关系作为独立任务钉死，再谈完整路径；否则仍会重复 geometry shortcut。"))
story.append(Spacer(1, 5 * mm))
story.append(P("下一步: 显式真实对应关系预训练", S_H2))
story.append(data_table(
    ["阶段", "设计", "晋级门"],
    [
        ["R0 数据", "visible prefix 的 horizontal GR <-> typewell TVT 真对应；按物理井分折", "覆盖与坐标误差审计通过"],
        ["R1 对比训练", "同井 ±24/48 ft alias hard negatives + 跨井 negatives", "held-out suffix Top8 within5 >=70%"],
        ["R2 井内适配", "只用验证井可见 prefix 做合法自校准；不看 suffix truth", "median rank <=8；alias24 win >=72%"],
        ["R3 因果门", "no_gr / shuffled_typewell / wrong-prefix", "Top8 分别产生明确下降"],
        ["R4 路径", "仅当检索门通过后接固定 SGraph/CRF", "先固定权重，不学习 precision"],
    ],
    widths=[20 * mm, 98 * mm, CONTENT_W - 118 * mm],
    row_colors=[LIGHT, TEAL_LIGHT, TEAL_LIGHT, AMBER_LIGHT, LIGHT],
))
story.append(Spacer(1, 6 * mm))
story.append(callout(
    "执行纪律",
    "保留 production 6.792 链作为独立安全端点。standalone 研究不再使用 CP12、Gold、Neighbor 或 hidden LB 反馈；"
    "新 matcher 未通过独立检索门前，不训练新的全路径网络。",
    TEAL,
    TEAL_LIGHT,
))
story.append(Spacer(1, 8 * mm))
story.append(P("主要本地产物", S_H2))
for artifact in [
    "prefixridge_longrange_gateway_v0",
    "global_longedge_sgraph_nested_gateway_v0",
    "learned_interval_sgraph_p2a_onefold_v0",
]:
    story.append(P(f"/root/autodl-tmp/oof_exports/{artifact}", style("Artifact" + artifact[-2:], fontSize=7.4, leading=10.5, textColor=MUTED)))
story.append(Spacer(1, 5 * mm))
story.append(P(
    "口径说明: 所有 RMSE 为 pooled row-level ft；gain = base RMSE - candidate RMSE。"
    "relative MSE gain 为正表示改善，为负表示恶化。数值来自本轮用户提供的本地日志与对应 verdict。",
    S_NOTE,
))


doc = ReviewDocTemplate(str(OUT_PATH))
doc.build(story)
print(OUT_PATH)
