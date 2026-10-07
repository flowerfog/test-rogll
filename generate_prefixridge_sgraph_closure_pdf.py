from pathlib import Path
import math

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, Rect, String
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "pdf"
TMP_DIR = ROOT / "tmp" / "pdfs" / "prefixridge_closure"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT_DIR / "ROGII_PrefixRidge_SGraph_v1_1_validation_closure_2026-07-19.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")

pdfmetrics.registerFont(TTFont("CN", str(FONT_PATH)))
pdfmetrics.registerFont(TTFont("CN-Bold", str(FONT_PATH)))


INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#59636E")
TEAL = colors.HexColor("#007F73")
TEAL_LIGHT = colors.HexColor("#E7F4F1")
RED = colors.HexColor("#B4232C")
RED_LIGHT = colors.HexColor("#FBEAEC")
AMBER = colors.HexColor("#A35A00")
AMBER_LIGHT = colors.HexColor("#FFF3DD")
GREEN = colors.HexColor("#247A48")
GREEN_LIGHT = colors.HexColor("#EAF5ED")
BLUE = colors.HexColor("#245A92")
BLUE_LIGHT = colors.HexColor("#EAF1F8")
LINE = colors.HexColor("#D9DEE3")
PAPER = colors.white
SOFT = colors.HexColor("#F5F7F8")


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="CNBody",
        fontName="CN",
        fontSize=9.2,
        leading=14,
        textColor=INK,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="CNSmall",
        fontName="CN",
        fontSize=7.6,
        leading=10.5,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        name="CNTitle",
        fontName="CN-Bold",
        fontSize=25,
        leading=31,
        textColor=INK,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="CNSubtitle",
        fontName="CN",
        fontSize=11,
        leading=17,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        name="CNH1",
        fontName="CN-Bold",
        fontSize=16,
        leading=21,
        textColor=INK,
        spaceBefore=3,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="CNH2",
        fontName="CN-Bold",
        fontSize=11.5,
        leading=16,
        textColor=TEAL,
        spaceBefore=7,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="CNCallout",
        fontName="CN-Bold",
        fontSize=10.5,
        leading=16,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        name="CNCenter",
        fontName="CN",
        fontSize=8.2,
        leading=11,
        alignment=TA_CENTER,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        name="CNTableHeader",
        fontName="CN-Bold",
        fontSize=7.4,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.white,
    )
)


ROWS = 757_738


def sse(rmse, rows=ROWS):
    return float(rmse) ** 2 * int(rows)


PATH_METRICS = [
    ("Anchor / carry-forward", 15.543113008659821, "deployable baseline"),
    ("Fixed-weight SGraph", 13.997578864404042, "deployable diagnostic"),
    ("Learned interval SGraph", 14.000491342153365, "deployable diagnostic"),
    ("Matcher + nested DP", 15.21833122035433, "deployable diagnostic"),
    ("Dense Viterbi", 19.441736966232554, "deployable selector"),
    ("K64 connected oracle", 6.081928669959935, "truth-only oracle"),
    ("B192 mode oracle", 5.284353996394837, "truth-only, 193-search oracle"),
    ("Common-energy selector", 19.441736966232554, "deployable selector"),
]


def p(text, style="CNBody"):
    return Paragraph(text, styles[style])


def status_cell(label):
    mapping = {
        "PASS": (GREEN_LIGHT, GREEN),
        "PARTIAL": (AMBER_LIGHT, AMBER),
        "FAIL": (RED_LIGHT, RED),
        "NOT RECORDED": (SOFT, MUTED),
        "STOP": (RED_LIGHT, RED),
    }
    background, foreground = mapping[label]
    cell = Table([[p(label, "CNCenter")]], colWidths=[25 * mm], rowHeights=[7 * mm])
    cell.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("TEXTCOLOR", (0, 0), (-1, -1), foreground),
                ("BOX", (0, 0), (-1, -1), 0.4, foreground),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    return cell


def data_table(headers, rows, widths, font_size=7.5, header_bg=INK):
    data = [[p(str(value), "CNTableHeader") for value in headers]]
    for row in rows:
        converted = []
        for value in row:
            if isinstance(value, Table):
                converted.append(value)
            else:
                converted.append(p(str(value), "CNSmall"))
        data.append(converted)
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "CN-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), SOFT))
    table.setStyle(TableStyle(commands))
    return table


def callout(title, body, kind="info"):
    palette = {
        "info": (BLUE_LIGHT, BLUE),
        "warn": (AMBER_LIGHT, AMBER),
        "fail": (RED_LIGHT, RED),
        "pass": (GREEN_LIGHT, GREEN),
    }
    background, border = palette[kind]
    table = Table(
        [[p(title, "CNCallout")], [p(body, "CNBody")]],
        colWidths=[174 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("LINEBEFORE", (0, 0), (0, -1), 3, border),
                ("BOX", (0, 0), (-1, -1), 0.35, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def metric_cards(items):
    cells = []
    for value, label, color in items:
        content = Table(
            [[p(value, "CNH1")], [p(label, "CNSmall")]],
            colWidths=[41 * mm],
        )
        content.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("LINEABOVE", (0, 0), (-1, 0), 3, color),
                    ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        cells.append(content)
    table = Table([cells], colWidths=[43.5 * mm] * len(cells), hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _chart_axes(drawing, left, bottom, width, height, y_ticks, y_max):
    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=MUTED, strokeWidth=0.7))
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=MUTED, strokeWidth=0.7))
    for value in y_ticks:
        y = bottom + height * value / y_max
        drawing.add(Line(left, y, left + width, y, strokeColor=LINE, strokeWidth=0.45))
        drawing.add(String(left - 7, y - 2.5, str(value), fontName="CN", fontSize=6.5, textAnchor="end", fillColor=MUTED))


def make_charts():
    labels = ["Anchor", "Fixed SG", "Learned SG", "Nested DP", "Dense Vit", "B192 oracle", "Energy sel"]
    values = [15.5431, 13.9976, 14.0005, 15.2183, 19.4417, 5.2844, 19.4417]
    bar_colors = [MUTED, TEAL, TEAL, BLUE, RED, AMBER, RED]
    path = Drawing(493, 190)
    left, bottom, width, height = 38, 34, 445, 135
    _chart_axes(path, left, bottom, width, height, [0, 5, 10, 15, 20], 21.5)
    bar_slot = width / len(values)
    bar_width = bar_slot * 0.58
    for index, (label, value, fill) in enumerate(zip(labels, values, bar_colors)):
        x = left + index * bar_slot + (bar_slot - bar_width) / 2
        bar_height = height * value / 21.5
        path.add(Rect(x, bottom, bar_width, bar_height, fillColor=fill, strokeColor=None))
        path.add(String(x + bar_width / 2, bottom - 12, label, fontName="CN", fontSize=6.3, textAnchor="middle", fillColor=INK))
        path.add(String(x + bar_width / 2, bottom + bar_height + 4, f"{value:.2f}", fontName="CN", fontSize=6.5, textAnchor="middle", fillColor=INK))
    target_y = bottom + height * 5.5 / 21.5
    path.add(Line(left, target_y, left + width, target_y, strokeColor=GREEN, strokeWidth=1.2, strokeDashArray=[4, 3]))
    path.add(String(left + 3, target_y + 4, "5.5 ft target", fontName="CN", fontSize=6.5, fillColor=GREEN))
    path.add(String(4, bottom + height / 2, "RMSE (ft)", fontName="CN", fontSize=7, fillColor=MUTED))

    budgets = [64, 96, 128, 160, 192]
    rmse_values = [6.003164, 5.629252, 5.388286, 5.321888, 5.284354]
    oracle = Drawing(493, 180)
    left, bottom, width, height = 42, 32, 432, 125
    y_min, y_max = 4.8, 6.2
    oracle.add(Line(left, bottom, left, bottom + height, strokeColor=MUTED, strokeWidth=0.7))
    oracle.add(Line(left, bottom, left + width, bottom, strokeColor=MUTED, strokeWidth=0.7))
    for value in [4.8, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2]:
        y = bottom + height * (value - y_min) / (y_max - y_min)
        oracle.add(Line(left, y, left + width, y, strokeColor=LINE, strokeWidth=0.45))
        oracle.add(String(left - 7, y - 2.5, f"{value:.1f}", fontName="CN", fontSize=6.2, textAnchor="end", fillColor=MUTED))
    points = []
    for index, (budget, value) in enumerate(zip(budgets, rmse_values)):
        x = left + width * index / (len(budgets) - 1)
        y = bottom + height * (value - y_min) / (y_max - y_min)
        points.extend([x, y])
        oracle.add(Circle(x, y, 3.2, fillColor=TEAL, strokeColor=colors.white, strokeWidth=0.7))
        oracle.add(String(x, y + 8, f"{value:.3f}", fontName="CN", fontSize=6.7, textAnchor="middle", fillColor=INK))
        oracle.add(String(x, bottom - 12, str(budget), fontName="CN", fontSize=6.7, textAnchor="middle", fillColor=INK))
    oracle.add(PolyLine(points, strokeColor=TEAL, strokeWidth=2.0, fillColor=None))
    for gate, color, dash, label in [(5.5, GREEN, [4, 3], "minimum 5.5"), (5.0, RED, [2, 2], "strong 5.0")]:
        y = bottom + height * (gate - y_min) / (y_max - y_min)
        oracle.add(Line(left, y, left + width, y, strokeColor=color, strokeWidth=1.0, strokeDashArray=dash))
        oracle.add(String(left + 4, y + 3, label, fontName="CN", fontSize=6.3, fillColor=color))
    oracle.add(String(left + width / 2, 5, "Whole-well mode budget", fontName="CN", fontSize=7, textAnchor="middle", fillColor=MUTED))

    strata = ["0-500", "500-1000", "1000-2000", ">2000"]
    series = {
        "Anchor": ([5.8638, 11.0970, 14.2472, 17.6398], MUTED),
        "Fixed SGraph": ([3.4327, 7.5395, 11.5076, 16.5427], TEAL),
        "B192 oracle": ([4.5363, 4.9760, 4.6944, 5.6363], AMBER),
        "Energy selector": ([6.1548, 12.6321, 16.7761, 22.5332], RED),
    }
    distance = Drawing(493, 195)
    left, bottom, width, height = 42, 34, 430, 132
    _chart_axes(distance, left, bottom, width, height, [0, 5, 10, 15, 20, 25], 25)
    x_values = [left + width * index / 3 for index in range(4)]
    for (label, (values, color)), legend_y in zip(series.items(), [184, 174, 184, 174]):
        points = []
        for x, value in zip(x_values, values):
            y = bottom + height * value / 25
            points.extend([x, y])
            distance.add(Circle(x, y, 2.7, fillColor=color, strokeColor=colors.white, strokeWidth=0.6))
        distance.add(PolyLine(points, strokeColor=color, strokeWidth=1.8, fillColor=None))
    for x, label in zip(x_values, strata):
        distance.add(String(x, bottom - 12, label, fontName="CN", fontSize=6.7, textAnchor="middle", fillColor=INK))
    legend_positions = [(42, 184), (160, 184), (285, 184), (390, 184)]
    for (label, (_, color)), (x, y) in zip(series.items(), legend_positions):
        distance.add(Line(x, y, x + 14, y, strokeColor=color, strokeWidth=2))
        distance.add(String(x + 18, y - 2.5, label, fontName="CN", fontSize=6.5, fillColor=INK))
    target_y = bottom + height * 5.5 / 25
    distance.add(Line(left, target_y, left + width, target_y, strokeColor=GREEN, strokeWidth=0.9, strokeDashArray=[4, 3]))
    return path, oracle, distance


class ClosureDocTemplate(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=17 * mm,
            title="ROGII PrefixRidge-SGraph v1.1 validation closure",
            author="ROGII experiment audit",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._decorate))

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, 13 * mm, 192 * mm, 13 * mm)
        canvas.setFont("CN", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 8.5 * mm, "ROGII | PrefixRidge-SGraph v1.1 validation closure")
        canvas.drawRightString(192 * mm, 8.5 * mm, f"{doc.page}")
        canvas.restoreState()


def build_story(path_chart, oracle_chart, distance_chart):
    story = []

    story.extend(
        [
            Spacer(1, 13 * mm),
            p("R&D VALIDATION CLOSURE / 2026-07-19", "CNSmall"),
            Spacer(1, 3 * mm),
            p("ROGII PrefixRidge-SGraph v1.1", "CNTitle"),
            p("本地验证总结、证伪边界与交付检查清单", "CNSubtitle"),
            Spacer(1, 9 * mm),
            metric_cards(
                [
                    ("13.998 ft", "最优可部署 SGraph Fold1", TEAL),
                    ("5.284 ft", "B192 truth-only oracle", AMBER),
                    ("19.442 ft", "无标签全局选择器", RED),
                    ("STOP", "PDF 主路线最终裁决", RED),
                ]
            ),
            Spacer(1, 10 * mm),
            callout(
                "最终裁决：STOP / 方案级证伪",
                "原 PDF 的可执行链条 PrefixRidge -> learned interval edge -> anchored SGraph -> dense matcher/DP -> global mode scorer 未形成 <=5.5 ft 的可部署单模型。B192 oracle 虽为 5.284 ft，但它使用每井最多 193 条 truth-selected 搜索路径，违反 test search=0，且无标签选择器为 19.442 ft。该 oracle 只能证明局部/候选信息存在，不能作为交付模型。",
                "fail",
            ),
            Spacer(1, 7 * mm),
            p("证伪范围", "CNH2"),
            p(
                "被证伪的是本 PDF 的 PrefixRidge-SGraph 机制和后续 matcher-posterior 组织方式，不是所有 GR matching，也不是“未知的完整井模型可以达到 5.x”这一外部事实主张。",
                "CNBody",
            ),
            Spacer(1, 6 * mm),
            p("审计口径", "CNH2"),
            p(
                "主验证为 untouched Fold1、155 口井、757,738 行、physical-well split。所有 oracle 均明确标为 truth-only，不与可部署结果混排。exact SSE 在原始逐行数组未随交付提供时，使用完整精度 RMSE^2 x row_count 重建并标记为 reconstructed。",
                "CNBody",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("01  原方案命题与验证边界", "CNH1"),
            p(
                "原 PDF 的核心命题是：跨井共享局部区间 Delta S 规律，同井可见 prefix 通过闭式 Ridge 估计低维随机效应，再由多尺度 SGraph 用短边保证连通、长边治理远端漂移。输出必须是一个 checkpoint、一次确定性 forward、测试井无 optimizer/search。",
                "CNBody",
            ),
            data_table(
                ["原阶段", "目的", "预注册晋级门", "本次状态"],
                [
                    ("P-1", "重放口径", "row IDs / SSE / anchor / pooled 一致", status_cell("PARTIAL")),
                    ("P0a", "grid + graph oracle", "pooled <=0.20; p95 <=0.50; 0 solve failure", status_cell("PARTIAL")),
                    ("P0.5", "same-well PrefixRidge", ">=10% edge MSE; >=60% wells; matched controls", status_cell("FAIL")),
                    ("P1", "冻结特征 + analytic ridge", "edge >=8%; path >=0.15 ft", status_cell("FAIL")),
                    ("P2a", "learned edge trunk + ridge-only", "path >=0.30; edge >=12%; controls", status_cell("FAIL")),
                    ("P3a", "multiscale SGraph", "1000+ >=0.50; worst-20 >=1.0; near <=0.10", status_cell("FAIL")),
                    ("P5", "完整 Fold1", "<=9.0 且优于最强 legal standalone >=0.50", status_cell("FAIL")),
                    ("P6+", "三折和 add-on", "P5 通过后才允许", status_cell("STOP")),
                ],
                [18 * mm, 47 * mm, 77 * mm, 30 * mm],
            ),
            Spacer(1, 6 * mm),
            callout(
                "执行偏离说明",
                "P0.5/P1 已按原决策树失败，本应立即关闭。后续 SGraph、matcher、dense posterior 和 mode-bank 实验属于额外的上限审计，用于判断是否遗漏了可组织的全局 GR 模式；它们不能反向把已失败的 PrefixRidge 门改判为 PASS。",
                "warn",
            ),
            Spacer(1, 6 * mm),
            path_chart,
            p("图 1. 可部署路径与 truth-only oracle 的 RMSE 不可混为同一性能序列。", "CNSmall"),
            PageBreak(),
        ]
    )

    timeline_rows = [
        ("PrefixRidge P0.5/P1", "late gain -55.5%; far gain -3.86%; 36.1% wells positive", "same/shuffle/other", "FAIL"),
        ("Fixed SGraph", "13.9976 ft; far 16.5427; oracle 0.0897", "chain / multiscale / oracle", "FAIL P5"),
        ("Learned interval P2a", "14.0005 ft; path gain -0.0029; edge gain -0.033%", "no-GR / shuffled TW / shuffled heel", "FAIL"),
        ("Real matcher + DP", "top8 57.49%; DP 15.2183 ft", "NCC / no-GR / shuffled TW", "FAIL"),
        ("Prefix adaptation", "top8 +0.90 pp; deep top8 -0.80 pp; argmax 32.72", "wrong prefix / no-GR / shuffled TW", "FAIL"),
        ("Dense posterior Viterbi", "19.4417 ft vs anchor 15.5431", "frozen/adapted x lambda grid", "FAIL"),
        ("K64 path ceiling", "local oracle 4.1002; connected oracle 6.0819", "full-state integrity 0.2515", "MENU FAIL"),
        ("Global mode bank", "B192 oracle 5.2844; selector 19.4417; saturated", "64-192 path budget", "STOP"),
    ]
    story.extend(
        [
            p("02  实验证据链", "CNH1"),
            data_table(
                ["实验", "核心结果", "matched/control", "裁决"],
                timeline_rows,
                [35 * mm, 74 * mm, 45 * mm, 20 * mm],
            ),
            Spacer(1, 7 * mm),
            p("为什么最优可部署结果仍不够", "CNH2"),
            p(
                "Fixed SGraph 将 anchor 15.543 降到 13.998，证明长边图能减少部分累积漂移；但它仍远高于 P5 的 9.0 ft 门。learned interval correction 只有约 0.017 ft 平均边移动，Fold1 path 反而恶化 0.0029 ft，说明模型没有学到可迁移的长边残差。",
                "CNBody",
            ),
            p("PrefixRidge 因果门", "CNH2"),
            data_table(
                ["子集 / 变体", "Global", "Same-well", "Shuffle", "Other-well", "结论"],
                [
                    ("late-prefix balanced RMSE", "2.153", "2.685", "3.097", "2.838", "Same 不优于 global"),
                    ("suffix >=1000 balanced RMSE", "8.914", "9.084", "9.216", "9.279", "仅弱于 controls"),
                    ("positive well fraction", "-", "36.1%", "-", "-", "低于 60% 门"),
                ],
                [46 * mm, 24 * mm, 27 * mm, 24 * mm, 27 * mm, 26 * mm],
            ),
            Spacer(1, 6 * mm),
            callout(
                "P0.5/P1 结论",
                "Same-well 统计优于 matched shuffle/other-well，说明井内标签含有弱因果信息；但它比 outer-train global mean 更差，且 suffix 和 late-prefix 均未过增益/正井比例门。弱信号存在不等于 PrefixRidge 可作为主路径。",
                "fail",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("03  距离漂移、SGraph 与 learned interval", "CNH1"),
            distance_chart,
            p("图 2. Fixed SGraph 改善近中段，但远端仍为 16.54 ft；B192 是 truth-only 上限。", "CNSmall"),
            Spacer(1, 5 * mm),
            data_table(
                ["方法", "0-500", "500-1000", "1000-2000", ">2000", "Pooled"],
                [
                    ("Anchor", "5.864", "11.097", "14.247", "17.640", "15.543"),
                    ("Fixed SGraph", "3.433", "7.540", "11.508", "16.543", "13.998"),
                    ("Learned P2a", "3.433", "7.539", "11.509", "16.547", "14.000"),
                    ("Dense Viterbi", "6.155", "12.632", "16.776", "22.533", "19.442"),
                    ("B192 oracle", "4.536", "4.976", "4.694", "5.636", "5.284"),
                ],
                [45 * mm, 25 * mm, 28 * mm, 29 * mm, 24 * mm, 23 * mm],
            ),
            Spacer(1, 6 * mm),
            p("P2a matched interventions", "CNH2"),
            data_table(
                ["变体", "Fold1 RMSE", "相对 full 变化", "平均边修正", "因果判定"],
                [
                    ("Full", "14.00049", "0", "0.01696 ft", "主结果"),
                    ("No GR", "13.99964", "-0.00086", "0.01453 ft", "移除后反而更好"),
                    ("Shuffled typewell", "14.00060", "+0.00010", "0.01694 ft", "无实质退化"),
                    ("Shuffled heel", "14.00051", "+0.00001", "0.01696 ft", "无实质退化"),
                ],
                [42 * mm, 29 * mm, 35 * mm, 34 * mm, 34 * mm],
            ),
            Spacer(1, 6 * mm),
            callout(
                "P3a/P2a 判决",
                "图求解本身有效，但 learned edge 模块未通过 matched controls，也未改善最终 path RMSE。根据原 PDF 的晋级规则，不能以 edge loss、局部子集或 oracle 结果替代最终路径门。",
                "fail",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("04  Matcher posterior：局部召回不等于全局可选", "CNH1"),
            p(
                "Real-correspondence matcher 明显优于 NCC，但 Fold1 真值中位排名仍约 19-20。Prefix analytic adaptation 在 late-prefix 上提升 top8 5.46 pp，却只在 official suffix 提升 0.90 pp，且 deep top8 从 57.69% 降至 56.90%。这是明确的 prefix-to-suffix 域移。",
                "CNBody",
            ),
            data_table(
                ["Matcher 诊断", "Frozen", "Adapted", "Control / reference", "结论"],
                [
                    ("Official top8 within 5 ft", "57.84%", "58.74%", "NCC 23.78%", "绝对召回不足"),
                    ("Median truth rank", "18", "19", "No-GR 31; shuffled TW 37", "打分尺度仍错"),
                    ("Alias24 win", "65.58%", "64.62%", "门 72%", "适配后下降"),
                    ("Argmax RMSE", "32.54", "32.72", "Shuffled TW 36.70", "不可直接解码"),
                    ("Late-prefix top8", "61.73%", "67.19%", "+5.46 pp", "局部收益不迁移"),
                ],
                [45 * mm, 27 * mm, 27 * mm, 44 * mm, 31 * mm],
            ),
            Spacer(1, 6 * mm),
            p("Dense posterior ceiling", "CNH2"),
            data_table(
                ["菜单", "Node within5", "Unconstrained oracle", "Connected oracle", "解释"],
                [
                    ("K8", "58.26%", "21.466", "22.080", "真值经常不在菜单"),
                    ("K16", "71.85%", "16.990", "17.794", "仍不足"),
                    ("K32", "85.29%", "10.433", "11.938", "远离 5.5"),
                    ("K64", "95.64%", "4.100", "6.082", "alias branch connectivity"),
                    ("Full 257", "100% axis", "0.251", "-", "实现/表示完整性通过"),
                ],
                [25 * mm, 31 * mm, 34 * mm, 31 * mm, 53 * mm],
            ),
            Spacer(1, 6 * mm),
            callout(
                "核心结构发现",
                "K64 局部无约束 oracle 为 4.100 ft，但物理连续路径 oracle 为 6.082 ft。最差 1 口井贡献 39.0% SSE，最差 5 口贡献 80.1%。正确局部状态大量存在，却没有被 posterior 分数组织成同一条可部署路径。",
                "warn",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("05  Global mode bank：上限触线，交付仍失败", "CNH1"),
            oracle_chart,
            p("图 3. Oracle 在 B128 后进入饱和；B160 -> B192 仅改善 0.038 ft。", "CNSmall"),
            Spacer(1, 5 * mm),
            data_table(
                ["Budget", "Oracle RMSE", "Median well", "p95 well", "Far >2000", "边际增益"],
                [
                    ("64", "6.003", "4.193", "10.939", "6.485", "-"),
                    ("96", "5.629", "4.051", "10.051", "5.968", "0.374"),
                    ("128", "5.388", "4.035", "9.531", "5.744", "0.241"),
                    ("160", "5.322", "4.010", "9.423", "5.749", "0.066"),
                    ("192", "5.284", "4.010", "9.423", "5.636", "0.038"),
                ],
                [24 * mm, 31 * mm, 31 * mm, 30 * mm, 30 * mm, 28 * mm],
            ),
            Spacer(1, 6 * mm),
            callout(
                "为什么 5.284 ft 仍是失败",
                "B192 使用真值在每井最多 193 条完整路径中选择最佳路径，只剩 0.216 ft 的 RMSE 余量到 5.5。无标签 common-energy 选择器为 19.442 ft。更关键的是，多轮 suppression + 8 组 transition 配置构成测试时 search，直接违反原 PDF 的 optimizer/search=0 合同。",
                "fail",
            ),
            Spacer(1, 6 * mm),
            p("SSE 预算", "CNH2"),
            p(
                "目标 5.5 ft 对应 MSE 30.25。B192 oracle MSE 27.819，仅留下约 2.431 ft^2 的选择误差预算；现有 selector MSE 约 378.0。即使训练 scorer，也必须接近 truth oracle 才能过线，和已观测选择能力相差两个数量级。",
                "CNBody",
            ),
            PageBreak(),
        ]
    )

    checklist_rows = [
        ("Fold 可追溯 outer-train", "各实验记录 train/tune/test folds；Fold1 untouched", "PASS"),
        ("Episode 可追溯", "采样位置/seed 有记录，但无统一 immutable episode manifest/hash", "PARTIAL"),
        ("Normalizer 可追溯", "代码中由 core/outer-train 拟合；未统一导出 hash 与版本", "PARTIAL"),
        ("Grid 可追溯", "32-ft nodes、257 states、Fold2 选择记录明确", "PASS"),
        ("Ridge prior 可追溯", "lambda=1000 由 Fold2；matcher adapter lambda=0.1/dose=1", "PASS"),
        ("重复 forward 逐位一致", "未执行并落盘 bitwise repeat test", "NOT RECORDED"),
        ("测试 optimizer/search=0", "analytic adapter 为 0；最终 mode bank 每井最多 193 路 search", "FAIL"),
        ("Row count", "主路径均记录 757,738 rows / 155 wells", "PASS"),
        ("Exact SSE", "本报告由 full-precision RMSE^2 x rows 重建；非逐行原始 SSE dump", "PARTIAL"),
        ("Anchor error", "anchor baseline 与硬锚代码存在；缺统一 anchor residual artifact", "PARTIAL"),
        ("Distance strata", "0-500 / 500-1000 / 1000-2000 / >2000 全部记录", "PASS"),
        ("Condition number", "仅 condition_pass 与阈值；实际 p50/p95 未进入 verdict", "PARTIAL"),
        ("Solve failure", "无观察到崩溃，但 failure count 未作为显式字段落盘", "NOT RECORDED"),
        ("Worst-20% wells", "有 p95 与 K64 tail concentration；最终 B192 worst-20 未导出", "PARTIAL"),
        ("新增模块 matched control", "Prefix/P2a 有 controls；dense mode bank 无 matched shuffle", "PARTIAL"),
        ("以 final path RMSE 晋级", "P2a/SGraph/matcher 均按 path 停止；oracle 未冒充晋级", "PASS"),
    ]
    story.extend(
        [
            p("06  原 PDF 交付检查清单审计", "CNH1"),
            p(
                "状态含义：PASS 为已由输出直接证明；PARTIAL 为代码/间接证据存在但交付字段不完整；NOT RECORDED 为没有落盘证据；FAIL 为与合同直接冲突。",
                "CNBody",
            ),
            data_table(
                ["检查项", "审计证据", "状态"],
                [(item, evidence, status_cell(status)) for item, evidence, status in checklist_rows],
                [48 * mm, 96 * mm, 30 * mm],
            ),
            Spacer(1, 7 * mm),
            callout(
                "交付合规裁决",
                "性能门失败之外，bitwise repeat、condition-number 实值、solve-failure count、最终 worst-20% 和 test search=0 也未全部满足。因此不能把当前产物登记为原 PDF 所定义的 standalone checkpoint。",
                "fail",
            ),
            PageBreak(),
        ]
    )

    sse_rows = []
    for name, rmse_value, classification in PATH_METRICS:
        sse_rows.append(
            (
                name,
                f"{ROWS:,}",
                f"{rmse_value:.6f}",
                f"{sse(rmse_value):,.3f}",
                classification,
            )
        )
    story.extend(
        [
            p("07  Row count 与 reconstructed exact SSE", "CNH1"),
            p(
                "以下 SSE 使用输出中的 full-precision RMSE 按 SSE = RMSE^2 x rows 重建。数值足以审计相对结果，但不替代原始逐行 error^2 求和文件；因此检查清单仍标为 PARTIAL。",
                "CNBody",
            ),
            data_table(
                ["路径", "Rows", "RMSE", "Reconstructed SSE", "分类"],
                sse_rows,
                [44 * mm, 25 * mm, 25 * mm, 45 * mm, 35 * mm],
            ),
            Spacer(1, 7 * mm),
            p("尾部证据", "CNH2"),
            data_table(
                ["K64 connected oracle", "剩余 wells", "剩余 RMSE", "已移除 SSE 占比"],
                [
                    ("移除 0", "155", "6.082", "0.0%"),
                    ("移除最差 1", "154", "4.771", "39.0%"),
                    ("移除最差 5", "150", "2.761", "80.1%"),
                    ("移除最差 10", "145", "1.808", "91.8%"),
                    ("移除最差 25", "130", "0.947", "97.9%"),
                    ("移除最差 40", "115", "0.563", "99.4%"),
                ],
                [52 * mm, 35 * mm, 39 * mm, 48 * mm],
            ),
            Spacer(1, 7 * mm),
            callout(
                "尾部解释",
                "中位井已经较好，但少数 alias/connectivity 失败井决定 pooled SSE。原 PDF 要求 worst-20% wells 单独晋级是正确的；当前实验未为最终 B192 路径导出完整 worst-20% 指标，因此不能声称尾部治理已完成。",
                "warn",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("08  最终决策与保留资产", "CNH1"),
            callout(
                "STOP：不再扩展本路线",
                "停止增加 mode 数、调 transition lambda、训练当前 193-path bank 的 listwise scorer、扩大 PrefixRidge 维度或引入 learned precision。上述动作要么已被最终 path RMSE 否定，要么没有足够 oracle 误差预算。",
                "fail",
            ),
            Spacer(1, 7 * mm),
            p("可以保留", "CNH2"),
            data_table(
                ["资产", "保留原因", "允许用途"],
                [
                    ("32-ft anchored SGraph solver", "representation oracle 0.0897; 数值结构有效", "后续不同 emission/edge 模型的固定解码器"),
                    ("Distance-strata harness", "稳定暴露远趾端漂移", "所有新单模型的固定报告"),
                    ("Matched prefix controls", "same/shuffle/other-well 归因清晰", "新 adaptation 模块的最低因果门"),
                    ("Dense posterior audit", "区分 node recall、connected path 和 selector", "候选系统的预训练零成本门"),
                    ("Tail concentration harness", "量化少数井 SSE 支配", "worst-20% 与 pooled 双门"),
                ],
                [50 * mm, 62 * mm, 62 * mm],
            ),
            Spacer(1, 7 * mm),
            p("必须关闭", "CNH2"),
            p(
                "PrefixRidge 作为主适配器、learned interval residual、当前 matcher posterior 的 DP/Viterbi、suppression mode-bank 与基于其能量的路径选择器。它们可以作为负结果和诊断资产保存，不再进入 5.5 主线。",
                "CNBody",
            ),
            Spacer(1, 7 * mm),
            p("下一条单模型路线的约束", "CNH2"),
            p(
                "新路线必须直接学习完整井的全局证据和连贯路径，而不是继续消费当前局部 posterior。第一道门仍应是完整 Fold1 final path RMSE，并继续保留 same-well shuffle、distance strata、condition/failure、worst-20% 和 bitwise repeat 合同。",
                "CNBody",
            ),
            Spacer(1, 8 * mm),
            callout(
                "一句话结论",
                "正确局部状态经常存在，但 PrefixRidge 和当前 posterior 无法为正确全局分支提供可迁移、可选择、零搜索的证据；因此原 PDF 方案在本地被方案级证伪。",
                "info",
            ),
            PageBreak(),
        ]
    )

    evidence_rows = [
        ("prefixridge_longrange_gateway_v0", "/root/autodl-tmp/oof_exports/prefixridge_longrange_gateway_v0", "P0.5/P1"),
        ("global_longedge_sgraph_nested_gateway_v0", "/root/autodl-tmp/oof_exports/global_longedge_sgraph_nested_gateway_v0", "fixed SGraph"),
        ("learned_interval_sgraph_p2a_onefold_v0", "/root/autodl-tmp/oof_exports/learned_interval_sgraph_p2a_onefold_v0", "P2a"),
        ("real_correspondence_multiscale_matcher_gateway_v0", "/root/autodl-tmp/oof_exports/real_correspondence_multiscale_matcher_gateway_v0", "matcher"),
        ("real_correspondence_matcher_nested_dp_audit_v1", "/root/autodl-tmp/oof_exports/real_correspondence_matcher_nested_dp_audit_v1", "nested DP"),
        ("prefix_contrastive_matcher_adaptation_gateway_v1", "/root/autodl-tmp/oof_exports/prefix_contrastive_matcher_adaptation_gateway_v1", "prefix controls"),
        ("dense_matcher_posterior_path_ceiling_audit_v0", "/root/autodl-tmp/oof_exports/dense_matcher_posterior_path_ceiling_audit_v0", "dense path"),
        ("dense_posterior_oracle_tail_decomposition_v1", "/root/autodl-tmp/oof_exports/dense_posterior_oracle_tail_decomposition_v1", "tail"),
        ("dense_posterior_global_mode_bank_audit_v2", "/root/autodl-tmp/oof_exports/dense_posterior_global_mode_bank_audit_v2", "B64"),
        ("dense_posterior_global_mode_bank_extension_v3", "/root/autodl-tmp/oof_exports/dense_posterior_global_mode_bank_extension_v3", "B192"),
    ]
    story.extend(
        [
            p("附录 A  证据清单", "CNH1"),
            data_table(
                ["Experiment", "Artifact directory", "范围"],
                evidence_rows,
                [62 * mm, 88 * mm, 24 * mm],
                font_size=6.8,
            ),
            Spacer(1, 7 * mm),
            p("本地代码交付", "CNH2"),
            p(
                "prefixridge_longrange_gateway_v0_cell.py; global_longedge_sgraph_nested_gateway_v0_cell.py; learned_interval_sgraph_p2a_onefold_v0_cell.py; prefix_contrastive_matcher_adaptation_gateway_v1_cell.py; dense_matcher_posterior_path_ceiling_audit_v0_cell.py; dense_posterior_oracle_tail_decomposition_v1_cell.py; dense_posterior_global_mode_bank_audit_v2_cell.py; dense_posterior_global_mode_bank_extension_v3_cell.py.",
                "CNSmall",
            ),
            Spacer(1, 7 * mm),
            p("限制", "CNH2"),
            p(
                "本报告根据用户提供的控制台 verdict、完整精度指标和本地实验代码编制。未同步回本机的远程 CSV/NPZ 不被假定存在于交付包中；condition-number 实值、bitwise replay、solve-failure count 和最终 B192 worst-20% 因而保持未完成状态。",
                "CNBody",
            ),
        ]
    )
    return story


def main():
    path_chart, oracle_chart, distance_chart = make_charts()
    document = ClosureDocTemplate(str(OUTPUT))
    document.build(build_story(path_chart, oracle_chart, distance_chart))
    print(OUTPUT)


if __name__ == "__main__":
    main()
