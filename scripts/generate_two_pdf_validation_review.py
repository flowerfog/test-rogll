from pathlib import Path
import math

from reportlab.graphics.shapes import Drawing, Line, Rect, String, Circle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / "ROGII_two_single_model_roadmaps_validation_review_2026-07-19.pdf"

PAGE_W, PAGE_H = A4
MARGIN_X = 16 * mm
MARGIN_TOP = 18 * mm
MARGIN_BOTTOM = 16 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

INK = HexColor("#18242B")
MUTED = HexColor("#5B6870")
TEAL = HexColor("#087E8B")
TEAL_LIGHT = HexColor("#DDEFF0")
CORAL = HexColor("#D75A4A")
CORAL_LIGHT = HexColor("#F8E6E2")
GREEN = HexColor("#2F7D5B")
GREEN_LIGHT = HexColor("#E3F1E9")
AMBER = HexColor("#B87818")
AMBER_LIGHT = HexColor("#F8EEDB")
GRID = HexColor("#D9E0E3")
PAPER = HexColor("#F7F9F9")
WHITE = colors.white


def register_fonts():
    regular = Path(r"C:\Windows\Fonts\Deng.ttf")
    bold = Path(r"C:\Windows\Fonts\Dengb.ttf")
    if not regular.exists() or not bold.exists():
        raise FileNotFoundError("Deng Chinese fonts are required")
    pdfmetrics.registerFont(TTFont("Deng", str(regular)))
    pdfmetrics.registerFont(TTFont("Deng-Bold", str(bold)))


register_fonts()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="CNBody",
        fontName="Deng",
        fontSize=9.2,
        leading=14.2,
        textColor=INK,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="CNSmall",
        fontName="Deng",
        fontSize=7.6,
        leading=10.6,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        name="CNTitle",
        fontName="Deng-Bold",
        fontSize=24,
        leading=31,
        textColor=INK,
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        name="CNSubtitle",
        fontName="Deng",
        fontSize=11.5,
        leading=17,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        name="CNSection",
        fontName="Deng-Bold",
        fontSize=16,
        leading=22,
        textColor=INK,
        spaceBefore=2,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="CNSubsection",
        fontName="Deng-Bold",
        fontSize=11.2,
        leading=15,
        textColor=TEAL,
        spaceBefore=5,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="CNCallout",
        fontName="Deng-Bold",
        fontSize=10.2,
        leading=15.5,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        name="CNTable",
        fontName="Deng",
        fontSize=7.5,
        leading=10.2,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        name="CNTableHeader",
        fontName="Deng-Bold",
        fontSize=7.5,
        leading=10,
        textColor=WHITE,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        name="CNMetric",
        fontName="Deng-Bold",
        fontSize=18,
        leading=22,
        textColor=INK,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        name="CNMetricLabel",
        fontName="Deng-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=INK,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        name="CNMetricNote",
        fontName="Deng",
        fontSize=7.2,
        leading=9.2,
        textColor=MUTED,
        alignment=TA_CENTER,
    )
)


def p(text, style="CNBody"):
    return Paragraph(text, styles[style])


def tag(text, color):
    return Table(
        [[p(text, "CNTableHeader")]],
        colWidths=[31 * mm],
        rowHeights=[7 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0, color),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        ),
    )


def callout(title, body, tone="neutral"):
    palette = {
        "neutral": (PAPER, TEAL),
        "pass": (GREEN_LIGHT, GREEN),
        "fail": (CORAL_LIGHT, CORAL),
        "hold": (AMBER_LIGHT, AMBER),
    }
    background, accent = palette[tone]
    content = [[p(title, "CNCallout")], [p(body, "CNBody")]]
    table = Table(content, colWidths=[CONTENT_W - 8 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, 0), 6),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 6),
            ]
        )
    )
    return table


def styled_table(headers, rows, widths, header_color=INK, font_size=7.5):
    data = [[p(str(value), "CNTableHeader") for value in headers]]
    for row in rows:
        data.append([p(str(value), "CNTable") for value in row])
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("GRID", (0, 0), (-1, -1), 0.45, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PAPER]),
            ]
        )
    )
    return table


def metric_cards(items):
    cells = []
    for label, value, note, tone in items:
        background = {
            "pass": GREEN_LIGHT,
            "fail": CORAL_LIGHT,
            "hold": AMBER_LIGHT,
            "neutral": PAPER,
        }[tone]
        cell = Table(
            [[p(value, "CNMetric")], [p(label, "CNMetricLabel")], [p(note, "CNMetricNote")]],
            colWidths=[(CONTENT_W - 9 * mm) / len(items)],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), background),
                    ("TEXTCOLOR", (0, 1), (-1, 1), INK),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 5),
                ]
            ),
        )
        cells.append(cell)
    outer = Table([cells], colWidths=[(CONTENT_W - 9 * mm) / len(items)] * len(items))
    outer.setStyle(
        TableStyle(
            [
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return outer


def horizontal_bar_chart(items, maximum, width=CONTENT_W, height=69 * mm):
    d = Drawing(width, height)
    left = 42 * mm
    right = 15 * mm
    top = 9 * mm
    row_h = (height - top - 7 * mm) / len(items)
    chart_w = width - left - right
    for i, (label, value, color, note) in enumerate(items):
        y = height - top - (i + 1) * row_h + 3
        d.add(String(0, y + 3, label, fontName="Deng", fontSize=8, fillColor=INK))
        d.add(Rect(left, y, chart_w, 6, fillColor=PAPER, strokeColor=GRID, strokeWidth=0.4))
        bar_w = chart_w * min(value / maximum, 1.0)
        d.add(Rect(left, y, bar_w, 6, fillColor=color, strokeColor=None))
        d.add(
            String(
                left + bar_w + 4,
                y + 1,
                f"{value:.3f} ft  {note}",
                fontName="Deng-Bold",
                fontSize=7.4,
                fillColor=color,
            )
        )
    return d


def line_chart(series, x_values, y_max=4.5, width=CONTENT_W, height=72 * mm):
    d = Drawing(width, height)
    left = 18 * mm
    bottom = 13 * mm
    top = 8 * mm
    right = 16 * mm
    chart_w = width - left - right
    chart_h = height - bottom - top
    d.add(Line(left, bottom, left, bottom + chart_h, strokeColor=INK, strokeWidth=0.7))
    d.add(Line(left, bottom, left + chart_w, bottom, strokeColor=INK, strokeWidth=0.7))
    for y_value in [0, 1, 1.5, 2, 3, 4]:
        y = bottom + chart_h * y_value / y_max
        d.add(Line(left, y, left + chart_w, y, strokeColor=GRID, strokeWidth=0.35))
        d.add(String(1, y - 2, f"{y_value:g}", fontName="Deng", fontSize=6.8, fillColor=MUTED))
    for x in x_values:
        x_pos = left + chart_w * (x - min(x_values)) / (max(x_values) - min(x_values))
        d.add(String(x_pos - 4, 2, str(x), fontName="Deng", fontSize=6.8, fillColor=MUTED))
    d.add(String(left + chart_w / 2 - 15, 2, "维度 / controls", fontName="Deng", fontSize=7, fillColor=MUTED))
    d.add(Line(left, bottom + chart_h * 1.5 / y_max, left + chart_w, bottom + chart_h * 1.5 / y_max, strokeColor=CORAL, strokeWidth=1, strokeDashArray=[3, 2]))
    d.add(String(left + chart_w - 37, bottom + chart_h * 1.5 / y_max + 3, "E0 gate 1.5", fontName="Deng-Bold", fontSize=7, fillColor=CORAL))
    for label, values, color in series:
        points = []
        for x, value in values:
            x_pos = left + chart_w * (x - min(x_values)) / (max(x_values) - min(x_values))
            y_pos = bottom + chart_h * min(value, y_max) / y_max
            points.append((x_pos, y_pos))
        for a, b in zip(points, points[1:]):
            d.add(Line(a[0], a[1], b[0], b[1], strokeColor=color, strokeWidth=1.5))
        for point in points:
            d.add(Circle(point[0], point[1], 2, fillColor=color, strokeColor=WHITE, strokeWidth=0.5))
        legend_x = left + chart_w - 80
        legend_y = height - 10 - 10 * series.index((label, values, color))
        d.add(Line(legend_x, legend_y, legend_x + 15, legend_y, strokeColor=color, strokeWidth=2))
        d.add(String(legend_x + 20, legend_y - 3, label, fontName="Deng", fontSize=7.2, fillColor=INK))
    return d


def grouped_bar_chart(groups, width=CONTENT_W, height=67 * mm):
    d = Drawing(width, height)
    left = 21 * mm
    bottom = 13 * mm
    top = 8 * mm
    chart_w = width - left - 12 * mm
    chart_h = height - bottom - top
    max_value = max(value for _, values in groups for value, _ in values) * 1.12
    d.add(Line(left, bottom, left, bottom + chart_h, strokeColor=INK, strokeWidth=0.7))
    d.add(Line(left, bottom, left + chart_w, bottom, strokeColor=INK, strokeWidth=0.7))
    total_bars = sum(len(values) for _, values in groups)
    gap = 8
    bar_w = (chart_w - gap * (len(groups) + 1)) / total_bars
    x = left + gap
    for group_label, values in groups:
        group_start = x
        for value, color in values:
            h = chart_h * value / max_value
            d.add(Rect(x, bottom, bar_w, h, fillColor=color, strokeColor=None))
            d.add(String(x + 1, bottom + h + 3, f"{value:.2f}", fontName="Deng-Bold", fontSize=6.8, fillColor=color))
            x += bar_w
        group_end = x
        d.add(String((group_start + group_end) / 2 - 18, 2, group_label, fontName="Deng", fontSize=7, fillColor=INK))
        x += gap
    d.add(String(2, height - 8, "ft", fontName="Deng", fontSize=7, fillColor=MUTED))
    return d


class ReviewDocTemplate(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=A4,
            rightMargin=MARGIN_X,
            leftMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title="ROGII 两份单模型路线验证复盘",
            author="OpenAI Codex",
            subject="Full-Well v3 与 DeltaCorr-SteerSpline v0.1 验证结果",
        )
        frame = Frame(
            MARGIN_X,
            MARGIN_BOTTOM,
            CONTENT_W,
            PAGE_H - MARGIN_TOP - MARGIN_BOTTOM,
            id="normal",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self.draw_page))

    def draw_page(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.5)
            canvas.line(MARGIN_X, PAGE_H - 11 * mm, PAGE_W - MARGIN_X, PAGE_H - 11 * mm)
            canvas.setFillColor(MUTED)
            canvas.setFont("Deng", 7.2)
            canvas.drawString(MARGIN_X, PAGE_H - 8.5 * mm, "ROGII 单模型路线验证复盘")
            canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 8.5 * mm, "2026-07-19")
        canvas.setFillColor(MUTED)
        canvas.setFont("Deng", 7.2)
        canvas.drawCentredString(PAGE_W / 2, 7 * mm, f"{doc.page}")
        canvas.restoreState()


story = []


# Cover
story.append(Spacer(1, 17 * mm))
story.append(tag("VALIDATION REVIEW", TEAL))
story.append(Spacer(1, 13 * mm))
story.append(p("ROGII 两份单模型路线验证复盘", "CNTitle"))
story.append(
    p(
        "Full-Well Single-Model Plan v3 与 DeltaCorr-SteerSpline v0.1 的本地实验结果、网关判决和保留资产",
        "CNSubtitle",
    )
)
story.append(Spacer(1, 13 * mm))
story.append(
    callout(
        "最终裁决：两份路线作为 5.5 ft 完整执行方案均为 NO-GO",
        "第一份路线的整井架构在 held-out Fold 1 无法形成可迁移学习；第二份路线的低维表示和 typewell 对应机制得到局部验证，但 E1 24井容量门与 E6a 20% 信息增益门均未通过。这里的结论是否定当前工程充分性，不等于否定所有局部机制。",
        "fail",
    )
)
story.append(Spacer(1, 11 * mm))
story.append(
    metric_cards(
        [
            ("生产安全端点", "6.792", "CP12 -> Gold -> Full-progress Neighbor", "neutral"),
            ("Full-Well Fold 1", "15.40", "几何中心 S-head，训练侧同样不学习", "fail"),
            ("E1 24井容量", "1.050", "目标 < 1.000，差 0.050 ft", "fail"),
            ("DeltaCorr 增益", "10.0%", "目标 >= 20%，但因果信号真实", "hold"),
        ]
    )
)
story.append(Spacer(1, 15 * mm))
story.append(p("报告日期：2026-07-19", "CNSmall"))
story.append(p("验证口径：固定 physical-well Fold 1、official visible-prefix mask、exact pooled RMSE / held-out Delta MAE", "CNSmall"))
story.append(PageBreak())


# Executive summary
story.append(p("1. 执行摘要", "CNSection"))
story.append(
    callout(
        "应该如何理解“证伪”",
        "被否定的是两份 PDF 作为完整 5.5 ft 路线的工程假设和继续投入资格。PCA 低维表示、joint-gradient 和配对 typewell 局部证据均得到正面结果；未执行的 E2-E5、E7-E8 也没有被逐项证明无效，但由于上游网关失败，不再具有合理投入依据。",
        "neutral",
    )
)
story.append(Spacer(1, 7 * mm))
summary_rows = [
    ["Full-Well v3", "完整井原始分辨率 + 双参考 + 低维轨迹头", "容量可记忆，Fold 1 约 13-15 ft；训练损失平坦", "NO-GO"],
    ["DeltaCorr-SteerSpline", "Fold-clean PCA 主头 + 稠密 DeltaCorr 对应监督", "E0 PASS；E1 24井 1.050；E6a 信息增益 10%", "NO-GO"],
    ["PCA 表示", "outer-train only 的 anchored-S basis", "PCA16 oracle 0.581；PCA32 0.280", "保留"],
    ["Joint-gradient", "多井损失累计后统一 optimizer step", "四井从 1.291 改善到 0.657", "保留"],
    ["DeltaCorr typewell", "真实 reference vs no-reference / shuffled", "真实信号成立，但量级不足且尾部改善弱", "研究资产"],
]
story.append(
    styled_table(
        ["对象", "目标", "最重要证据", "裁决"],
        summary_rows,
        [31 * mm, 51 * mm, 76 * mm, 22 * mm],
    )
)
story.append(Spacer(1, 8 * mm))
story.append(p("统一验证纪律", "CNSubsection"))
story.append(
    p(
        "所有 episode 与 basis 均继承 physical-well outer fold；验证使用与测试一致的 TVT_input visible-prefix / hidden-suffix mask。Capacity 只证明同一训练井可拟合，不能与 held-out Fold 1 或 OOF 混为一谈。Oracle 只用于表示上限，不是可部署成绩。",
        "CNBody",
    )
)
story.append(
    callout(
        "停止原则",
        "未通过预注册门槛后，不再通过延长 epoch、换 seed、扩大模型、改 PCA 维度或追加下游模块来移动终点。",
        "hold",
    )
)
story.append(PageBreak())


# PDF A results
story.append(p("2. PDF A：Full-Well Single-Model Plan v3", "CNSection"))
story.append(p("目标与已测试范围", "CNSubsection"))
story.append(
    p(
        "该路线试图用一个完整井 checkpoint 联合消费原始 GR、合法 heel prefix、typewell 与完整 post-PS XYZ，再输出 Delta-TVT 状态或锚定 S 曲线。开发门为 Fold 1 <= 7 ft，最终五折目标为 <= 5.5 ft。",
        "CNBody",
    )
)
fullwell_rows = [
    ["Native dual-reference capacity", "24 train wells", "0.729", "p95 1.028；no-GR 恶化 36.13 ft", "PASS capacity"],
    ["Native dual-reference onefold v0", "155 held-out wells", "约 13.35", "明显高于现有 8.8-9.0 参考", "FAIL"],
    ["Full-well global TVT U-Net", "Fold 1", "15.554", "最佳 posterior mean decoder", "FAIL"],
    ["Full-well neural CRF", "Fold 1", "约 15.19", "长程结构化解码未恢复信号", "FAIL"],
    ["Geometry-centred B-spline capacity", "24 train wells", "4.374", "门 4.376，边界通过", "PASS capacity"],
    ["Geometry-centred S-head", "Fold 1 epoch 4", "15.397", "train subset 15.650，loss 基本不降", "FAIL transfer"],
]
story.append(
    styled_table(
        ["实验", "口径", "RMSE ft", "诊断", "结论"],
        fullwell_rows,
        [42 * mm, 31 * mm, 20 * mm, 65 * mm, 23 * mm],
    )
)
story.append(Spacer(1, 6 * mm))
story.append(
    horizontal_bar_chart(
        [
            ("24井 capacity", 0.729, GREEN, "同井记忆"),
            ("开发网关", 7.000, AMBER, "Fold 1 目标"),
            ("Native onefold", 13.350, CORAL, "held-out"),
            ("S-head Fold 1", 15.397, CORAL, "held-out"),
        ],
        maximum=16.0,
    )
)
story.append(PageBreak())


# PDF A conclusion
story.append(p("3. PDF A 的失败机制与正式判决", "CNSection"))
story.append(p("容量成功没有转化为跨井学习", "CNSubsection"))
story.append(
    p(
        "小集合 overfit 可以达到亚英尺或数英尺，但扩展到 618 口 outer-train 井后，训练子集与验证集同时停留在约 15 ft。该模式不是普通过拟合，而是共享编码器无法从不同井的完整序列中学习稳定的全局轨迹映射。",
        "CNBody",
    )
)
mechanism_rows = [
    ["跨井系数梯度冲突", "逐井 optimizer step 让不同井连续覆盖更新", "joint-gradient 后四井显著改善，但 24井仍边界失败"],
    ["错误中心轨迹", "population-mean S 与 zero-S 基线均在 100 ft 量级", "改成 geometry-centred Delta-TVT 后容量改善，Fold 仍不迁移"],
    ["原始分辨率不足以自动产生信息", "模型确实看到 native GR 与 XYZ", "held-out RMSE 仍为 13-15 ft"],
    ["复杂解码不是补救", "U-Net posterior、CRF、DP 均已测试", "未出现接近 7 ft 的路径"],
]
story.append(
    styled_table(
        ["问题", "证据", "修复后的结果"],
        mechanism_rows,
        [40 * mm, 68 * mm, 73 * mm],
    )
)
story.append(Spacer(1, 8 * mm))
story.append(
    callout(
        "PDF A 正式判决：NO-GO",
        "被较充分否定的是“直接由完整井编码器推断低维轨迹即可进入 7 ft”的核心实现。其数据协议、完整井 sample unit 和原始分辨率 loader 仍可复用，但不能继续以增加 backbone、pseudo-PS 或双头一致性来挽救同一架构。",
        "fail",
    )
)
story.append(Spacer(1, 7 * mm))
story.append(p("相关旁证", "CNSubsection"))
story.append(
    p(
        "S3 Axial/RAFT Fold 1 为 9.687 ft；合成预训练版本 9.788 ft；S5 合法 posterior 11.883 ft，而 pruned oracle 6.142 ft。这些实验共同表明：表示或候选覆盖可以很强，但合法证据聚合无法回收上限。",
        "CNBody",
    )
)
story.append(PageBreak())


# PDF B E0
story.append(p("4. PDF B：DeltaCorr-SteerSpline v0.1 - E0", "CNSection"))
story.append(
    p(
        "第二份路线首先重构输出表示：basis 只在 618 个 outer-train 井拟合，155 个 Fold 1 井的真值仅用于 oracle 投影。E0 证明锚定 S 曲线高度低维，这一结论成立。",
        "CNBody",
    )
)
e0_rows = [
    ["PCA", "32", "0.280", "0.445", "最佳 oracle"],
    ["B-spline", "32 controls", "0.289", "0.465", ""],
    ["DCT", "32", "0.325", "0.486", ""],
    ["B-spline", "24 controls", "0.410", "0.716", ""],
    ["PCA", "16", "0.581", "0.942", "E1 冻结基底"],
    ["B-spline", "12 controls", "0.879", "1.364", ""],
    ["PCA", "8", "1.165", "1.785", ""],
    ["B-spline", "8 controls", "1.376", "2.043", "最小通过 E0"],
]
story.append(
    styled_table(
        ["方法", "维度", "pooled oracle ft", "p95 well ft", "备注"],
        e0_rows,
        [31 * mm, 31 * mm, 37 * mm, 34 * mm, 48 * mm],
    )
)
story.append(Spacer(1, 5 * mm))
story.append(
    line_chart(
        [
            ("PCA", [(4, 2.090), (8, 1.165), (16, 0.581), (32, 0.280)], TEAL),
            ("DCT", [(4, 4.141), (8, 1.791), (16, 0.759), (32, 0.325)], AMBER),
            ("B-spline", [(8, 1.376), (12, 0.879), (16, 0.638), (24, 0.410), (32, 0.289)], GREEN),
        ],
        x_values=[4, 8, 12, 16, 24, 32],
    )
)
story.append(
    callout(
        "E0 结论：PASS",
        "共有 10 个配置达到 <= 1.5 ft。为满足下一步 absolute overfit < 1 ft，同时降低局部 B-spline 系数相关性，E1 冻结 PCA16，而不是自动选择的最小 B-spline 7D。",
        "pass",
    )
)
story.append(PageBreak())


# PDF B E1
story.append(p("5. PDF B - E1：PCA16 容量与训练协议", "CNSection"))
story.append(p("从假失败到严格 NO-GO", "CNSubsection"))
e1_rows = [
    ["初始 PCA16 one-well", "sequential，140 steps", "5.615", "FAIL；系数 loss 几乎不降"],
    ["Loss diagnostic free vector", "同一 basis/decoder", "0.562", "到达 representation oracle"],
    ["Loss diagnostic network", "系数预热 + path", "0.605", "PASS implementation"],
    ["Corrected one-well", "固定 updates", "0.900", "PASS < 1"],
    ["Corrected four-well", "sequential", "1.291", "FAIL；井间覆盖"],
    ["Four-well joint-gradient", "累计四井后 step", "0.657", "PASS；max 0.805"],
    ["24-well joint-gradient", "冻结日程，600+600 cycles", "1.050", "FAIL；门 1.000"],
]
story.append(
    styled_table(
        ["阶段", "训练方式", "pooled RMSE ft", "判读"],
        e1_rows,
        [48 * mm, 48 * mm, 34 * mm, 51 * mm],
    )
)
story.append(Spacer(1, 8 * mm))
story.append(
    metric_cards(
        [
            ("24井 oracle", "0.643", "表示上限", "pass"),
            ("24井 actual", "1.050", "最佳 cycle 600", "fail"),
            ("中位井", "0.862", "多数井已拟合", "hold"),
            ("p95 / max", "1.731 / 1.863", "尾部仍高于门", "fail"),
        ]
    )
)
story.append(Spacer(1, 9 * mm))
story.append(
    callout(
        "E1 严格判决：FAIL",
        "24井只差 0.050 ft，但继续增加 cycle、换 seed 或改 PCA32 会移动预注册终点。结果同时留下一个可靠工程结论：joint-gradient 能显著缓解多井 sequential-update 覆盖。",
        "fail",
    )
)
story.append(PageBreak())


# PDF B E6a
story.append(p("6. PDF B - E6a：DeltaCorr-257 局部 teacher", "CNSection"))
story.append(
    p(
        "为避免把 E1 的失败等同于 DeltaCorr 未知机制，额外执行了一次独立 held-out gateway。每个 outer-train 井抽 128 个 suffix 位置，每个验证井抽 192 个位置；paired 与 no-reference 为匹配训练模型，shuffled typewell 为因果干预。",
        "CNBody",
    )
)
e6_rows = [
    ["Paired typewell", "9.942", "14.286", "3.753", "37.60%", "7.830", "23.468"],
    ["No-reference", "11.047", "15.454", "4.097", "33.62%", "8.893", "23.804"],
    ["Shuffled typewell", "11.674", "16.137", "4.403", "31.56%", "9.286", "25.227"],
]
story.append(
    styled_table(
        ["模式", "Delta MAE", "Delta RMSE", "NLL", "within 5", "median MAE", "p95 MAE"],
        e6_rows,
        [34 * mm, 25 * mm, 27 * mm, 20 * mm, 23 * mm, 25 * mm, 27 * mm],
    )
)
story.append(Spacer(1, 5 * mm))
story.append(
    grouped_bar_chart(
        [
            ("Delta MAE", [(9.942, TEAL), (11.047, AMBER), (11.674, CORAL)]),
            ("Delta RMSE", [(14.286, TEAL), (15.454, AMBER), (16.137, CORAL)]),
            ("p95 MAE", [(23.468, TEAL), (23.804, AMBER), (25.227, CORAL)]),
        ]
    )
)
story.append(Spacer(1, 4 * mm))
story.append(
    metric_cards(
        [
            ("相对 MAE 增益", "10.0%", "门槛 >= 20%", "fail"),
            ("Shuffle 恶化", "+1.732", "真实配对信号明确", "pass"),
            ("正收益井", "64.5%", "门槛 >= 55%", "pass"),
            ("Bootstrap p05", "+0.724", "井级方向稳定", "pass"),
        ]
    )
)
story.append(Spacer(1, 8 * mm))
story.append(
    callout(
        "E6a 判决：因果信号确认，量级门失败",
        "配对 typewell 确实提供真实局部对应信息，不是井身份或随机统计量；但 10% 改善不足 20% 门，absolute Delta RMSE 仍为 14.286 ft，p95 仅改善 0.336 ft，无法承担 5.5 ft 路线的尾部治理。",
        "hold",
    )
)
story.append(PageBreak())


# Final synthesis
story.append(p("7. 合并结论、保留资产与停止清单", "CNSection"))
final_rows = [
    ["Full-Well v3 完整执行路线", "关闭", "多个 held-out 架构停留 13-15 ft，未接近 7 ft"],
    ["Deterministic PCA SteerSpline", "关闭", "E1 24井 1.050 > 1.000"],
    ["DeltaCorr 接回整井头", "关闭", "E6a 相对 MAE 增益 10% < 20%"],
    ["E2-E5、E7-E8", "不执行", "上游 E1/E6a 网关失败，不代表逐项科学证伪"],
    ["Fold-clean PCA basis", "保留", "PCA16 oracle 0.581；PCA32 0.280"],
    ["Joint-gradient 训练实现", "保留", "四井 1.291 -> 0.657，跨井冲突诊断明确"],
    ["DeltaCorr paired checkpoint", "研究资产", "证明 typewell 有约 1.1 ft 局部 MAE 价值"],
    ["6.792 生产安全链", "冻结", "与 standalone 研究分离，继续作为提交端事实锚点"],
]
story.append(
    styled_table(
        ["对象", "状态", "依据"],
        final_rows,
        [61 * mm, 28 * mm, 92 * mm],
    )
)
story.append(Spacer(1, 9 * mm))
story.append(
    callout(
        "最终工程结论",
        "两份 PDF 中若干局部假设成立，但“这些组件足以形成 pooled 5.5 ft standalone 单模型”的充分性假设没有通过预注册网关。下一条单模型路线必须引入新的可迁移证据或新的监督分解，而不是继续调节当前 SteerSpline、DeltaCorr、pseudo-PS 或 backbone 剂量。",
        "fail",
    )
)
story.append(Spacer(1, 8 * mm))
story.append(p("可复用实验产物", "CNSubsection"))
artifact_rows = [
    ["E0 basis bank", "deltacorr_steerspline_e0_basis_audit_v0/fold1_basis_bank.npz"],
    ["24井 PCA checkpoint", "deltacorr_steerspline_e1_pca16_24well_jointgrad_v4/models/best_24well_joint.pt"],
    ["DeltaCorr paired checkpoint", "deltacorr_e6a_local_teacher_fold1_v0/models/best_paired.pt"],
    ["DeltaCorr held-out predictions", "deltacorr_e6a_local_teacher_fold1_v0/predictions_paired.npz"],
]
story.append(
    styled_table(
        ["资产", "相对 /root/autodl-tmp/oof_exports 的路径"],
        artifact_rows,
        [48 * mm, 133 * mm],
        header_color=TEAL,
    )
)
story.append(Spacer(1, 9 * mm))
story.append(p("数据来源与限定", "CNSubsection"))
story.append(
    p(
        "本报告数字来自 2026-07-18 至 2026-07-19 的本地实验输出。Fold 1 使用 618/155 physical-well split；E6a 使用 29,760 个 held-out sampled positions。约数仅用于历史实验缺少完整最终表的项目，并已显式标注。LB 6.792 属于旧生产链，不与 standalone Fold 1 数字直接比较。",
        "CNSmall",
    )
)


doc = ReviewDocTemplate(str(OUTPUT))
doc.build(story)
print(OUTPUT)
