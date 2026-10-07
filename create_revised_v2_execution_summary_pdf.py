from __future__ import annotations

from pathlib import Path
from typing import Iterable

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "pdf" / "ROGII_revised_V2_execution_summary_2026-07-18.pdf"
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")

PAGE_W, PAGE_H = A4
MARGIN_X = 17 * mm
MARGIN_TOP = 18 * mm
MARGIN_BOTTOM = 17 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_X

INK = colors.HexColor("#172126")
MUTED = colors.HexColor("#5C6A70")
TEAL = colors.HexColor("#087E73")
TEAL_LIGHT = colors.HexColor("#E5F3F1")
BLUE = colors.HexColor("#276A8C")
BLUE_LIGHT = colors.HexColor("#E9F2F7")
RED = colors.HexColor("#B23A3A")
RED_LIGHT = colors.HexColor("#F9EAEA")
AMBER = colors.HexColor("#A36A00")
AMBER_LIGHT = colors.HexColor("#FFF3D7")
GREEN = colors.HexColor("#2E7D4F")
GREEN_LIGHT = colors.HexColor("#E8F4EC")
GRID = colors.HexColor("#D7DEE1")
PAPER = colors.white
SOFT = colors.HexColor("#F4F7F8")


def register_fonts() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(FONT_PATH)
    pdfmetrics.registerFont(TTFont("CN", str(FONT_PATH)))


register_fonts()

styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="CNTitle",
        fontName="CN",
        fontSize=24,
        leading=31,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=7 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="CNSubtitle",
        fontName="CN",
        fontSize=10.5,
        leading=16,
        textColor=MUTED,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="CNH1",
        fontName="CN",
        fontSize=17,
        leading=23,
        textColor=INK,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="CNH2",
        fontName="CN",
        fontSize=12.5,
        leading=18,
        textColor=TEAL,
        spaceBefore=3 * mm,
        spaceAfter=2.2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="CNBody",
        fontName="CN",
        fontSize=9.4,
        leading=15.2,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=2.3 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="CNSmall",
        fontName="CN",
        fontSize=7.8,
        leading=11.5,
        textColor=MUTED,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="CNTable",
        fontName="CN",
        fontSize=7.7,
        leading=10.5,
        textColor=INK,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        name="CNTableHeader",
        fontName="CN",
        fontSize=7.6,
        leading=10,
        textColor=colors.white,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        name="CNCallout",
        fontName="CN",
        fontSize=10.2,
        leading=16,
        textColor=INK,
        alignment=TA_LEFT,
    )
)


def p(text: str, style: str = "CNBody") -> Paragraph:
    return Paragraph(text, styles[style])


def table(
    headers: Iterable[str],
    rows: Iterable[Iterable[str]],
    widths: list[float],
    header_color=TEAL,
    alignments: dict[int, str] | None = None,
) -> Table:
    data = [[p(str(item), "CNTableHeader") for item in headers]]
    data.extend([[p(str(item), "CNTable") for item in row] for row in rows])
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]
    for row in range(1, len(data)):
        commands.append(
            ("BACKGROUND", (0, row), (-1, row), PAPER if row % 2 else SOFT)
        )
    for column, alignment in (alignments or {}).items():
        commands.append(("ALIGN", (column, 1), (column, -1), alignment))
    result.setStyle(TableStyle(commands))
    return result


def callout(title: str, body: str, tone: str = "teal") -> Table:
    palette = {
        "teal": (TEAL, TEAL_LIGHT),
        "red": (RED, RED_LIGHT),
        "amber": (AMBER, AMBER_LIGHT),
        "green": (GREEN, GREEN_LIGHT),
        "blue": (BLUE, BLUE_LIGHT),
    }
    accent, background = palette[tone]
    content = p(f"<font color='{accent.hexval()}'>{title}</font><br/>{body}", "CNCallout")
    box = Table([["", content]], colWidths=[3.5 * mm, CONTENT_W - 3.5 * mm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), accent),
                ("BACKGROUND", (1, 0), (1, 0), background),
                ("BOX", (0, 0), (-1, -1), 0.4, accent),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
                ("RIGHTPADDING", (1, 0), (1, 0), 8),
                ("TOPPADDING", (1, 0), (1, 0), 8),
                ("BOTTOMPADDING", (1, 0), (1, 0), 8),
            ]
        )
    )
    return box


def baseline_chart() -> Drawing:
    width = CONTENT_W
    height = 53 * mm
    drawing = Drawing(width, height)
    values = [
        ("无 Neighbor", 6.991, MUTED),
        ("旧 Frozen Neighbor", 6.882, BLUE),
        ("当前 Full-progress", 6.792, TEAL),
    ]
    chart_left = 44 * mm
    chart_right = width - 13 * mm
    chart_width = chart_right - chart_left
    minimum, maximum = 6.70, 7.05
    for index, (label, value, color) in enumerate(values):
        y = height - 13 * mm - index * 14 * mm
        drawing.add(String(0, y + 2, label, fontName="CN", fontSize=8.5, fillColor=INK))
        drawing.add(Rect(chart_left, y, chart_width, 5.5 * mm, fillColor=SOFT, strokeColor=None))
        bar = (value - minimum) / (maximum - minimum) * chart_width
        drawing.add(Rect(chart_left, y, bar, 5.5 * mm, fillColor=color, strokeColor=None))
        drawing.add(
            String(
                chart_left + bar + 2.5 * mm,
                y + 2,
                f"{value:.3f}",
                fontName="CN",
                fontSize=8.5,
                fillColor=color,
            )
        )
    drawing.add(
        String(
            chart_left,
            1.5 * mm,
            "公开 LB RMSE，越低越好",
            fontName="CN",
            fontSize=7,
            fillColor=MUTED,
        )
    )
    return drawing


def gate_flow() -> Drawing:
    width = CONTENT_W
    height = 48 * mm
    drawing = Drawing(width, height)
    boxes = [
        ("G1\n低自由度增量", BLUE, BLUE_LIGHT),
        ("G2\n整井动作 oracle", AMBER, AMBER_LIGHT),
        ("G3a\n前缀适应信号", TEAL, TEAL_LIGHT),
        ("T1\n冻结确认", GREEN, GREEN_LIGHT),
        ("Fallback\n回到单井模型", RED, RED_LIGHT),
    ]
    gap = 3 * mm
    box_w = (width - gap * (len(boxes) - 1)) / len(boxes)
    y = 12 * mm
    for index, (label, accent, background) in enumerate(boxes):
        x = index * (box_w + gap)
        drawing.add(Rect(x, y, box_w, 21 * mm, fillColor=background, strokeColor=accent, strokeWidth=0.8))
        parts = label.split("\n")
        drawing.add(String(x + box_w / 2, y + 13.5 * mm, parts[0], textAnchor="middle", fontName="CN", fontSize=10, fillColor=accent))
        drawing.add(String(x + box_w / 2, y + 6.5 * mm, parts[1], textAnchor="middle", fontName="CN", fontSize=7.2, fillColor=INK))
        if index < len(boxes) - 1:
            x1 = x + box_w
            x2 = x + box_w + gap
            drawing.add(Line(x1 + 0.5 * mm, y + 10.5 * mm, x2 - 0.5 * mm, y + 10.5 * mm, strokeColor=GRID, strokeWidth=1.1))
    drawing.add(String(0, 1.5 * mm, "每一层只有通过预注册网关，才允许消耗下一层训练或确认预算。", fontName="CN", fontSize=7.5, fillColor=MUTED))
    return drawing


def page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.35)
    canvas.line(MARGIN_X, PAGE_H - 10.5 * mm, PAGE_W - MARGIN_X, PAGE_H - 10.5 * mm)
    canvas.setFont("CN", 7.3)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN_X, PAGE_H - 8 * mm, "ROGII Revised V2 执行总结")
    canvas.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 8 * mm, "2026-07-18")
    canvas.line(MARGIN_X, 10.5 * mm, PAGE_W - MARGIN_X, 10.5 * mm)
    canvas.drawString(MARGIN_X, 6.5 * mm, "本地 OOF 用于开发判断；公开 LB 用于提交端事实确认。")
    canvas.drawRightString(PAGE_W - MARGIN_X, 6.5 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_story() -> list:
    story = []

    story.extend(
        [
            Spacer(1, 7 * mm),
            p("ROGII Revised V2<br/>方案执行总结与最终判决", "CNTitle"),
            p("范围：ROGII_6.882_to_5x_revised_execution_plan_v2 及其已授权网关", "CNSubtitle"),
            Spacer(1, 8 * mm),
            callout(
                "最终状态：FALLBACK",
                "最新 PDF 的短期增量路线已经走完。没有任何可部署挑战者通过完整网关；冻结当前公开 LB 6.792，并将研究资源转回原始分辨率的单井完整路径模型。",
                "red",
            ),
            Spacer(1, 7 * mm),
            p("执行摘要", "CNH1"),
            table(
                ["项目", "结论"],
                [
                    ["当前生产安全端点", "CP12 -> Gold -> Full-progress geometric Neighbor，公开 LB 6.792"],
                    ["本地开发代理", "neighbor_geom_cp12，OOF RMSE 8.738；仅用于发现，不是 6.792 的本地复现"],
                    ["G1 低自由度增量", "最好候选整体有正收益，但 p95 尾部恶化，D-gate 失败"],
                    ["G2 整井动作篮子", "oracle 上限明显，但所有部署选择器均未回收该上限"],
                    ["G3a/T1 在线适应", "前缀信号不稳定，G3a 失败，T1 按规则不启动"],
                    ["最终动作", "保留 6.792 与独立 6.991 备份；关闭短期补丁路线"],
                ],
                [42 * mm, CONTENT_W - 42 * mm],
            ),
            Spacer(1, 6 * mm),
            p("一句话结论", "CNH2"),
            p(
                "现有链仍是最可靠的提交系统，但 PDF 没有找到可以安全叠加的新处理。下一阶段不应继续围绕候选路由和后处理微调，而应重新解决单井模型的输入分辨率与完整路径表示。"
            ),
        ]
    )

    story.append(PageBreak())
    story.extend(
        [
            p("1. 基线、口径与执行规则", "CNH1"),
            p("生产端点校正", "CNH2"),
            p(
                "PDF 原文以旧 Frozen Neighbor 的 6.882 为起点。执行时根据真实提交结果，将安全端点校正为 6.792，且固定处理顺序为 CP12 -> Gold -> Full-progress geometric Neighbor。"
            ),
            baseline_chart(),
            Spacer(1, 3 * mm),
            callout(
                "重要口径",
                "hidden 井真值不可见，因此无法在本地重建公开 LB 6.792 的残差向量。本地 OOF 8.738 只承担开发筛选职责；任何挑战者即使本地通过，也必须冻结后再进入未使用确认折。",
                "amber",
            ),
            Spacer(1, 5 * mm),
            p("预注册网关", "CNH2"),
            table(
                ["网关", "主要阈值", "用途"],
                [
                    ["D", "nested gain >= 0.05；p95 delta <= 0；worst20 gain >= 0", "筛选低自由度、尾部安全的端点增量"],
                    ["C", "新确认折 gain >= 0.05；保留开发收益 >= 50%", "防止开发折调参幻觉"],
                    ["P", "联合 gain >= 0.10；自适应 gain >= 0.15；至少 4 正折", "筛选组合或自适应方案"],
                    ["G2", "oracle gain >= 0.20；尾部 SSE 恢复 >= 15%", "判断动作篮子是否存在理论价值"],
                    ["G3a", "Spearman >= 0.25 或 AUC >= 0.60，并通过覆盖和负对照", "授权或关闭 T1 前缀在线适应"],
                ],
                [18 * mm, 65 * mm, CONTENT_W - 83 * mm],
            ),
            Spacer(1, 6 * mm),
            gate_flow(),
        ]
    )

    story.append(PageBreak())
    story.extend(
        [
            p("2. G1：低自由度增量测试", "CNH1"),
            p(
                "G1 将现有候选端点以单一全局剂量接到本地安全代理上，并使用 nested OOF 选择剂量。主判据不仅看 pooled RMSE，还要求五折稳定、最差 20% 井不恶化、p95 尾部不恶化。"
            ),
            p("最佳候选：strict-spatial LGB", "CNH2"),
            table(
                ["指标", "读数", "门槛/解释", "结果"],
                [
                    ["本地安全代理 RMSE", "8.7383", "开发基准", "-"],
                    ["Nested 总体增益", "+0.1032", ">= +0.05", "通过"],
                    ["平均 nested 剂量", "0.3418", "自动由外层训练折确定", "诊断"],
                    ["最差折增益", "+0.0589", "不得为负", "通过"],
                    ["正收益折", "5 / 5", "稳定性要求", "通过"],
                    ["改善井比例", "55.11%", "逐井分布", "边缘"],
                    ["逐井 delta 中位数", "-0.0856", "负值代表改善", "通过"],
                    ["Worst-20 增益", "+0.2201", ">= 0", "通过"],
                    ["p95 逐井 delta", "+0.3537", "<= 0", "失败"],
                ],
                [43 * mm, 28 * mm, 62 * mm, CONTENT_W - 133 * mm],
            ),
            Spacer(1, 6 * mm),
            callout(
                "G1 判决：FAIL",
                "strict-spatial 在平均意义上有效，却把高分位尾井进一步推坏。根据预注册 D-gate，不能因总体 +0.103 而忽略 p95 +0.354。其余候选的总体增益不足或稳定性更差，因此 G1 没有挑战者进入确认折。",
                "red",
            ),
            Spacer(1, 5 * mm),
            p("机制解释", "CNH2"),
            p(
                "这复现了此前的核心风险：许多空间或结构模型能改善平均井，却与现有 CP12/Gold/Neighbor 链共享相似误差方向，并在少数重尾井上产生灾难性偏移。PDF 的 D-gate 正是为了阻止这种“总体涨分、尾部失控”的方案进入提交端。"
            ),
        ]
    )

    story.append(PageBreak())
    story.extend(
        [
            p("3. G2：动作篮子 oracle 与部署鸿沟", "CNH1"),
            p(
                "G2 不直接训练新轨迹，而是将安全预测及若干合法候选视作整井动作，计算每口井选择最优动作的 oracle。它回答的是“候选库里有没有可用收益”，而不是“是否已经会选”。"
            ),
            table(
                ["指标", "读数", "结论"],
                [
                    ["安全代理 RMSE", "8.7383", "当前本地开发起点"],
                    ["整井 oracle RMSE", "7.7144", "动作篮子存在明显理论上限"],
                    ["Oracle gain", "+1.0240", "远高于 +0.20 网关"],
                    ["尾部 SSE 恢复", "20.12%", "超过 15% 网关"],
                    ["存在更优动作的井", "73.61%", "不是少数井偶然贡献"],
                    ["已验证部署选择器", "dSSE / listwise / ranker", "均无法稳定回收 oracle"],
                ],
                [53 * mm, 35 * mm, CONTENT_W - 88 * mm],
            ),
            Spacer(1, 6 * mm),
            callout(
                "G2 判决：ORACLE ONLY",
                "动作篮子值得保留为研究诊断，但 oracle 使用了真实 toe 标签，不能部署。dSSE nested selector 最终选择安全动作，实际增益 0；其他 ranker/listwise 方案也已失败。因此不得因为 oracle 很高而重新训练同类路由器。",
                "amber",
            ),
            Spacer(1, 6 * mm),
            p("为什么没有继续训练更大的选择器", "CNH2"),
            p(
                "已有实验反复显示：菜单的理论上限很高，但合法前缀证据不足以判断整井哪个候选会赢。增大模型容量只会加强开发折选择偏差。除非出现新的可部署证据源，否则继续做候选排序属于重复已失败实验。"
            ),
            p("保留资产", "CNH2"),
            p(
                "候选路径及其来源标签仍可用于表示上限、困难井分析或未来模型的数据增强，但不再作为主预测器或提交端硬路由菜单。"
            ),
        ]
    )

    story.append(PageBreak())
    story.extend(
        [
            p("4. G3a、T1 与既有结构分支", "CNH1"),
            p("G3a：前缀适应信号", "CNH2"),
            table(
                ["指标", "读数", "门槛", "结果"],
                [
                    ["有效井", "107 / 155", "-", "-"],
                    ["覆盖率", "69.03%", ">= 70%", "失败"],
                    ["主信号 Spearman", "0.0830", ">= 0.25", "失败"],
                    ["主信号 AUC", "0.5488", ">= 0.60", "失败"],
                    ["Spearman bootstrap p05", "-0.0993", "方向应稳定为正", "失败"],
                    ["AUC bootstrap p05", "0.4539", "应稳定高于随机", "失败"],
                    ["Wrong-well Spearman", "-0.1411", "绝对值应 <= 0.10", "失败"],
                    ["同折乐观增益", "+0.0578", "仅诊断，不能提升", "不足"],
                    ["Shuffled adapter RMSE", "10.2194", "说明前缀含信息，但不足以预测 toe 行动收益", "诊断"],
                ],
                [49 * mm, 34 * mm, 58 * mm, CONTENT_W - 141 * mm],
            ),
            Spacer(1, 5 * mm),
            callout(
                "G3a 判决：FAIL，T1 不启动",
                "前缀内拟合改善与 toe 段真实收益几乎无相关性，而且 bootstrap 和 wrong-well 控制均未通过。T1 依赖 G3a 授权，因此跳过 T1 是执行计划，不是漏跑。",
                "red",
            ),
            Spacer(1, 5 * mm),
            p("其他已关闭分支", "CNH2"),
            table(
                ["分支", "关键数据", "结论"],
                [
                    ["G4 / S3 模式修复", "S3 最佳 standalone 约 9.687；修复审计未达到架构门", "解码或迭代修补不足以进入 5.x"],
                    ["P3 / E1 absolute energy", "K192 oracle 5.119；实际选择约 10.686；oracle 回收约 5.97%", "生成器上限存在，选择器失败"],
                    ["P4 / 固定 S 轴", "±96 ft eval 覆盖仅约 60.8%；S 漂移 p95 约 205.5 ft", "状态轴与数据几何不匹配"],
                    ["Set96 / S5 / dSSE", "均未稳定优于安全基线", "关闭学习型候选选择"],
                    ["Neighbor 多维响应拟合", "2-D 响应融合公开 LB 7.846", "明显劣于 6.792，关闭"],
                ],
                [40 * mm, 74 * mm, CONTENT_W - 114 * mm],
            ),
        ]
    )

    story.append(PageBreak())
    story.extend(
        [
            p("5. 最终结论与下一阶段", "CNH1"),
            callout(
                "Day-5 正式判决：FALLBACK",
                "没有 G1 D-gate 挑战者；G2 只有不可部署 oracle；G3a 失败并关闭 T1；E1、S-space 和候选选择器已被既有证据关闭。PDF 的所有条件分支均已得到执行或合法跳过。",
                "red",
            ),
            Spacer(1, 6 * mm),
            p("提交端动作", "CNH2"),
            table(
                ["动作", "决定"],
                [
                    ["主安全提交", "保留 LB 6.792：CP12 -> Gold -> Full-progress geometric Neighbor"],
                    ["独立备份", "保留无 Neighbor 的 LB 6.991"],
                    ["不进入提交", "strict-spatial、T1、E1、S-space、学习型路由器、2-D Neighbor 响应融合"],
                    ["不再重开", "Set96/listwise/dSSE、Neighbor multi-k/local-plane/jitter 等已失败同类路线"],
                ],
                [42 * mm, CONTENT_W - 42 * mm],
            ),
            Spacer(1, 6 * mm),
            p("研究结论", "CNH2"),
            p(
                "PDF 验证的不是“GR matching 无效”，而是现有低分辨率表示、候选菜单选择和后处理叠加已经到顶。当前 standalone 模型长期停留在 9-10 ft，而高排名选手明确报告完整井样本的 pooled 5-fold 低于 5 ft；差距应回到基础模型表示本身解决。"
            ),
            p("下一阶段唯一主线", "CNH2"),
            table(
                ["设计项", "下一版约束"],
                [
                    ["样本单位", "完整单井，不依赖邻井、CP12、Gold 或 hidden 反馈"],
                    ["水平井输入", "先在原始约 1-ft 分辨率上编码，再下采样；不再预压缩至 256/384 列"],
                    ["双参考", "已知 heel 前缀 GR+TVT_input 与完整 typewell TVT/GR 分开编码"],
                    ["状态轴", "TVT - last_known_TVT，约 ±128 ft、1-ft 状态；不再使用固定 S 轴"],
                    ["输出", "完整平滑路径或 24-32 个控制点，硬锚定 prediction start"],
                    ["首个闸门", "24 井过拟合 <1 ft；随后 fold1 standalone <=7 ft 才扩展五折，最终目标 <=5.5 ft"],
                    ["因果负对照", "no-GR、shuffled typewell、shuffled heel-reference 必须显著退化"],
                ],
                [42 * mm, CONTENT_W - 42 * mm],
            ),
            Spacer(1, 6 * mm),
            callout(
                "最终判断",
                "最新 PDF 已经完成其职责：用严格网关关闭缺乏部署证据的短期补丁。接下来不是再为 6.792 寻找小修正，而是训练原始分辨率、完整井、双参考的 standalone 单井系统。",
                "green",
            ),
            Spacer(1, 5 * mm),
            p(
                "复现实验入口：revised_execution_plan_v2_gateway_cell.py；G3a 审计入口：revised_v2_g3a_prefix_signal_audit_cell.py。",
                "CNSmall",
            ),
        ]
    )

    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=MARGIN_X,
        rightMargin=MARGIN_X,
        topMargin=MARGIN_TOP,
        bottomMargin=MARGIN_BOTTOM,
        title="ROGII Revised V2 方案执行总结与最终判决",
        author="Codex",
        subject="ROGII Wellbore Geology Prediction revised V2 execution summary",
    )
    document.build(build_story(), onFirstPage=page_header_footer, onLaterPages=page_header_footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
