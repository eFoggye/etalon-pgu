"""Таблица экономики в макете шаблона ментора: reports/economics.xlsx.

Исходные цифры берутся из powerplant.economics.PlantAssumptions, расчёт — живыми формулами
шаблона: поменяйте любую синюю ячейку, и таблица пересчитается.
Запуск: python3 scripts/economics_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openpyxl import Workbook  # noqa: E402
from openpyxl.chart import LineChart, Reference  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.worksheet.worksheet import Worksheet  # noqa: E402

from powerplant.economics import HOURS_PER_YEAR, WORKDAY_HOURS, PlantAssumptions  # noqa: E402

OUTPUT = ROOT / "reports" / "economics.xlsx"
CALC, SENS, CHART = "Расчёт", "Чувствительность", "График"
CALC_REF = f"'{CALC}'!"

RUB = '#,##0;[Red]\\(#,##0\\);"-"'
NUMBER = '#,##0.0;[Red]\\(#,##0.0\\);"-"'
HOURS = '0.00;[Red]\\(0.00\\);"-"'
MONTHS = '0.0;[Red]\\(0.0\\);"-"'
SHARE = "0.0%"
FINE_SHARE = "0.00%"
TIMES = '"×"0'

FONT = "Arial"
INPUT, FORMULA, LINK = "0000FF", "000000", "008000"
HEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
TITLE_FILL = PatternFill("solid", fgColor="1F3864")
KEY_FILL = PatternFill("solid", fgColor="E2EFDA")
ASSUMPTION_FILL = PatternFill("solid", fgColor="FFFF00")
WRAP = Alignment(wrap_text=True, vertical="center")

MONTHS_SHOWN = 12
REDUCTIONS = (0.01, 0.05, 0.10, 0.25)
COST_FACTORS = (1, 3, 10)


def font(color: str = FORMULA, bold: bool = False, white: bool = False) -> Font:
    return Font(name=FONT, size=10, bold=bold, color="FFFFFF" if white else color)


def title(ws: Worksheet, cell: str, last_col: str, text: str) -> None:
    row = ws[cell].row
    ws.merge_cells(f"{cell}:{last_col}{row}")
    ws[cell] = text
    ws[cell].font = font(bold=True, white=True)
    ws[cell].fill = TITLE_FILL
    ws[cell].alignment = WRAP


def header(ws: Worksheet, row: int, labels: tuple[str, ...], first_col: int = 2) -> None:
    for offset, label in enumerate(labels):
        cell = ws.cell(row=row, column=first_col + offset, value=label)
        cell.font = font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = WRAP


def line(ws: Worksheet, row: int, values: tuple, fmt: str, color: str = FORMULA,
         fill: PatternFill | None = None, bold: bool = False) -> None:
    """Строка таблицы: показатель, значение, ед. изм., пояснение, источник."""
    for offset, value in enumerate(values):
        cell = ws.cell(row=row, column=2 + offset, value=value)
        cell.font = font(color if offset == 1 else FORMULA, bold=bold)
        cell.alignment = WRAP
        if fill is not None:
            cell.fill = fill
    ws.cell(row=row, column=3).number_format = fmt


def build_calc(ws: Worksheet, plant: PlantAssumptions) -> None:
    for col, width in {"A": 3, "B": 46, "C": 16, "D": 13, "E": 46, "F": 46, "G": 3}.items():
        ws.column_dimensions[col].width = width

    capacity = f"{plant.plant_mw:g}".replace(".", ",")
    title(ws, "B2", "F", f"Исходные данные: ПГУ-450 Ярославской ТЭС, {capacity} МВт")
    header(ws, 3, ("Показатель", "Значение", "Ед. изм.", "Пояснение", "Источник"))
    line(ws, 4, ("Простои оборудования за месяц", "=C24*C26*C27*C28/C23", "экв. ч/мес.",
                 "Недобор газовых турбин от загрязнения компрессора в часах полного простоя блока: "
                 "мощность ГТУ × недобор × доля среднего / мощность блока × часы",
                 "Расчёт по строкам 23–28"), HOURS)
    line(ws, 5, ("Стоимость одного часа простоя", "=C23*C25", "руб./ч",
                 "Выручка за час работы ПГУ на полной мощности (выручка, не прибыль)",
                 "Расчёт по строкам 23, 25"), RUB)
    line(ws, 6, ("Предполагаемое сокращение простоев", plant.reduction_share, "%",
                 "Осторожная оценка: программа возвращает 1% потерь от загрязнения. "
                 "Другие значения — на листе «Чувствительность»", "Допущение"),
         SHARE, color=INPUT, fill=ASSUMPTION_FILL)
    line(ws, 7, ("Затраты на внедрение программы", "=C33*C34*C32", "руб.",
                 "Один раз: трудоёмкость × длина рабочего дня × стоимость часа разработчика",
                 "Расчёт по строкам 32–34"), RUB)
    line(ws, 8, ("Расходы на сопровождение программы", "=C35+C36*C32", "руб./мес.",
                 "Ежемесячно: облачный сервер + часы поддержки", "Расчёт по строкам 32, 35–36"),
         RUB)

    title(ws, "B10", "F", "Расчёт экономии")
    header(ws, 11, ("Показатель", "Значение", "Ед. изм.", "Как считается"))
    calc_rows = (
        (12, "Потери от простоев сейчас", "=C4*C5", "руб./мес.", "простои × стоимость часа", RUB),
        (13, "На сколько сократятся простои", "=C4*C6", "ч/мес.", "простои × сокращение", HOURS),
        (14, "Экономия от сокращения простоев", "=C13*C5", "руб./мес.",
         "сокращённые часы × стоимость часа", RUB),
        (15, "Экономия после расходов на сопровождение", "=C14-C8", "руб./мес.",
         "экономия − сопровождение", RUB),
        (16, "Экономия за 12 месяцев", "=C15*12", "руб.", "экономия после сопровождения × 12", RUB),
        (17, "Результат за первый год", "=C16-C7", "руб.",
         "экономия за 12 месяцев − внедрение", RUB),
        (18, "Срок окупаемости", '=IF(C15>0,C7/C15,"нет")', "мес.",
         "внедрение / экономия после сопровождения", MONTHS),
        (19, "Окупится за год уже при сокращении простоев от", "=(C7/12+C8)/C12", "%",
         "(внедрение / 12 + сопровождение) / потери сейчас", FINE_SHARE),
    )
    for row, label, formula, unit, how, fmt in calc_rows:
        key = row in (15, 18)
        line(ws, row, (label, formula, unit, how), fmt, fill=KEY_FILL if key else None, bold=key)

    title(ws, "B21", "F", "Откуда исходные данные")
    header(ws, 22, ("Показатель", "Значение", "Ед. изм.", "Пояснение", "Источник"))
    sources = (
        (23, "Мощность ПГУ", plant.plant_mw, "МВт",
         "ПГУ-450: две газовые турбины ГТЭ-160 и паровая турбина 150 МВт",
         "Википедия, статья «Ярославская ТЭС»", NUMBER, INPUT),
        (24, "Мощность газовых турбин", plant.gas_turbines_mw, "МВт",
         "Две ГТЭ-160: 156,2 + 157,7 МВт. Компрессор есть только у них, паровую часть не считаем",
         "Википедия, статья «Ярославская ТЭС»", NUMBER, INPUT),
        (25, "Цена электроэнергии на рынке на сутки вперёд", plant.price_rub_per_mwh, "руб./МВт·ч",
         "Индекс равновесных цен на покупку, 1-я ценовая зона, среднее за 2024 год: 1 747,29",
         "Ассоциация «НП Совет рынка», по Neftegaz.RU от 12.03.2025", RUB, INPUT),
        (26, "Недобор мощности от загрязнения компрессора к 8 000 ч", plant.fouling_loss_at_8000h,
         "доля", "Умеренное загрязнение, кривая выходит на плато",
         "Meher-Homji, Bromley, Stalder. 2nd Middle East Turbomachinery Symposium, 2013 "
         "(turbolab.tamu.edu)", SHARE, INPUT),
        (27, "Средний недобор за цикл, доля от недобора к 8 000 ч", plant.average_loss_ratio,
         "доля", "Недобор растёт от нуля после промывки, в среднем берём половину",
         "Допущение", SHARE, INPUT),
        (28, "Часов работы на полной нагрузке в месяц", f"={HOURS_PER_YEAR}/12", "ч",
         "Весь месяц на полной нагрузке, как в данных. При меньшей загрузке эффект "
         "пропорционально меньше", "Допущение", NUMBER, FORMULA),
        (29, "Зарплата Python-разработчика", plant.salary_rub_per_month, "руб./мес.",
         "Медиана по России, 1-е полугодие 2026",
         "Хабр Карьера, career.habr.com/professions/python-razrabotchik", RUB, INPUT),
        (30, "Страховые взносы работодателя", plant.insurance_share, "доля",
         "Общий тариф в пределах базы", "Налоговый кодекс РФ", SHARE, INPUT),
        (31, "Рабочих часов в году", plant.work_hours_per_year, "ч",
         "40-часовая рабочая неделя", "Производственный календарь 2026, КонсультантПлюс",
         RUB, INPUT),
        (32, "Стоимость часа разработчика", "=C29*(1+C30)/(C31/12)", "руб./ч",
         "Оклад со взносами / рабочие часы в месяце", "Расчёт", RUB, FORMULA),
        (33, "Трудоёмкость внедрения", plant.implementation_person_days, "чел.-дней",
         "Подключение к данным АСУ ТП, обучение на данных станции, пилот", "Допущение",
         RUB, INPUT),
        (34, "Длина рабочего дня", WORKDAY_HOURS, "ч", "40 часов в неделю при пятидневке",
         "Трудовой кодекс РФ, ст. 91", RUB, INPUT),
        (35, "Облачный сервер", plant.server_rub_per_month, "руб./мес.",
         "4 vCPU, 8 ГБ памяти, SSD 50 ГБ, с НДС. "
         "Можно заменить своим компьютером в контуре станции",
         "Yandex Cloud, yandex.cloud/ru/services/compute", RUB, INPUT),
        (36, "Поддержка", plant.support_hours_per_month, "ч/мес.",
         "Переобучение модели и разбор тревог", "Допущение", RUB, INPUT),
    )
    for row, label, value, unit, note, source, fmt, color in sources:
        assumption = source == "Допущение"
        line(ws, row, (label, value, unit, note, source), fmt, color=color,
             fill=ASSUMPTION_FILL if assumption else None)

    ws["B38"] = ("Синий шрифт — исходные цифры, их можно менять; чёрный — формулы; "
                 "зелёный — ссылки на другой лист; жёлтая заливка — допущения.")
    ws["B38"].font = font()
    ws.merge_cells("B38:F38")


def build_sensitivity(ws: Worksheet) -> None:
    for col, width in {"A": 3, "B": 34, "C": 22, "D": 22, "E": 22, "F": 3}.items():
        ws.column_dimensions[col].width = width
    title(ws, "B2", "E", "Срок окупаемости, мес.: сокращение простоев × затраты на внедрение")
    ws["B3"] = "Синие ячейки можно менять, остальное берётся с листа «Расчёт»."
    ws["B3"].font = font()

    header(ws, 5, ("Затраты на внедрение: во сколько раз больше расчёта",))
    header(ws, 6, ("Затраты на внедрение, руб.",))
    header(ws, 7, ("Сокращение простоев",))
    for offset, factor in enumerate(COST_FACTORS):
        col = 3 + offset
        letter = ws.cell(row=5, column=col).column_letter
        ws.cell(row=5, column=col, value=factor).number_format = TIMES
        ws.cell(row=5, column=col).font = font(INPUT, bold=True)
        cost = ws.cell(row=6, column=col, value=f"={CALC_REF}$C$7*{letter}5")
        cost.number_format, cost.font = RUB, font(LINK)
        ws.cell(row=7, column=col, value="срок окупаемости, мес.").font = font(bold=True)
    for offset, share in enumerate(REDUCTIONS):
        row = 8 + offset
        ws.cell(row=row, column=2, value=share).number_format = "0%"
        ws.cell(row=row, column=2).font = font(INPUT, bold=True)
        net = f"($B{row}*{CALC_REF}$C$12-{CALC_REF}$C$8)"
        for col in range(3, 3 + len(COST_FACTORS)):
            letter = ws.cell(row=row, column=col).column_letter
            cell = ws.cell(row=row, column=col, value=f'=IF({net}>0,{letter}$6/{net},"нет")')
            cell.number_format, cell.font = MONTHS, font()

    title(ws, "B13", "E", "При затратах на внедрение из расчёта")
    header(ws, 14, ("Сокращение простоев", "Экономия после сопровождения, руб./мес.",
                    "Результат за первый год, руб.", "Срок окупаемости, мес."))
    for offset in range(len(REDUCTIONS)):
        row, source_row = 15 + offset, 8 + offset
        values = (f"=B{source_row}", f"=B{row}*{CALC_REF}$C$12-{CALC_REF}$C$8",
                  f"=C{row}*12-{CALC_REF}$C$7", f'=IF(C{row}>0,{CALC_REF}$C$7/C{row},"нет")')
        for col, (value, fmt) in enumerate(zip(values, ("0%", RUB, RUB, MONTHS), strict=True), 2):
            cell = ws.cell(row=row, column=col, value=value)
            cell.number_format, cell.font = fmt, font(LINK if col > 2 else FORMULA)

    ws["B20"] = "Окупится за год уже при сокращении простоев от"
    ws["B20"].font = font(bold=True)
    ws["C20"] = f"={CALC_REF}C19"
    ws["C20"].number_format, ws["C20"].font = FINE_SHARE, font(LINK, bold=True)


def payback_chart(data: Worksheet) -> LineChart:
    chart = LineChart()
    chart.title = "Накопленный результат после внедрения, руб."
    chart.title.overlay = False
    chart.add_data(Reference(data, min_col=3, min_row=4, max_row=4 + MONTHS_SHOWN))
    chart.set_categories(Reference(data, min_col=2, min_row=4, max_row=4 + MONTHS_SHOWN))
    chart.legend = None
    chart.roundedCorners = False
    chart.x_axis.title = "Месяц после внедрения"
    chart.x_axis.title.overlay = False
    chart.y_axis.numFmt = "#,##0"
    chart.x_axis.tickLblPos = "low"  # подписи месяцев внизу, а не на нулевой линии
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    series = chart.series[0]
    series.marker.symbol = "circle"
    series.smooth = False
    series.graphicalProperties.line.solidFill = "1D6FB8"
    series.graphicalProperties.line.width = 28575
    chart.width, chart.height = 17, 9
    return chart


def build_chart(ws: Worksheet, calc: Worksheet) -> None:
    for col, width in {"A": 3, "B": 12, "C": 30}.items():
        ws.column_dimensions[col].width = width
    title(ws, "B2", "C", "Данные графика")
    header(ws, 3, ("Месяц", "Накопленный результат, руб."))
    for month in range(MONTHS_SHOWN + 1):
        row = 4 + month
        ws.cell(row=row, column=2, value=month).font = font(INPUT)
        cell = ws.cell(row=row, column=3, value=f"=-{CALC_REF}$C$7+{CALC_REF}$C$15*B{row}")
        cell.number_format, cell.font = RUB, font(LINK)
    ws.add_chart(payback_chart(ws), "E3")
    calc.add_chart(payback_chart(ws), "H3")


def main() -> None:
    workbook = Workbook()
    calc = workbook.active
    calc.title = CALC
    plant = PlantAssumptions()
    build_calc(calc, plant)
    build_sensitivity(workbook.create_sheet(SENS))
    build_chart(workbook.create_sheet(CHART), calc)
    for sheet in workbook.worksheets:
        sheet.page_setup.orientation = "landscape"
        sheet.page_setup.fitToWidth, sheet.page_setup.fitToHeight = 1, 0
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT)
    print(f"Таблица сохранена: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
