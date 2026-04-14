import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import openpyxl
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_problem_cell(cell) -> bool:
    fill = cell.fill
    if fill is None:
        return False

    fill_type = (fill.fill_type or fill.patternType or "").lower()
    if fill_type in {"", "none"}:
        return False

    # 默认未填充通常为 fgColor rgb=00000000，且 fill_type 为空；
    # 这里 fill_type 已过滤为空的情况，保留其余填充作为“已标色”。
    return True


def find_required_columns(headers: Iterable[object]) -> Dict[str, int]:
    header_map: Dict[str, int] = {}
    for idx, raw in enumerate(headers, start=1):
        text = normalize_text(raw)
        if not text:
            continue
        if "单位名称" in text and "unit" not in header_map:
            header_map["unit"] = idx
        if "省" in text and text in {"省份", "省", "省级"} and "province" not in header_map:
            header_map["province"] = idx
        if "地市" in text and "city" not in header_map:
            header_map["city"] = idx
        if "区县" in text and "county" not in header_map:
            header_map["county"] = idx
        if "重点行业" in text and "problem_industry" not in header_map:
            header_map["problem_industry"] = idx

    required = ["unit", "city", "county", "problem_industry"]
    missing = [k for k in required if k not in header_map]
    if missing:
        raise ValueError(f"未找到必要列: {missing}，请检查表头是否包含单位名称/地市/区县/重点行业")

    return header_map


def write_xlsx(path: Path, sheet_name: str, headers: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(list(headers))
    for row in rows:
        ws.append(list(row))
    wb.save(path)


def write_city_county_summary_with_charts(
    path: Path,
    headers: Iterable[str],
    rows: Iterable[Iterable[object]],
    top5_cities: Iterable[Tuple[str, int]],
    top5_industries: Iterable[Tuple[str, int]],
) -> None:
    top5_cities = list(top5_cities)
    top5_industries = list(top5_industries)

    wb = Workbook()
    ws = wb.active
    ws.title = "地市区县汇总"
    ws.append(list(headers))
    for row in rows:
        ws.append(list(row))

    chart_ws = wb.create_sheet("图表")
    chart_ws.append(["Top5问题城市", "异常行业数量"])
    city_start_row = 2
    for city, count in top5_cities:
        chart_ws.append([city, count])

    industry_title_row = city_start_row + 7
    chart_ws.cell(row=industry_title_row, column=1, value="Top5问题行业")
    chart_ws.cell(row=industry_title_row, column=2, value="出现频次")
    industry_start_row = industry_title_row + 1
    for industry, count in top5_industries:
        chart_ws.append([industry, count])

    city_end_row = city_start_row + 4
    city_chart = BarChart()
    city_chart.type = "col"
    city_chart.title = "Top5问题城市（异常行业数量）"
    city_chart.y_axis.title = "异常行业数量"
    city_chart.x_axis.title = "城市"
    city_chart.y_axis.delete = False
    city_chart.x_axis.delete = False
    city_max = max([count for _, count in top5_cities], default=1)
    city_chart.y_axis.scaling.min = 0
    city_chart.y_axis.scaling.max = max(1, city_max + 1)
    city_chart.y_axis.majorUnit = max(1, math.ceil(city_max / 5))
    city_data = Reference(chart_ws, min_col=2, min_row=1, max_row=city_end_row)
    city_cats = Reference(chart_ws, min_col=1, min_row=city_start_row, max_row=city_end_row)
    city_chart.add_data(city_data, titles_from_data=True)
    city_chart.set_categories(city_cats)
    city_chart.height = 7
    city_chart.width = 12
    chart_ws.add_chart(city_chart, "D2")

    industry_end_row = industry_start_row + 4
    industry_chart = BarChart()
    industry_chart.type = "col"
    industry_chart.title = "Top5问题行业（出现频次）"
    industry_chart.y_axis.title = "出现频次"
    industry_chart.x_axis.title = "行业"
    industry_chart.y_axis.delete = False
    industry_chart.x_axis.delete = False
    industry_max = max([count for _, count in top5_industries], default=1)
    industry_chart.y_axis.scaling.min = 0
    industry_chart.y_axis.scaling.max = max(1, industry_max + math.ceil(industry_max * 0.1))
    industry_chart.y_axis.majorUnit = max(1, math.ceil(industry_max / 5))
    industry_data = Reference(chart_ws, min_col=2, min_row=industry_title_row, max_row=industry_end_row)
    industry_cats = Reference(chart_ws, min_col=1, min_row=industry_start_row, max_row=industry_end_row)
    industry_chart.add_data(industry_data, titles_from_data=True)
    industry_chart.set_categories(industry_cats)
    industry_chart.height = 7
    industry_chart.width = 12
    chart_ws.add_chart(industry_chart, "D20")

    wb.save(path)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="识别重点行业列中被标色的单元格，并汇总问题行业及企业数量。"
    )
    parser.add_argument(
        "excel_path",
        nargs="?",
        default="集群 - 一类集群和二类集群（终版）-20240830.xlsx",
        help="输入 Excel 文件路径",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="工作表名称（默认使用第一个工作表）",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="输出目录（默认脚本同目录）",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel_path)
    if not excel_path.is_absolute():
        excel_path = script_dir / excel_path
    if not excel_path.exists():
        raise FileNotFoundError(f"找不到文件: {excel_path}")

    wb = openpyxl.load_workbook(excel_path, data_only=True)
    ws = wb[args.sheet] if args.sheet else wb[wb.sheetnames[0]]

    headers = [ws.cell(1, col).value for col in range(1, ws.max_column + 1)]
    cols = find_required_columns(headers)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    problem_rows = []
    industry_frequency: Counter[str] = Counter()

    # key: (province, city, county)
    by_city_county: Dict[Tuple[str, str, str], Dict[str, Set[str]]] = defaultdict(
        lambda: {"industries": set(), "units": set()}
    )

    # key: (province, city)
    by_city: Dict[Tuple[str, str], Dict[str, Set[str]]] = defaultdict(
        lambda: {"industries": set(), "units": set(), "counties": set()}
    )

    for row_idx in range(2, ws.max_row + 1):
        industry_cell = ws.cell(row_idx, cols["problem_industry"])
        industry = normalize_text(industry_cell.value)

        if not industry:
            continue
        if not is_problem_cell(industry_cell):
            continue

        province = normalize_text(ws.cell(row_idx, cols.get("province", -1)).value) if "province" in cols else ""
        city = normalize_text(ws.cell(row_idx, cols["city"]).value)
        county = normalize_text(ws.cell(row_idx, cols["county"]).value)
        unit = normalize_text(ws.cell(row_idx, cols["unit"]).value)

        key_city_county = (province, city, county)
        by_city_county[key_city_county]["industries"].add(industry)
        if unit:
            by_city_county[key_city_county]["units"].add(unit)

        key_city = (province, city)
        by_city[key_city]["industries"].add(industry)
        by_city[key_city]["counties"].add(county)
        if unit:
            by_city[key_city]["units"].add(unit)

        problem_rows.append([province, city, county, unit, industry, row_idx])
        industry_frequency[industry] += 1

    detail_path = output_dir / "problem_records.xlsx"
    write_xlsx(
        detail_path,
        "问题记录明细",
        ["省份", "地市", "区县", "单位名称", "问题行业", "Excel行号"],
        problem_rows,
    )

    city_county_rows = []
    for (province, city, county), stats in sorted(by_city_county.items()):
        industries = sorted(stats["industries"])
        units = stats["units"]
        city_county_rows.append(
            [
                province,
                city,
                county,
                "、".join(industries),
                len(industries),
                len(units),
            ]
        )

    top5_cities = sorted(
        [(city, len(stats["industries"])) for (_, city), stats in by_city.items()],
        key=lambda x: (-x[1], x[0]),
    )[:5]
    top5_industries = industry_frequency.most_common(5)

    city_county_path = output_dir / "city_county_summary.xlsx"
    write_city_county_summary_with_charts(
        city_county_path,
        ["省份", "地市", "区县", "问题行业列表", "问题行业数量", "问题企业数量"],
        city_county_rows,
        top5_cities,
        top5_industries,
    )

    city_rows = []
    for (province, city), stats in sorted(by_city.items()):
        industries = sorted(stats["industries"])
        counties = sorted([c for c in stats["counties"] if c])
        units = stats["units"]
        city_rows.append(
            [
                province,
                city,
                "、".join(counties),
                len(counties),
                "、".join(industries),
                len(industries),
                len(units),
            ]
        )

    city_path = output_dir / "city_summary.xlsx"
    write_xlsx(
        city_path,
        "地市汇总",
        ["省份", "地市", "涉及区县列表", "涉及区县数量", "问题行业列表", "问题行业数量", "问题企业数量"],
        city_rows,
    )

    print(f"处理完成: {excel_path}")
    print(f"工作表: {ws.title}")
    print(f"识别到问题记录数: {len(problem_rows)}")
    print(f"涉及城市区县数: {len(by_city_county)}")
    print(f"涉及城市数: {len(by_city)}")
    print(f"输出文件:\n- {detail_path}\n- {city_county_path}\n- {city_path}")


if __name__ == "__main__":
    main()

