# -*- coding: utf-8 -*-  
"""  
scripts/compute_quality_matrix.py  

计算样例数据的质量评分与 SLA 颜色段，输出 CSV，并展示颜色段如何驱动“告警分级”。  

用法：  
    python scripts/compute_quality_matrix.py  

依赖：  
    仅使用标准库（csv、math、statistics、pathlib）  
"""  

# 导入必要的标准库
import csv  # 用于CSV文件读写
import math  # 用于数学计算，如百分位数计算
import statistics as stats  # 用于统计计算，如平均值
from pathlib import Path  # 用于文件路径操作
from typing import Dict, List, Any, Tuple  # 用于类型注解


# ---------------------------  
# 基础 IO 模块  
# ---------------------------  

# 设置基础路径和文件路径
BASE = Path(__file__).resolve().parent  # 当前脚本所在目录
DATA = BASE / "data" / "samples.csv"  # 数据文件路径
METRIC_DICT = BASE / "config" / "metric_dictionary.yaml"  # 指标字典配置文件
BANDS = BASE / "config" / "threshold_bands.yaml"  # 阈值带配置文件
OUT_DIR = BASE / "outputs"  # 输出目录
OUT_DIR.mkdir(parents=True, exist_ok=True)  # 创建输出目录（如果不存在）

# 打印路径信息用于调试
print(f"BASE directory: {BASE}")
print(f"DATA path: {DATA}")
print(f"METRIC_DICT path: {METRIC_DICT}")
print(f"BANDS path: {BANDS}")  

def read_csv(path: Path) -> List[Dict[str, str]]:
    """
    读取CSV文件，尝试多种编码格式以处理中文乱码问题
    
    参数:
        path: CSV文件路径
    
    返回:
        包含CSV数据的字典列表，每个字典代表一行数据
    
    异常:
        ValueError: 当所有编码尝试都失败时抛出
    """
    # 尝试的编码格式列表，按优先级排序
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    
    # 依次尝试每种编码格式
    for encoding in encodings:
        try:
            print(f"Trying to read {path} with {encoding} encoding")
            with path.open("r", encoding=encoding) as f:
                # 使用csv.DictReader将每行数据转换为字典（键为列名）
                reader = csv.DictReader(f)
                rows = [row for row in reader]
                print(f"Successfully read {len(rows)} rows with {encoding} encoding")
                return rows
        except UnicodeDecodeError:
            # 捕获编码错误，尝试下一种编码
            print(f"Failed to read with {encoding} encoding, trying next...")
        except Exception as e:
            # 捕获其他异常
            print(f"Error reading with {encoding} encoding: {str(e)}")
    
    # 如果所有编码都失败，抛出异常
    raise ValueError(f"Could not read {path} with any of the tried encodings: {encodings}")  

def read_yaml_simple(path: Path) -> Dict[str, Any]:  
    """  
    极简 YAML 解析器（仅支持本案例用到的 key:value 与两级嵌套结构）  
    目的：避免引入第三方依赖（PyYAML）。若用于生产环境，建议使用PyYAML库。  
    
    参数:
        path: YAML文件路径
    
    返回:
        解析后的YAML数据，以字典形式表示
    """  
    result: Dict[str, Any] = {}  # 最终解析结果
    # 栈结构用于处理嵌套，每个元素为(缩进级别, 对应的字典)
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, result)]  
    
    with path.open("r", encoding="utf-8") as f:  
        for raw in f:  # 逐行读取
            line = raw.rstrip()  
            # 跳过空行和注释行
            if not line or line.strip().startswith("#"):  
                continue  
            
            # 计算缩进级别（用于处理嵌套结构）
            indent = len(line) - len(line.lstrip(" "))  
            
            # 根据缩进级别调整栈（处理嵌套结构的闭合）
            while stack and indent < stack[-1][0]:  
                stack.pop()  
            
            current = stack[-1][1]  # 当前操作的字典
            s = line.strip()  # 去除首尾空白
            
            # 处理键值对
            if ":" in s:  
                key, val = s.split(":", 1)  # 分割键和值
                key = key.strip()  
                val = val.strip()  
                
                if val == "":  # 如果值为空，表示开始一个新的嵌套字典
                    new = {}  
                    current[key] = new  
                    stack.append((indent + 2, new))  # 通常YAML使用2个空格缩进
                else:  # 处理字面量值
                    # 解析字符串
                    if val.startswith('"') and val.endswith('"'):  
                        val = val[1:-1]  
                    elif val.startswith("'") and val.endswith("'"):  
                        val = val[1:-1]  
                    # 解析布尔值
                    elif val.lower() in ("true", "false"):  
                        val = val.lower() == "true"  
                    else:  
                        # 尝试解析数字
                        try:  
                            if "." in val:  
                                val = float(val)  
                            else:  
                                val = int(val)  
                        except:  
                            pass  # 非数字则保持原值
                    
                    current[key] = val  
            elif s.startswith("- "):  
                # 本案例不涉及列表解析，故跳过
                pass  
    
    return result  


# ---------------------------  
# 颜色段解析与判定模块  
# ---------------------------  

def eval_band(expr: str, value: float) -> bool:  
    """
    解析并评估阈值表达式是否满足条件
    
    参数:
        expr: 阈值表达式，如 '>=0.98', '<=30', '<0.90', '>30'
        value: 要评估的实际值
    
    返回:
        布尔值，表示实际值是否满足表达式条件
    
    异常:
        ValueError: 当表达式格式不支持时抛出
    """
    # 预处理表达式：去除空白字符
    expr = expr.strip().replace(" ", "")  
    
    # 根据表达式前缀执行相应的比较操作
    if expr.startswith(">="):  
        return value >= float(expr[2:])  
    if expr.startswith("<="):  
        return value <= float(expr[2:])  
    if expr.startswith(">"):  
        return value > float(expr[1:])  
    if expr.startswith("<"):  
        return value < float(expr[1:])  
    
    # 如果表达式格式不支持，抛出异常
    raise ValueError(f"Unsupported band expression: {expr}")  

def judge_band(metric_name: str, value: float, band_cfg: Dict[str, Dict[str, str]]) -> str:  
    """
    根据指标名称、实际值和阈值配置，判定指标所属的颜色段
    
    参数:
        metric_name: 指标名称
        value: 指标的实际值
        band_cfg: 颜色段配置字典
    
    返回:
        字符串，表示颜色段：'Good' / 'Warn' / 'Bad'
    
    注：判定顺序为 Good -> Warn -> Bad，确保最严格的条件优先匹配
    """
    # 获取当前指标的阈值配置
    cfg = band_cfg.get(metric_name, {})  
    
    # 如果没有配置，默认返回Good
    if not cfg:  
        return "Good"  
    
    # 按优先级顺序判定颜色段
    if "good" in cfg and eval_band(cfg["good"], value):  
        return "Good"  
    if "warn" in cfg and eval_band(cfg["warn"], value):  
        return "Warn"  
    
    # 兜底：如果不满足Good或Warn条件，返回Bad
    return "Bad"  


# ---------------------------  
# 指标计算模块（逐记录 -> 分层聚合）  
# ---------------------------  

def compute_record_flags(row: Dict[str, str]) -> Dict[str, Any]:  
    """
    计算单条记录的五项质量指标：完整性、准确性、标准化、新鲜度延迟和可治理性
    
    参数:
        row: 包含单条记录数据的字典
    
    返回:
        包含五项质量指标的字典
    """
    # 1. 完整性：关键字段是否都有值（通过has_required_fields标记判断）
    completeness = 1.0 if row.get("has_required_fields", "").lower() == "true" else 0.0  

    # 2. 准确性：数据是否正确（通过accuracy_label标记判断）
    accuracy = 1.0 if row.get("accuracy_label", "").lower() == "correct" else 0.0  

    # 3. 标准化：格式、单位和枚举值是否符合规范（三项的平均值）
    fmt_ok = 1.0 if row.get("format_ok", "").lower() == "true" else 0.0  
    unit_ok = 1.0 if row.get("unit_ok", "").lower() == "true" else 0.0  
    enum_ok = 1.0 if row.get("enum_ok", "").lower() == "true" else 0.0  
    standardization = (fmt_ok + unit_ok + enum_ok) / 3.0  

    # 4. 新鲜度延迟：数据更新时间距现在的分钟数（原子值，后续聚合时计算p95）
    try:  
        freshness_delay = float(row.get("updated_minutes_ago", "0") or 0.0)  
    except:  
        freshness_delay = float("nan")  

    # 5. 可治理性：是否具备版本、来源和哈希值信息（三项的平均值）
    has_source = 1.0 if row.get("source") else 0.0  
    has_version = 1.0 if row.get("version") else 0.0  
    has_hash = 1.0 if row.get("hash") else 0.0  
    governability = (has_source + has_version + has_hash) / 3.0  

    # 返回计算的各项指标
    return dict(  
        completeness=completeness,  
        accuracy=accuracy,  
        standardization=standardization,  
        freshness_delay=freshness_delay,  # 注意：这是逐条记录的延迟，不是p95值  
        governability=governability,  
    )  

def p95(values: List[float]) -> float:  
    """
    计算一组数值的95百分位数
    
    参数:
        values: 数值列表
    
    返回:
        95百分位数值（如果输入为空或全是NaN，返回NaN）
    """
    # 处理空输入
    if not values:  
        return float("nan")  
    
    # 过滤掉None值和NaN值
    vs = [v for v in values if not (v is None or (isinstance(v, float) and math.isnan(v)))]  
    
    # 如果过滤后为空，返回NaN
    if not vs:  
        return float("nan")  
    
    # 排序
    vs.sort()  
    
    # 计算95百分位数的索引
    idx = int(math.ceil(0.95 * len(vs))) - 1  
    
    # 确保索引在有效范围内
    idx = max(0, min(idx, len(vs) - 1))  
    
    # 返回95百分位数值
    return float(vs[idx])  

def aggregate_slice(records: List[Dict[str, Any]]) -> Dict[str, float]:  
    """
    将若干记录的原子指标聚合为切片指标
    
    参数:
        records: 包含原子指标的记录列表
    
    返回:
        聚合后的指标字典
    
    聚合方法:
        - completeness: 平均值
        - accuracy: 平均值
        - standardization: 平均值
        - freshness_p95: 取freshness_delay的p95值
        - governability: 平均值
    """
    # 处理空输入
    if not records:  
        return {k: float("nan") for k in ["completeness", "accuracy", "standardization", "freshness_p95", "governability"]}  
    
    # 计算各项指标的平均值或p95值
    comp = stats.fmean(r["completeness"] for r in records)  # 完整性平均值
    acc = stats.fmean(r["accuracy"] for r in records)  # 准确性平均值
    stdz = stats.fmean(r["standardization"] for r in records)  # 标准化平均值
    fresh = p95([r["freshness_delay"] for r in records])  # 新鲜度p95值
    gov = stats.fmean(r["governability"] for r in records)  # 可治理性平均值
    
    # 返回聚合结果
    return dict(  
        completeness=comp,  
        accuracy=acc,  
        standardization=stdz,  
        freshness_p95=fresh,  
        governability=gov,  
    )  


# ---------------------------  
# 主流程模块  
# ---------------------------  

def main():  
    """
    主流程函数：
    1. 读取数据和配置文件
    2. 预处理数据，计算每条记录的质量指标
    3. 按文档类型和语言分层数据
    4. 聚合各切片的质量指标并判定颜色段
    5. 输出质量矩阵和质量摘要
    """
    try:  
        # 读取数据文件
        print(f"Reading data from {DATA}")  
        rows = read_csv(DATA)  
        print(f"Successfully read {len(rows)} rows from data file")  
        
        # 读取指标字典配置（本案例未直接使用，但体现了配置管理的最佳实践）
        print(f"Reading metric dictionary from {METRIC_DICT}")  
        metric_dict = read_yaml_simple(METRIC_DICT)  
        print(f"Metric dictionary keys: {list(metric_dict.keys())}")  
        
        # 读取阈值带配置
        print(f"Reading bands configuration from {BANDS}")  
        cfg = read_yaml_simple(BANDS)  
        print(f"Configuration keys: {list(cfg.keys())}")  
        bands_cfg = cfg.get("bands", {})  # 阈值带配置
        print(f"Bands configuration keys: {list(bands_cfg.keys())}")  
        # 颜色段到告警等级的映射
        alert_map = cfg.get("alerts", {}).get("mapping", {"Good": "none", "Warn": "warning", "Bad": "critical"})  
        print(f"Alert mapping: {alert_map}")  
    except Exception as e:  
        # 异常处理
        print(f"Error in main function: {str(e)}")  
        import traceback  
        traceback.print_exc()  
        return  

    # 预处理：为每条记录计算原子质量指标
    enriched = []
    for r in rows:
        # 计算单条记录的质量指标
        flags = compute_record_flags(r)
        # 创建包含原始数据和质量指标的新字典
        r2 = dict(r)
        r2.update(flags)
        enriched.append(r2)
    print(f"Enriched rows count: {len(enriched)}")
    if enriched:
        print(f"First enriched row keys: {list(enriched[0].keys())[:5]}...")  

    # 分层：按文档类型(doctype)和语言(lang)对数据进行分组
    def key_slice(r): return (r.get("doctype", ""), r.get("lang", ""))  # 定义分组键
    slices: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}  # 存储分组后的结果
    
    # 对每条记录进行分组
    for r in enriched:
        slices.setdefault(key_slice(r), []).append(r)
    print(f"Number of slices: {len(slices)}")  

    # 聚合各切片并判定颜色段
    quality_by_slice: List[Dict[str, Any]] = []  # 存储各切片的质量评估结果
    
    # 遍历每个切片
    for (doctype, lang), recs in slices.items():  
        print(f"Processing slice: doctype={doctype}, lang={lang}, records={len(recs)}")  
        
        # 聚合该切片的质量指标
        agg = aggregate_slice(recs)  
        print(f"Aggregated metrics: {agg}")  
        
        # 判定各项指标的颜色段
        bands = {
            "completeness": judge_band("completeness", agg["completeness"], bands_cfg),
            "accuracy": judge_band("accuracy", agg["accuracy"], bands_cfg),
            "standardization": judge_band("standardization", agg["standardization"], bands_cfg),
            "freshness_p95": judge_band("freshness_p95", agg["freshness_p95"], bands_cfg),
            "governability": judge_band("governability", agg["governability"], bands_cfg),
        }  
        print(f"Band judgments: {bands}")  
        
        # 构建切片质量结果对象，包含：
        # 1. 切片维度信息（文档类型、语言）
        # 2. 聚合的质量指标值
        # 3. 各项指标的颜色段
        # 4. 映射的告警等级
        # 5. 记录数量
        quality_by_slice.append({
            "doctype": doctype,
            "lang": lang,
            **agg,  # 展开聚合的指标值
            "band_completeness": bands["completeness"],
            "band_accuracy": bands["accuracy"],
            "band_standardization": bands["standardization"],
            "band_freshness_p95": bands["freshness_p95"],
            "band_governability": bands["governability"],
            "alert_completeness": alert_map.get(bands["completeness"], "none"),
            "alert_accuracy": alert_map.get(bands["accuracy"], "none"),
            "alert_standardization": alert_map.get(bands["standardization"], "none"),
            "alert_freshness_p95": alert_map.get(bands["freshness_p95"], "none"),
            "alert_governability": alert_map.get(bands["governability"], "none"),
            "records": len(recs),
        })  
    print(f"Quality by slice count: {len(quality_by_slice)}")  

    # 输出逐分层矩阵（详细的质量评估结果）
    out_matrix = OUT_DIR / "quality_matrix.csv"  # 输出文件路径
    
    # 定义CSV文件的列名
    fieldnames = [
        "doctype", "lang", "records",  # 维度和记录数
        "completeness", "accuracy", "standardization", "freshness_p95", "governability",  # 质量指标值
        "band_completeness", "band_accuracy", "band_standardization", "band_freshness_p95", "band_governability",  # 颜色段
        "alert_completeness", "alert_accuracy", "alert_standardization", "alert_freshness_p95", "alert_governability",  # 告警等级
    ]  
    
    print(f"Writing quality matrix to {out_matrix}")  
    print(f"Fieldnames: {fieldnames}")  
    if quality_by_slice:  
        print(f"First row to write: {list(quality_by_slice[0].keys())}")  
    
    # 写入CSV文件
    with out_matrix.open("w", encoding="utf-8", newline="") as f:  
        writer = csv.DictWriter(f, fieldnames=fieldnames)  
        writer.writeheader()  # 写入表头
        rows_written = 0  
        # 写入每行数据
        for row in quality_by_slice:  
            writer.writerow(row)  
            rows_written += 1  
        print(f"Wrote {rows_written} rows to quality matrix")  

    # 汇总总体质量指标（所有记录的聚合结果）
    print(f"Computing total aggregation for {len(enriched)} records")  
    total_agg = aggregate_slice(enriched)  # 聚合所有记录的质量指标
    print(f"Total aggregated metrics: {total_agg}")  
    
    # 判定总体质量指标的颜色段
    total_bands = {
        "completeness": judge_band("completeness", total_agg["completeness"], bands_cfg),
        "accuracy": judge_band("accuracy", total_agg["accuracy"], bands_cfg),
        "standardization": judge_band("standardization", total_agg["standardization"], bands_cfg),
        "freshness_p95": judge_band("freshness_p95", total_agg["freshness_p95"], bands_cfg),
        "governability": judge_band("governability", total_agg["governability"], bands_cfg),
    }  
    print(f"Total band judgments: {total_bands}")  

    # 输出总体质量摘要
    out_summary = OUT_DIR / "quality_summary.csv"  # 输出文件路径
    
    # 定义摘要CSV的列名
    sum_fields = [
        "completeness", "accuracy", "standardization", "freshness_p95", "governability",  # 总体质量指标值
        "band_completeness", "band_accuracy", "band_standardization", "band_freshness_p95", "band_governability"  # 总体颜色段
    ]  
    
    print(f"Writing quality summary to {out_summary}")  
    # 写入总体质量摘要CSV
    with out_summary.open("w", encoding="utf-8", newline="") as f:  
        writer = csv.DictWriter(f, fieldnames=sum_fields)  
        writer.writeheader()  # 写入表头
        
        # 构建摘要行数据
        summary_row = {
            "completeness": total_agg["completeness"],
            "accuracy": total_agg["accuracy"],
            "standardization": total_agg["standardization"],
            "freshness_p95": total_agg["freshness_p95"],
            "governability": total_agg["governability"],
            "band_completeness": total_bands["completeness"],
            "band_accuracy": total_bands["accuracy"],
            "band_standardization": total_bands["standardization"],
            "band_freshness_p95": total_bands["freshness_p95"],
            "band_governability": total_bands["governability"],
        }  
        
        print(f"Summary row to write: {summary_row}")  
        writer.writerow(summary_row)  # 写入摘要行
        print("Successfully wrote quality summary")  

    # 控制台演示：颜色段到告警等级的映射关系
    print("=" * 80)  
    print("总体质量摘要已输出到:", out_summary)  
    print("- 颜色段到告警等级的映射示例：Good → none；Warn → warning；Bad → critical")  


# 当脚本直接运行时，执行main函数
if __name__ == "__main__":  
    main()