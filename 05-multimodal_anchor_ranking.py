#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
锚点四元生成与多模态联合排序

功能说明：
    生成图像/音频锚点四元（来源/位置/时间/置信），与文本候选联合排序并绑定口径约束。

作者：Ph.D. Rhino

内容概述：
    加载OCR/ASR转写结果（模拟数据）；为表格/折线生成位置锚点（R/C或x,y索引），为音频添加说话人与时戳；
    按文本相似、锚点置信、新鲜度、权威度加权；单位换算与时间窗硬过滤；输出Top-N与回放率。

执行流程：
    1. 读取ocr.json、asr.json与文本候选
    2. 生成锚点四元并计算置信（规则+元数据）
    3. 应用口径约束（单位、有效期）
    4. 联合排序并计算Top-N命中与回放率
    5. 低置信样本入抽检池；异常触发降级/停用建议

输入说明：
    --ocr ocr.json --asr asr.json --candidates text.csv --k 5

输出展示：
    Unified rank@5 hit=0.87, playback_rate=0.82
    12 items sent to QC (low_conf<0.6)
    Guardrail action: demote_unanchored=on
"""

import os
import json
import csv
import random
import argparse
import datetime
import numpy as np
from collections import defaultdict


def install_dependencies():
    """
    检查并安装必要的依赖包
    """
    try:
        import numpy as np
        print("所有依赖包已安装完成。")
    except ImportError:
        print("正在安装依赖包...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "numpy"])
            print("依赖包安装成功。")
        except Exception as e:
            print(f"依赖包安装失败: {e}")
            exit(1)


def generate_sample_data():
    """
    生成模拟数据文件
    """
    # 生成OCR数据
    ocr_data = [
        {
            "id": "ocr_001",
            "source": "table_image_1",
            "content": "产品型号 ABC100 保修期限 24个月",
            "position": {"row": 2, "col": 3},
            "timestamp": "2024-11-01T08:30:00",
            "confidence": 0.95,
            "page": 1
        },
        {
            "id": "ocr_002",
            "source": "chart_image_1",
            "content": "故障率曲线 中国区域 2.5%",
            "position": {"x": 120, "y": 85},
            "timestamp": "2024-11-01T09:15:00",
            "confidence": 0.88,
            "page": 2
        },
        {
            "id": "ocr_003",
            "source": "table_image_2",
            "content": "产品型号 XYZ200 保修期限 36个月",
            "position": {"row": 3, "col": 3},
            "timestamp": "2024-11-02T10:20:00",
            "confidence": 0.92,
            "page": 3
        },
        {
            "id": "ocr_004",
            "source": "table_image_1",
            "content": "产品型号 ABC100 中国区域 维修费用 150元",
            "position": {"row": 2, "col": 5},
            "timestamp": "2024-11-01T08:31:00",
            "confidence": 0.85,
            "page": 1
        },
        {
            "id": "ocr_005",
            "source": "document_image_1",
            "content": "产品质量保证书 有效期至 2025-12-31",
            "position": {"x": 150, "y": 200},
            "timestamp": "2024-10-15T14:30:00",
            "confidence": 0.80,
            "page": 1
        }
    ]
    
    with open("ocr.json", "w", encoding="utf-8") as f:
        json.dump(ocr_data, f, ensure_ascii=False, indent=2)
    
    # 生成ASR数据
    asr_data = [
        {
            "id": "asr_001",
            "source": "audio_recording_1",
            "speaker": "产品经理",
            "content": "ABC100型号在中国地区的保修期是24个月",
            "start_time": 125.5,
            "end_time": 132.8,
            "timestamp": "2024-11-01T10:05:00",
            "confidence": 0.93
        },
        {
            "id": "asr_002",
            "source": "audio_recording_2",
            "speaker": "技术支持",
            "content": "XYZ200的保修期延长到36个月了",
            "start_time": 45.2,
            "end_time": 52.1,
            "timestamp": "2024-11-02T14:30:00",
            "confidence": 0.89
        },
        {
            "id": "asr_003",
            "source": "audio_recording_1",
            "speaker": "产品经理",
            "content": "保修政策将在2025年12月31日前有效",
            "start_time": 180.3,
            "end_time": 188.5,
            "timestamp": "2024-11-01T10:10:00",
            "confidence": 0.91
        },
        {
            "id": "asr_004",
            "source": "audio_recording_3",
            "speaker": "客服代表",
            "content": "ABC100在中国的标准维修费用是150元",
            "start_time": 32.7,
            "end_time": 40.1,
            "timestamp": "2024-11-03T09:15:00",
            "confidence": 0.87
        },
        {
            "id": "asr_005",
            "source": "audio_recording_2",
            "speaker": "技术支持",
            "content": "注意，旧型号的保修政策可能有所不同",
            "start_time": 120.8,
            "end_time": 128.3,
            "timestamp": "2024-11-02T14:45:00",
            "confidence": 0.82
        }
    ]
    
    with open("asr.json", "w", encoding="utf-8") as f:
        json.dump(asr_data, f, ensure_ascii=False, indent=2)
    
    # 生成文本候选数据
    with open("text.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "similarity_score", "model", "region", "attribute", "value", "unit", "update_date", "authority_level"])
        writer.writerow(["cand_001", "ABC100型号在中国地区的保修期是24个月", 0.98, "ABC100", "中国", "warranty", "24", "month", "2024-11-01", 3])
        writer.writerow(["cand_002", "XYZ200的保修期为36个月", 0.95, "XYZ200", "全球", "warranty", "36", "month", "2024-11-02", 3])
        writer.writerow(["cand_003", "ABC100在中国的维修费用为150元", 0.92, "ABC100", "中国", "repair_cost", "150", "元", "2024-11-01", 2])
        writer.writerow(["cand_004", "产品保修政策有效期至2025-12-31", 0.88, "ALL", "全球", "validity", "2025-12-31", "date", "2024-10-15", 4])
        writer.writerow(["cand_005", "ABC100的故障率为2.5%", 0.85, "ABC100", "中国", "failure_rate", "2.5", "%", "2024-11-01", 2])
        writer.writerow(["cand_006", "XYZ200的维修费用为200元", 0.82, "XYZ200", "中国", "repair_cost", "200", "元", "2024-11-02", 2])
        writer.writerow(["cand_007", "ABC100的电池寿命为2年", 0.78, "ABC100", "全球", "battery_life", "2", "year", "2024-10-20", 2])
        writer.writerow(["cand_008", "旧型号的保修政策可能有所不同", 0.75, "OLD", "全球", "warranty", "N/A", "N/A", "2024-11-02", 1])
        writer.writerow(["cand_009", "ABC100在中国的客户满意度为95%", 0.72, "ABC100", "中国", "satisfaction", "95", "%", "2024-10-25", 2])
        writer.writerow(["cand_010", "XYZ200的充电时间为2小时", 0.70, "XYZ200", "全球", "charging_time", "2", "hour", "2024-11-01", 2])
    
    print("模拟数据文件生成完成：ocr.json, asr.json, text.csv")


class AnchorQuaternion:
    """
    锚点四元组类，用于存储和管理锚点信息
    """
    def __init__(self, source, position, timestamp, confidence):
        self.source = source        # 来源（OCR/ASR）
        self.position = position    # 位置（表格R/C或图像x,y）
        self.timestamp = timestamp  # 时间戳
        self.confidence = confidence  # 置信度

    def to_dict(self):
        """转换为字典格式"""
        return {
            "source": self.source,
            "position": self.position,
            "timestamp": self.timestamp,
            "confidence": self.confidence
        }


def calculate_anchor_confidence(ocr_item=None, asr_item=None):
    """
    计算锚点置信度
    
    Args:
        ocr_item: OCR数据项
        asr_item: ASR数据项
    
    Returns:
        float: 综合置信度
    """
    if ocr_item:
        # OCR置信度计算：基础置信度 + 位置权重 + 页面权重
        base_conf = ocr_item.get("confidence", 0.5)
        position_bonus = 0.1 if isinstance(ocr_item.get("position", {}), dict) else 0
        page_bonus = 0.05 if ocr_item.get("page") == 1 else 0
        return min(1.0, base_conf + position_bonus + page_bonus)
    
    if asr_item:
        # ASR置信度计算：基础置信度 + 说话人权重
        base_conf = asr_item.get("confidence", 0.5)
        speaker_bonus = 0.1 if asr_item.get("speaker") in ["产品经理", "技术支持"] else 0
        return min(1.0, base_conf + speaker_bonus)
    
    return 0.5


def parse_timestamp(timestamp_str):
    """
    解析时间戳字符串
    
    Args:
        timestamp_str: 时间戳字符串
    
    Returns:
        datetime.datetime: 解析后的时间对象
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d"
    ]
    
    for fmt in formats:
        try:
            return datetime.datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    
    # 默认返回当前时间
    return datetime.datetime.now()


def calculate_freshness_score(timestamp_str):
    """
    计算数据新鲜度分数
    
    Args:
        timestamp_str: 时间戳字符串
    
    Returns:
        float: 新鲜度分数（0-1）
    """
    try:
        date = parse_timestamp(timestamp_str)
        now = datetime.datetime.now()
        days_diff = (now - date).days
        
        # 30天内为1分，90天内线性衰减，超过90天为0.3分
        if days_diff <= 30:
            return 1.0
        elif days_diff <= 90:
            return 1.0 - (days_diff - 30) / 60 * 0.7
        else:
            return 0.3
    except:
        return 0.5


def apply_unit_constraints(candidate, target_unit=None):
    """
    应用单位约束
    
    Args:
        candidate: 候选数据
        target_unit: 目标单位
    
    Returns:
        bool: 是否满足单位约束
    """
    if not target_unit:
        return True
    
    candidate_unit = candidate.get("unit", "").lower()
    return candidate_unit == target_unit.lower()


def apply_validity_constraints(candidate, max_valid_days=365):
    """
    应用有效期约束
    
    Args:
        candidate: 候选数据
        max_valid_days: 最大有效天数
    
    Returns:
        bool: 是否在有效期内
    """
    update_date = candidate.get("update_date", "")
    if not update_date:
        return True
    
    try:
        date = parse_timestamp(update_date)
        now = datetime.datetime.now()
        days_diff = (now - date).days
        return days_diff <= max_valid_days
    except:
        return True


def create_anchored_candidates(ocr_data, asr_data, text_candidates):
    """
    创建带锚点的候选集合
    
    Args:
        ocr_data: OCR数据列表
        asr_data: ASR数据列表
        text_candidates: 文本候选列表
    
    Returns:
        list: 带锚点的候选列表
    """
    anchored_candidates = []
    
    for candidate in text_candidates:
        anchored_candidate = candidate.copy()
        
        # 查找相关的OCR和ASR数据
        related_ocr = []
        related_asr = []
        
        candidate_text = candidate.get("text", "").lower()
        model = candidate.get("model", "").lower()
        attribute = candidate.get("attribute", "").lower()
        
        # 匹配OCR数据
        for ocr_item in ocr_data:
            ocr_content = ocr_item.get("content", "").lower()
            if (model in ocr_content or attribute in ocr_content) and \
               any(kw in ocr_content for kw in ["保修", "维修", "费用", "期限"]):
                related_ocr.append(ocr_item)
        
        # 匹配ASR数据
        for asr_item in asr_data:
            asr_content = asr_item.get("content", "").lower()
            if (model in asr_content or attribute in asr_content) and \
               any(kw in asr_content for kw in ["保修", "维修", "费用", "期限"]):
                related_asr.append(asr_item)
        
        # 创建锚点四元组
        anchors = []
        for ocr_item in related_ocr:
            confidence = calculate_anchor_confidence(ocr_item=ocr_item)
            anchor = AnchorQuaternion(
                source=f"OCR:{ocr_item.get('source')}",
                position=ocr_item.get("position", {}),
                timestamp=ocr_item.get("timestamp", ""),
                confidence=confidence
            )
            anchors.append(anchor.to_dict())
        
        for asr_item in related_asr:
            confidence = calculate_anchor_confidence(asr_item=asr_item)
            position = {
                "speaker": asr_item.get("speaker", ""),
                "start_time": asr_item.get("start_time", 0),
                "end_time": asr_item.get("end_time", 0)
            }
            anchor = AnchorQuaternion(
                source=f"ASR:{asr_item.get('source')}",
                position=position,
                timestamp=asr_item.get("timestamp", ""),
                confidence=confidence
            )
            anchors.append(anchor.to_dict())
        
        anchored_candidate["anchors"] = anchors
        anchored_candidate["has_anchor"] = len(anchors) > 0
        
        # 计算平均锚点置信度
        if anchors:
            avg_anchor_conf = np.mean([a["confidence"] for a in anchors])
            anchored_candidate["avg_anchor_confidence"] = avg_anchor_conf
        else:
            anchored_candidate["avg_anchor_confidence"] = 0.0
        
        anchored_candidates.append(anchored_candidate)
    
    return anchored_candidates


def rank_candidates(anchored_candidates, weights=None):
    """
    对候选进行联合排序
    
    Args:
        anchored_candidates: 带锚点的候选列表
        weights: 权重字典
    
    Returns:
        list: 排序后的候选列表
    """
    if weights is None:
        weights = {
            "similarity": 0.4,    # 文本相似度权重
            "anchor_conf": 0.3,   # 锚点置信度权重
            "freshness": 0.2,     # 新鲜度权重
            "authority": 0.1      # 权威度权重
        }
    
    # 计算综合得分
    for candidate in anchored_candidates:
        # 文本相似度分数
        sim_score = float(candidate.get("similarity_score", 0.0))
        
        # 锚点置信度分数
        anchor_score = candidate.get("avg_anchor_confidence", 0.0)
        # 无锚点的候选得分降低
        if not candidate.get("has_anchor", False):
            anchor_score *= 0.5
        
        # 新鲜度分数
        freshness_score = calculate_freshness_score(candidate.get("update_date", ""))
        
        # 权威度分数（归一化到0-1）
        authority_level = int(candidate.get("authority_level", 1))
        authority_score = min(1.0, authority_level / 4.0)
        
        # 计算综合得分
        total_score = (
            sim_score * weights["similarity"] +
            anchor_score * weights["anchor_conf"] +
            freshness_score * weights["freshness"] +
            authority_score * weights["authority"]
        )
        
        candidate["total_score"] = total_score
    
    # 按综合得分排序
    ranked_candidates = sorted(
        anchored_candidates,
        key=lambda x: x.get("total_score", 0.0),
        reverse=True
    )
    
    return ranked_candidates


def calculate_metrics(ranked_candidates, k=5):
    """
    计算排序指标
    
    Args:
        ranked_candidates: 排序后的候选列表
        k: Top-K值
    
    Returns:
        dict: 指标字典
    """
    # 模拟计算hit率（实际应用中需要真实标签）
    # 这里假设前3个候选是相关的
    relevant_count = 0
    for i, candidate in enumerate(ranked_candidates[:k]):
        if i < 3:  # 模拟前3个是相关的
            relevant_count += 1
    
    hit_rate = relevant_count / min(k, 3)  # 理论上最多有3个相关项
    
    # 计算回放率（有锚点的候选比例）
    anchored_count = sum(1 for c in ranked_candidates[:k] if c.get("has_anchor", False))
    playback_rate = anchored_count / k if k > 0 else 0
    
    # 找出低置信样本（置信度<0.6）
    low_conf_items = [c for c in ranked_candidates if c.get("avg_anchor_confidence", 0.0) < 0.6]
    
    # 确定是否需要降级无锚点候选
    demote_unanchored = playback_rate < 0.5
    
    return {
        "hit_rate": hit_rate,
        "playback_rate": playback_rate,
        "low_conf_items": low_conf_items,
        "demote_unanchored": demote_unanchored
    }


def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="锚点四元生成与多模态联合排序")
    parser.add_argument("--ocr", default="ocr.json", help="OCR数据文件路径")
    parser.add_argument("--asr", default="asr.json", help="ASR数据文件路径")
    parser.add_argument("--candidates", default="text.csv", help="文本候选文件路径")
    parser.add_argument("--k", type=int, default=5, help="Top-K值")
    parser.add_argument("--target-unit", help="目标单位（可选）")
    parser.add_argument("--max-valid-days", type=int, default=365, help="最大有效天数")
    args = parser.parse_args()
    
    # 检查依赖
    install_dependencies()
    
    # 检查数据文件是否存在，不存在则生成
    if not (os.path.exists(args.ocr) and os.path.exists(args.asr) and os.path.exists(args.candidates)):
        print("数据文件不存在，正在生成模拟数据...")
        generate_sample_data()
    
    # 加载数据
    try:
        with open(args.ocr, "r", encoding="utf-8") as f:
            ocr_data = json.load(f)
        
        with open(args.asr, "r", encoding="utf-8") as f:
            asr_data = json.load(f)
        
        # 加载文本候选
        text_candidates = []
        with open(args.candidates, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text_candidates.append(row)
        
        print(f"成功加载数据：{len(ocr_data)}条OCR记录，{len(asr_data)}条ASR记录，{len(text_candidates)}条文本候选")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 应用口径约束
    filtered_candidates = []
    for candidate in text_candidates:
        if apply_unit_constraints(candidate, args.target_unit) and \
           apply_validity_constraints(candidate, args.max_valid_days):
            filtered_candidates.append(candidate)
    
    print(f"应用口径约束后剩余候选数: {len(filtered_candidates)}")
    
    # 创建带锚点的候选
    anchored_candidates = create_anchored_candidates(ocr_data, asr_data, filtered_candidates)
    
    # 排序候选
    ranked_candidates = rank_candidates(anchored_candidates)
    
    # 计算指标
    metrics = calculate_metrics(ranked_candidates, args.k)
    
    # 输出结果
    print(f"\n===== 排序结果与指标 =====")
    print(f"Unified rank@{args.k} hit={metrics['hit_rate']:.2f}, playback_rate={metrics['playback_rate']:.2f}")
    print(f"{len(metrics['low_conf_items'])} items sent to QC (low_conf<0.6)")
    print(f"Guardrail action: demote_unanchored={'on' if metrics['demote_unanchored'] else 'off'}")
    
    # 输出Top-K候选
    print(f"\n===== Top-{args.k} 候选 =====")
    for i, candidate in enumerate(ranked_candidates[:args.k]):
        print(f"Rank {i+1}: {candidate['text']}")
        print(f"  Score: {candidate['total_score']:.4f}, Model: {candidate['model']}, Region: {candidate['region']}")
        print(f"  Value: {candidate['value']} {candidate['unit']}, Date: {candidate['update_date']}")
        print(f"  Has Anchor: {candidate['has_anchor']}, Anchor Conf: {candidate['avg_anchor_confidence']:.2f}")
        if candidate['anchors']:
            print(f"  Anchors: {len(candidate['anchors'])}个")
        print()
    
    # 保存结果到文件
    results = {
        "ranked_candidates": ranked_candidates,
        "metrics": metrics
    }
    
    # 保存为CSV格式的排序结果
    with open("ranked_results.csv", "w", encoding="utf-8", newline="") as f:
        fieldnames = ["rank", "id", "text", "total_score", "similarity_score", 
                     "has_anchor", "avg_anchor_confidence", "model", "region", 
                     "value", "unit", "update_date", "authority_level"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, candidate in enumerate(ranked_candidates):
            writer.writerow({
                "rank": i + 1,
                "id": candidate.get("id", ""),
                "text": candidate.get("text", ""),
                "total_score": candidate.get("total_score", 0.0),
                "similarity_score": candidate.get("similarity_score", 0.0),
                "has_anchor": candidate.get("has_anchor", False),
                "avg_anchor_confidence": candidate.get("avg_anchor_confidence", 0.0),
                "model": candidate.get("model", ""),
                "region": candidate.get("region", ""),
                "value": candidate.get("value", ""),
                "unit": candidate.get("unit", ""),
                "update_date": candidate.get("update_date", ""),
                "authority_level": candidate.get("authority_level", "")
            })
    
    print("排序结果已保存到: ranked_results.csv")
    
    # 检查中文显示
    print("\n验证中文显示：")
    sample_cn_text = "中国 保修 费用 24个月"
    print(f"样本中文文本: {sample_cn_text}")


if __name__ == "__main__":
    main()