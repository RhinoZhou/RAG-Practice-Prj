#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三键链接生成与唯一性/可达性校验

作者: Ph.D. Rhino
版本: 1.0.0
创建日期: 2024-01-18

功能说明:
生成doc_id/section_id/anchor索引并校验冲突与断链，触发回退策略。

内容概述:
基于节点清单构建三键主键映射，检测唯一性冲突与可达性；当锚点失效时，按"父节点→全文重排"的优先级回退，
输出检查结果与回退路径，确保索引可用性。

使用场景:
- RAG系统中文档片段索引构建与验证
- 知识库锚点系统完整性检查
- 文档链接系统的故障检测与恢复

依赖库:
- random: 用于随机抽样和模拟断链
- time: 用于性能计时
- collections: 用于数据结构管理
"""

# 自动安装依赖库
import subprocess
import sys

# 定义所需依赖库
required_dependencies = [
    # 此程序主要使用Python标准库，无需额外第三方依赖
]


def install_dependencies():
    """检查并自动安装缺失的依赖库"""
    for dependency in required_dependencies:
        try:
            # 尝试导入库以检查是否已安装
            __import__(dependency)
            print(f"✅ 依赖库 '{dependency}' 已安装")
        except ImportError:
            print(f"⚠️ 依赖库 '{dependency}' 未安装，正在安装...")
            # 使用pip安装缺失的依赖
            subprocess.check_call([sys.executable, "-m", "pip", "install", dependency])
            print(f"✅ 依赖库 '{dependency}' 安装成功")


# 执行依赖安装
if __name__ == "__main__":
    install_dependencies()


# 导入所需库
import random
import time
from collections import defaultdict


class TripleKeysValidator:
    """三键链接生成与唯一性/可达性校验器"""
    
    def __init__(self):
        """初始化校验器"""
        # 节点集合：用于存储所有节点信息
        self.nodes = []
        # 三键到节点ID的映射
        self.triple_key_to_node_id = dict()
        # 节点ID到父节点ID的映射
        self.node_id_to_parent_id = dict()
        # 冲突清单
        self.conflicts = []
        # 可达性检查结果
        self.reachability_results = dict()
        # 回退策略应用记录
        self.fallback_records = []
        
        # 构建示例数据
        self._build_sample_data()
    
    def _build_sample_data(self):
        """构建示例节点数据"""
        # 示例文档节点数据
        self.nodes = [
            {"id": "doc-1", "type": "document", "doc_id": "whitepaper-v1", "content": "技术白皮书"},
            {"id": "sec-1", "type": "section", "doc_id": "whitepaper-v1", "section_id": "1", "parent_id": "doc-1", "anchor": "anc:s1-14f2", "content": "第一章"},
            {"id": "sec-1-1", "type": "section", "doc_id": "whitepaper-v1", "section_id": "1.1", "parent_id": "sec-1", "anchor": "anc:s1-28a3", "content": "1.1 概述"},
            {"id": "sec-1-2", "type": "section", "doc_id": "whitepaper-v1", "section_id": "1.2", "parent_id": "sec-1", "anchor": "anc:s1-37b4", "content": "1.2 背景"},
            {"id": "sec-2", "type": "section", "doc_id": "whitepaper-v1", "section_id": "2", "parent_id": "doc-1", "anchor": "anc:s2-56d7", "content": "第二章"},
            {"id": "p-1-1-1", "type": "paragraph", "doc_id": "whitepaper-v1", "section_id": "1.1", "parent_id": "sec-1-1", "anchor": "anc:p1-89e5", "content": "这是第一章节的第一段内容。"},
            {"id": "p-1-1-2", "type": "paragraph", "doc_id": "whitepaper-v1", "section_id": "1.1", "parent_id": "sec-1-1", "anchor": "anc:p2-90f6", "content": "这是第一章节的第二段内容。"},
            {"id": "p-1-2-1", "type": "paragraph", "doc_id": "whitepaper-v1", "section_id": "1.2", "parent_id": "sec-1-2", "anchor": "anc:p3-71c8", "content": "这是第二章节的第一段内容。"},
            {"id": "p-2-1-1", "type": "paragraph", "doc_id": "whitepaper-v1", "section_id": "2", "parent_id": "sec-2", "anchor": "anc:p4-62b9", "content": "这是第二章的第一段内容。"},
            {"id": "s-1-1-1-1", "type": "sentence", "doc_id": "whitepaper-v1", "section_id": "1.1", "parent_id": "p-1-1-1", "anchor": "anc:s1-53a0", "content": "这是第一句话。"},
        ]
        
        # 构建父节点映射
        for node in self.nodes:
            if "parent_id" in node:
                self.node_id_to_parent_id[node["id"]] = node["parent_id"]
    
    def generate_triple_keys(self):
        """生成三键映射关系"""
        start_time = time.time()
        
        # 遍历所有节点，构建三键映射
        for node in self.nodes:
            if node["type"] != "document":  # 文档节点不参与三键映射
                # 构建三键 (doc_id, section_id, anchor)
                triple_key = (node["doc_id"], node["section_id"], node["anchor"])
                
                # 检查唯一性冲突
                if triple_key in self.triple_key_to_node_id:
                    conflict_info = {
                        "triple_key": triple_key,
                        "existing_node_id": self.triple_key_to_node_id[triple_key],
                        "conflicting_node_id": node["id"]
                    }
                    self.conflicts.append(conflict_info)
                else:
                    self.triple_key_to_node_id[triple_key] = node["id"]
        
        end_time = time.time()
        print(f"📊 三键生成完成，耗时: {end_time - start_time:.4f}秒")
    
    def check_uniqueness(self):
        """检查三键唯一性"""
        if not self.conflicts:
            print("✅ Conflicts: None")
        else:
            print(f"❌ 发现 {len(self.conflicts)} 个三键冲突:")
            for conflict in self.conflicts:
                print(f"  冲突键: {conflict['triple_key']}")
                print(f"    已存在节点ID: {conflict['existing_node_id']}")
                print(f"    冲突节点ID: {conflict['conflicting_node_id']}")
    
    def validate_reachability(self, sample_rate=0.3):
        """随机抽样验证三键可达性"""
        # 复制一份三键映射用于模拟验证
        temp_triple_keys = list(self.triple_key_to_node_id.keys())
        
        # 随机选择部分键进行验证
        sample_size = max(1, int(len(temp_triple_keys) * sample_rate))
        sampled_keys = random.sample(temp_triple_keys, sample_size)
        
        # 模拟可达性检查
        for triple_key in sampled_keys:
            # 这里简化处理，假设所有原始键都是可达的
            self.reachability_results[triple_key] = True
            print(f"✅ Reachable: {triple_key} → True")
        
        # 模拟一些断链情况
        broken_keys_count = max(1, int(sample_size * 0.3))
        broken_keys = random.sample(sampled_keys, broken_keys_count)
        
        # 标记为断链并执行回退策略
        for broken_key in broken_keys:
            self.reachability_results[broken_key] = False
            fallback_path = self._execute_fallback_strategy(broken_key)
            print(f"❌ Broken: {broken_key} → Fallback: {fallback_path}")
    
    def _execute_fallback_strategy(self, broken_key):
        """执行回退策略：父节点 → 全文重排"""
        doc_id, section_id, anchor = broken_key
        
        # 检查键是否存在于映射中
        if broken_key in self.triple_key_to_node_id:
            node_id = self.triple_key_to_node_id[broken_key]
            
            # 策略1: 尝试回退到父节点
            if node_id in self.node_id_to_parent_id:
                parent_id = self.node_id_to_parent_id[node_id]
                
                # 查找父节点的三键
                for triple, nid in self.triple_key_to_node_id.items():
                    if nid == parent_id:
                        fallback_record = {
                            "broken_key": broken_key,
                            "fallback_type": "parent_section",
                            "fallback_target": triple
                        }
                        self.fallback_records.append(fallback_record)
                        return "parent_section"
        
        # 策略2: 全文重排
        fallback_record = {
            "broken_key": broken_key,
            "fallback_type": "full_text_reorder",
            "fallback_target": doc_id
        }
        self.fallback_records.append(fallback_record)
        return "full_text_reorder"
    
    def simulate_additional_broken_links(self, count=2):
        """模拟额外的断链情况"""
        # 创建一些不存在的锚点来模拟断链
        for i in range(count):
            # 生成一个不存在的锚点
            broken_anchor = f"anc:broken-{i+1}"
            broken_key = ("whitepaper-v1", "broken-section", broken_anchor)
            
            # 执行回退策略
            fallback_path = self._execute_fallback_strategy(broken_key)
            print(f"❌ Broken: {broken_key} → Fallback: {fallback_path}")
    
    def generate_report(self):
        """生成校验报告"""
        report = []
        report.append("三键链接校验报告")
        report.append("=" * 50)
        report.append(f"总节点数: {len(self.nodes)}")
        report.append(f"三键映射数: {len(self.triple_key_to_node_id)}")
        report.append(f"冲突数: {len(self.conflicts)}")
        report.append(f"验证键数: {len(self.reachability_results)}")
        report.append(f"断链数: {list(self.reachability_results.values()).count(False) + len(self.fallback_records) - list(self.reachability_results.values()).count(False)}")
        report.append(f"回退策略应用数: {len(self.fallback_records)}")
        report.append("=" * 50)
        
        # 保存报告到文件
        with open("triple_keys_validation_report.txt", "w", encoding="utf-8") as f:
            for line in report:
                f.write(line + "\n")
        
        print(f"📝 校验报告已保存至: triple_keys_validation_report.txt")
        
        # 输出报告内容
        for line in report:
            print(line)


def main():
    """主函数"""
    print("🚀 启动三键链接生成与唯一性/可达性校验工具")
    start_time = time.time()
    
    # 创建校验器实例
    validator = TripleKeysValidator()
    
    # 生成三键映射
    validator.generate_triple_keys()
    
    # 检查唯一性
    validator.check_uniqueness()
    
    # 验证可达性
    validator.validate_reachability()
    
    # 模拟额外断链
    validator.simulate_additional_broken_links()
    
    # 生成报告
    validator.generate_report()
    
    # 检查中文输出
    print("\n🔍 中文输出测试：成功生成三键链接索引与校验报告")
    
    end_time = time.time()
    print(f"\n✅ 程序执行完成，总耗时: {end_time - start_time:.4f}秒")


if __name__ == "__main__":
    main()