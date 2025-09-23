#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查询DAG最小演示程序

作者：Ph.D. Rhino

功能说明：
  本程序实现了查询意图的DAG(有向无环图)分解与执行框架，将查询意图拆分为多个功能节点（如self_query、sql、
  cypher、hybrid_retrieval、merge等），并按照拓扑顺序执行这些节点，最终汇总得到查询结果。

主要功能：
  1. 意图分析：识别查询类型和所需执行的节点
  2. DAG构建：构建节点之间的依赖关系图
  3. 拓扑排序：计算节点的执行顺序
  4. 节点执行：按照拓扑顺序执行各个功能节点
  5. 结果合并：汇总各节点执行结果并输出最终结果

执行流程：
  分析意图并确定节点 -> 构建依赖图 -> 拓扑排序执行 -> 输出 plan 与最终结果

执行效率优化：
  采用高效的拓扑排序算法，使用邻接表表示图结构，确保程序执行效率
"""

import re
import random
import time
import os
import sys
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict, deque

# 设置随机种子，确保结果可复现
random.seed(42)

# 依赖检查与自动安装
def check_dependencies():
    """检查必要的依赖并自动安装"""
    try:
        # 本程序主要使用Python标准库，无额外依赖
        print("依赖检查通过，无需额外安装依赖包。")
        return True
    except Exception as e:
        print(f"依赖检查失败: {e}")
        return False

class QueryNode:
    """查询节点基类，代表DAG中的一个功能节点"""
    def __init__(self, node_id: str, node_type: str):
        """初始化查询节点
        
        Args:
            node_id: 节点唯一标识符
            node_type: 节点类型（如'self_query', 'sql', 'cypher', 'hybrid_retrieval', 'merge'等）
        """
        self.node_id = node_id
        self.node_type = node_type
        self.inputs = []  # 输入数据列表
        self.outputs = None  # 输出结果
        self.is_executed = False  # 执行状态标记
        self.execution_time = 0.0  # 执行时间（毫秒）
    
    def execute(self, context: Dict[str, Any]) -> Any:
        """执行节点逻辑
        
        Args:
            context: 执行上下文，包含全局信息和共享数据
        
        Returns:
            节点执行结果
        """
        start_time = time.time()
        
        # 子类需要实现具体的执行逻辑
        result = self._execute_impl(context)
        
        # 记录执行时间
        self.execution_time = (time.time() - start_time) * 1000
        self.is_executed = True
        self.outputs = result
        
        print(f"节点 [{self.node_id}({self.node_type})] 执行完成，耗时: {self.execution_time:.2f}毫秒")
        return result
    
    def _execute_impl(self, context: Dict[str, Any]) -> Any:
        """节点执行的具体实现，由子类重写
        
        Args:
            context: 执行上下文
        
        Returns:
            执行结果
        """
        raise NotImplementedError("子类必须实现_execute_impl方法")

class SelfQueryNode(QueryNode):
    """SelfQuery节点，实现带有过滤器的语义检索"""
    def __init__(self, node_id: str = "self_query_1"):
        super().__init__(node_id, "self_query")
    
    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行SelfQuery检索
        
        Args:
            context: 执行上下文
        
        Returns:
            检索结果，包含文档ID列表和路由信息
        """
        # 获取查询
        query = context.get("query", "")
        
        # 模拟提取过滤条件（实际应用中应使用更复杂的NLP技术）
        filters = {}
        if "类型" in query or "种类" in query:
            filters["type"] = "article"
        if "日期" in query or "时间" in query:
            filters["date"] = "2023-2024"
        
        # 模拟检索结果
        docs = [f"doc_{i}" for i in random.sample(range(1, 100), 5)]
        
        return {
            "docs": docs,
            "filters": filters,
            "route": "self_query",
            "score": round(random.uniform(0.7, 0.9), 2)
        }

class SQLNode(QueryNode):
    """SQL节点，实现结构化数据查询"""
    def __init__(self, node_id: str = "sql_1"):
        super().__init__(node_id, "sql")
    
    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行SQL查询
        
        Args:
            context: 执行上下文
        
        Returns:
            查询结果，包含数据记录和SQL语句
        """
        # 获取查询
        query = context.get("query", "")
        
        # 模拟生成SQL查询（实际应用中应使用更复杂的SQL生成技术）
        if "数量" in query or "统计" in query:
            sql = "SELECT COUNT(*) FROM documents WHERE category = 'tech'"
            data = [{"count": random.randint(100, 1000)}]
        else:
            sql = "SELECT id, title FROM documents LIMIT 5"
            data = [{"id": f"data_{i}", "title": f"文档标题 {i}"} for i in range(5)]
        
        return {
            "data": data,
            "sql": sql,
            "route": "sql",
            "execution_time": round(random.uniform(5, 20), 2)
        }

class CypherNode(QueryNode):
    """Cypher节点，实现图数据查询"""
    def __init__(self, node_id: str = "cypher_1"):
        super().__init__(node_id, "cypher")
    
    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行Cypher查询
        
        Args:
            context: 执行上下文
        
        Returns:
            查询结果，包含图数据和Cypher语句
        """
        # 获取查询
        query = context.get("query", "")
        
        # 模拟生成Cypher查询
        if "关系" in query or "关联" in query:
            cypher = "MATCH (a)-[:RELATED_TO]->(b) RETURN a, b LIMIT 3"
            relationships = [
                {"source": "entity_1", "target": "entity_2", "type": "RELATED_TO"},
                {"source": "entity_3", "target": "entity_4", "type": "SIMILAR_TO"},
                {"source": "entity_5", "target": "entity_6", "type": "HAS_PART"}
            ]
        else:
            cypher = "MATCH (n) WHERE n.type = 'concept' RETURN n LIMIT 5"
            nodes = [{"id": f"node_{i}", "type": "concept", "name": f"概念 {i}"} for i in range(5)]
            relationships = []
        
        return {
            "nodes": nodes if 'nodes' in locals() else [],
            "relationships": relationships,
            "cypher": cypher,
            "route": "cypher",
            "execution_time": round(random.uniform(8, 25), 2)
        }

class HybridRetrievalNode(QueryNode):
    """混合检索节点，结合多种检索方法"""
    def __init__(self, node_id: str = "hybrid_1"):
        super().__init__(node_id, "hybrid_retrieval")
    
    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行混合检索
        
        Args:
            context: 执行上下文
        
        Returns:
            检索结果，包含文档ID列表和混合策略信息
        """
        # 获取查询
        query = context.get("query", "")
        
        # 模拟混合检索结果
        docs = [f"doc_{i}" for i in random.sample(range(1, 100), 8)]
        
        # 模拟混合策略权重
        strategies = {
            "vector": round(random.uniform(0.3, 0.7), 2),
            "keyword": round(random.uniform(0.2, 0.5), 2),
            "semantic": round(random.uniform(0.1, 0.3), 2)
        }
        # 归一化权重
        total = sum(strategies.values())
        strategies = {k: round(v/total, 2) for k, v in strategies.items()}
        
        return {
            "docs": docs,
            "strategies": strategies,
            "route": "hybrid",
            "score": round(random.uniform(0.75, 0.95), 2)
        }

class MergeNode(QueryNode):
    """合并节点，汇总多个节点的执行结果"""
    def __init__(self, node_id: str = "merge_1"):
        super().__init__(node_id, "merge")
    
    def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个节点的结果
        
        Args:
            context: 执行上下文，包含各个节点的输出结果
        
        Returns:
            合并后的最终结果
        """
        # 收集所有前置节点的输出
        all_docs = set()
        all_data = []
        all_routes = set()
        
        # 遍历所有节点的输出
        for node_id, node in context.get("nodes", {}).items():
            if node.outputs:
                # 收集文档
                if "docs" in node.outputs:
                    all_docs.update(node.outputs["docs"])
                # 收集数据
                if "data" in node.outputs:
                    all_data.extend(node.outputs["data"])
                # 收集路由信息
                if "route" in node.outputs:
                    all_routes.add(node.outputs["route"])
        
        # 模拟去重和排序
        merged_docs = list(all_docs)[:5]  # 只保留前5个结果
        
        # 确定最终路由策略
        final_route = "hybrid" if len(all_routes) > 1 else list(all_routes)[0] if all_routes else "unknown"
        
        return {
            "docs": merged_docs,
            "route": final_route,
            "sources": list(all_routes),
            "total_docs": len(all_docs),
            "total_data": len(all_data)
        }

class QueryDAG:
    """查询DAG（有向无环图）类，用于管理和执行查询节点"""
    def __init__(self):
        """初始化查询DAG"""
        self.nodes = {}  # 节点字典: {node_id: QueryNode}
        self.edges = defaultdict(list)  # 边字典: {source_node_id: [target_node_id]}
        self.in_degree = defaultdict(int)  # 入度字典: {node_id: in_degree}
    
    def add_node(self, node: QueryNode):
        """添加节点到DAG
        
        Args:
            node: QueryNode实例
        """
        self.nodes[node.node_id] = node
        if node.node_id not in self.in_degree:
            self.in_degree[node.node_id] = 0
    
    def add_edge(self, source_id: str, target_id: str):
        """添加边到DAG（source_id -> target_id）
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
        """
        # 确保源节点和目标节点存在
        if source_id not in self.nodes:
            raise ValueError(f"源节点 {source_id} 不存在")
        if target_id not in self.nodes:
            raise ValueError(f"目标节点 {target_id} 不存在")
        
        # 添加边并更新入度
        self.edges[source_id].append(target_id)
        self.in_degree[target_id] += 1
    
    def topological_sort(self) -> List[str]:
        """执行拓扑排序，返回节点的执行顺序
        
        Returns:
            节点ID列表，按照执行顺序排列
        """
        # 初始化队列，将入度为0的节点加入队列
        queue = deque([node_id for node_id, degree in self.in_degree.items() if degree == 0])
        result = []
        
        # 执行拓扑排序
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            
            # 更新相邻节点的入度
            for neighbor_id in self.edges[node_id]:
                self.in_degree[neighbor_id] -= 1
                # 如果入度变为0，则加入队列
                if self.in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)
        
        # 检查是否存在环
        if len(result) != len(self.nodes):
            raise ValueError("DAG中存在环，无法进行拓扑排序")
        
        return result
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行DAG中的所有节点
        
        Args:
            context: 执行上下文
        
        Returns:
            包含执行计划和最终结果的字典
        """
        # 执行拓扑排序，获取执行顺序
        execution_plan = self.topological_sort()
        
        # 保存执行计划到上下文
        context["execution_plan"] = execution_plan
        context["nodes"] = self.nodes
        
        print(f"执行计划: {[self.nodes[node_id].node_type for node_id in execution_plan]}")
        
        # 按照执行顺序执行每个节点
        for node_id in execution_plan:
            node = self.nodes[node_id]
            
            # 收集输入数据
            node.inputs = []
            for source_id, targets in self.edges.items():
                if node_id in targets and source_id in self.nodes:
                    source_node = self.nodes[source_id]
                    if source_node.outputs is not None:
                        node.inputs.append(source_node.outputs)
            
            # 执行节点
            node.execute(context)
        
        # 查找最终结果（通常是最后一个节点的输出）
        final_result = None
        if execution_plan:
            last_node_id = execution_plan[-1]
            last_node = self.nodes[last_node_id]
            if last_node.outputs is not None:
                final_result = last_node.outputs
        
        return {
            "plan": [self.nodes[node_id].node_type for node_id in execution_plan],
            "result": final_result,
            "execution_time": sum(node.execution_time for node in self.nodes.values())
        }

class QueryAnalyzer:
    """查询分析器，用于分析查询意图并构建DAG"""
    def __init__(self):
        """初始化查询分析器"""
        # 定义意图模式
        self.intent_patterns = {
            "self_query": [r"查找.*文档", r"搜索.*内容", r"检索.*信息"],
            "sql": [r"有多少.*", r"统计.*数量", r"查询.*数据"],
            "cypher": [r"什么关系", r"哪些关联", r"相关的.*实体"],
            "hybrid_retrieval": [r"综合查询", r"混合检索", r"全面查找"],
        }
    
    def analyze_intent(self, query: str) -> List[str]:
        """分析查询意图，确定需要执行的节点类型
        
        Args:
            query: 查询字符串
        
        Returns:
            节点类型列表
        """
        # 转换为小写便于匹配
        query_lower = query.lower()
        
        # 识别意图
        detected_intents = []
        
        # 检查每种意图模式
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_intents.append(intent_type)
                    break
        
        # 如果没有检测到特定意图，使用默认意图
        if not detected_intents:
            # 对于大多数通用查询，使用self_query作为默认
            detected_intents.append("self_query")
        
        # 去重并返回
        return list(set(detected_intents))
    
    def build_dag(self, query: str) -> QueryDAG:
        """根据查询构建DAG
        
        Args:
            query: 查询字符串
        
        Returns:
            构建好的QueryDAG实例
        """
        # 分析意图
        intents = self.analyze_intent(query)
        
        print(f"检测到的意图: {intents}")
        
        # 创建DAG
        dag = QueryDAG()
        
        # 创建节点
        node_instances = {}
        
        # 根据意图创建节点
        for intent in intents:
            if intent == "self_query":
                node = SelfQueryNode()
            elif intent == "sql":
                node = SQLNode()
            elif intent == "cypher":
                node = CypherNode()
            elif intent == "hybrid_retrieval":
                node = HybridRetrievalNode()
            else:
                continue
            
            node_instances[intent] = node
            dag.add_node(node)
        
        # 如果有多个意图节点，添加merge节点
        if len(intents) > 1:
            merge_node = MergeNode()
            dag.add_node(merge_node)
            
            # 连接所有意图节点到merge节点
            for intent_node in node_instances.values():
                dag.add_edge(intent_node.node_id, merge_node.node_id)
        
        # 特殊情况处理：如果只有hybrid_retrieval意图，可以不需要merge节点
        elif len(intents) == 1 and intents[0] == "hybrid_retrieval":
            pass  # hybrid_retrieval节点自身就是最终节点
        
        # 如果只有单个意图节点，确保它可以作为最终节点
        elif len(intents) == 1:
            # 对于某些意图（如self_query），可以添加一个简单的merge节点来统一输出格式
            merge_node = MergeNode()
            dag.add_node(merge_node)
            
            # 连接意图节点到merge节点
            for intent_node in node_instances.values():
                dag.add_edge(intent_node.node_id, merge_node.node_id)
        
        return dag

# 主函数
def main():
    """主函数，执行查询DAG演示"""
    print("===== 查询DAG最小演示程序 ======")
    
    # 检查依赖
    if not check_dependencies():
        print("依赖检查失败，程序退出。")
        sys.exit(1)
    
    # 准备测试查询
    test_queries = [
        "查找关于人工智能的文档",
        "有多少篇关于机器学习的文章",
        "这些概念之间有什么关系",
        "综合查询技术发展趋势相关资料"
    ]
    
    # 选择一个测试查询（这里选择第一个）
    selected_query = test_queries[0]
    print(f"\n使用测试查询: {selected_query}")
    
    # 创建查询分析器
    analyzer = QueryAnalyzer()
    
    # 构建DAG
    dag = analyzer.build_dag(selected_query)
    
    # 准备执行上下文
    context = {
        "query": selected_query,
        "timestamp": time.time(),
        "user_id": "test_user"
    }
    
    # 执行DAG
    start_time = time.time()
    result = dag.execute(context)
    end_time = time.time()
    
    # 计算总执行时间
    total_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    # 显示执行计划和结果
    print("\n=== 执行结果 ===")
    print(f"plan={result['plan']}")
    print(f"result={result['result']}")
    print(f"总执行时间: {total_time:.2f}毫秒")
    
    # 检查执行效率
    if total_time > 50:
        print("警告: 执行效率较低，可能需要进一步优化。")
    else:
        print("执行效率良好，满足实时处理需求。")
    
    # 检查输出是否达到演示目的
    if result['plan'] and result['result']:
        print("\n演示目的已达成: 成功构建并执行了查询DAG，生成了执行计划和最终结果。")
    else:
        print("\n注意: 输出不完整，可能需要调整DAG构建逻辑。")
    
    # 保存结果到文件
    save_results(result, selected_query)
    
    print("\n===== 程序执行完成 =====")

def save_results(result: Dict[str, Any], query: str):
    """保存执行结果到文件
    
    Args:
        result: 执行结果
        query: 查询字符串
    """
    output_data = {
        "query": query,
        "plan": result["plan"],
        "result": result["result"],
        "execution_time_ms": result["execution_time"]
    }
    
    output_file = "08-query_dag_results.json"
    try:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n执行结果已保存到: {output_file}")
    except Exception as e:
        print(f"保存执行结果失败: {e}")

# 程序入口
if __name__ == "__main__":
    main()