#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自然语言到Cypher查询转换与安全校验工具

功能说明：
此程序将自然语言查询（特别是与社交关系相关的查询，如"同部门1–2度朋友"）映射为Cypher查询语句，并进行安全检查。
程序会使用预定义的白名单对标签、关系和属性进行校验，并过滤删除、写入等危险操作指令，确保生成的Cypher查询安全可靠。

执行流程：
1. 初始化白名单与危险操作正则表达式
2. 从自然语言中提取用户ID、关系深度与查询条件
3. 生成符合语法规范的Cypher查询
4. 执行白名单校验与危险指令检测
5. 输出生成的Cypher查询或错误信息

作者：Ph.D. Rhino
"""

import re
import sys
import subprocess
import importlib.util


def check_and_install_dependencies():
    """
    检查并安装必要的依赖包
    本程序主要使用Python标准库，不依赖外部包
    但为了演示依赖检查功能，我们仍保留此函数
    """
    required_packages = []
    missing_packages = []
    
    # 检查所需的包是否已安装
    for package in required_packages:
        try:
            # 使用importlib代替废弃的pkg_resources
            importlib.util.find_spec(package)
        except ModuleNotFoundError:
            missing_packages.append(package)
    
    # 安装缺失的包
    if missing_packages:
        print(f"正在安装缺失的依赖包: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing_packages])
            print("依赖包安装成功!")
        except subprocess.CalledProcessError as e:
            print(f"依赖包安装失败: {e}")
            sys.exit(1)
    else:
        print("所有依赖包已安装完成。")


class TextToCypherConverter:
    """自然语言到Cypher查询转换与安全校验的主类"""
    
    def __init__(self):
        # 定义Neo4j图数据库的白名单（允许使用的标签、关系和属性）
        self.whitelist = {
            'labels': {'User', 'Department', 'Company', 'Team', 'Project'},
            'relations': {'FRIEND', 'COLLEAGUE', 'MANAGES', 'WORKS_IN', 'PARTICIPATES_IN'},
            'attributes': {'id', 'name', 'department', 'email', 'position', 'hire_date'}
        }
        
        # 定义危险操作的正则表达式模式
        self.dangerous_patterns = [
            # 匹配写操作
            r'\b(?:CREATE|MERGE|SET|DELETE|REMOVE)\b',
            # 匹配修改操作
            r'\b(?:UPDATE|INSERT|DROP|ALTER)\b',
            # 匹配删除操作
            r'\b(?:删除|移除|清空|销毁|修改|更新|插入|添加)\b',
            # 匹配特殊字符和注入尝试
            r';\s*[^\s]',  # 检测多个语句
            r'\b(?:OR|AND)\b\s*\d+\s*=',  # 检测可能的布尔盲注
        ]
        
        # 部门名称映射（自然语言到系统中的部门名称）
        self.department_mapping = {
            '销售': 'sales',
            '市场': 'marketing',
            '研发': 'rd',
            '技术': 'tech',
            '产品': 'product',
            '人事': 'hr',
            '财务': 'finance',
            '运营': 'operations',
            '客服': 'customer_service'
        }
    
    def extract_query_params(self, natural_language):
        """
        从自然语言中提取查询参数
        参数: natural_language - 自然语言查询
        返回: dict - 包含用户ID、关系深度、部门等查询参数的字典
        """
        params = {
            'user_id': None,
            'depth_min': 1,
            'depth_max': 2,
            'department': None,
            'relation_type': 'FRIEND'
        }
        
        # 提取用户ID
        user_id_match = re.search(r'用户(\d+)', natural_language)
        if user_id_match:
            params['user_id'] = user_id_match.group(1)
        
        # 提取关系深度
        depth_match = re.search(r'(\d+)到(\d+)度', natural_language) or \
                      re.search(r'(\d+)–(\d+)度', natural_language) or \
                      re.search(r'(\d+)—(\d+)度', natural_language)  # 处理不同的连字符
        if depth_match:
            params['depth_min'] = int(depth_match.group(1))
            params['depth_max'] = int(depth_match.group(2))
        else:
            # 检查是否是固定深度
            single_depth_match = re.search(r'(\d+)度', natural_language)
            if single_depth_match:
                depth = int(single_depth_match.group(1))
                params['depth_min'] = depth
                params['depth_max'] = depth
        
        # 提取部门信息
        for chinese_dept, english_dept in self.department_mapping.items():
            if chinese_dept in natural_language:
                params['department'] = [english_dept]
                break
        
        # 如果没有明确指定部门，但提到了"同部门"，则默认使用用户所在部门
        if '同部门' in natural_language and params['department'] is None:
            params['department'] = ['sales']  # 默认销售部门
        
        # 提取关系类型（朋友、同事等）
        if '同事' in natural_language:
            params['relation_type'] = 'COLLEAGUE'
        
        return params
    
    def is_safe_cypher(self, cypher):
        """
        检查Cypher查询是否安全（不包含危险操作）
        参数: cypher - 要检查的Cypher查询
        返回: (布尔值, 错误信息) - 如果安全返回(True, None)，否则返回(False, 错误信息)
        """
        # 将Cypher转为小写进行检查
        cypher_lower = cypher.lower()
        
        # 检查是否包含危险模式
        for pattern in self.dangerous_patterns:
            if re.search(pattern, cypher_lower):
                return False, f"Cypher包含危险操作: 匹配模式 '{pattern}'"
        
        # 检查是否只包含只读操作
        if not (cypher_lower.startswith('match') or cypher_lower.startswith('explain') or cypher_lower.startswith('profile')):
            return False, "只允许MATCH、EXPLAIN或PROFILE查询操作"
        
        return True, None
    
    def validate_whitelist(self, cypher):
        """
        验证Cypher查询是否符合预定义的白名单
        参数: cypher - 要验证的Cypher查询
        返回: (布尔值, 错误信息) - 如果符合白名单返回(True, None)，否则返回(False, 错误信息)
        """
        # 检查标签
        labels = re.findall(r'\b([A-Za-z0-9_]+):([A-Za-z0-9_]+)', cypher)
        for var, label in labels:
            if label not in self.whitelist['labels']:
                return False, f"不允许的标签: {label}"
        
        # 检查关系
        relations = re.findall(r'\[:([A-Za-z0-9_]+)', cypher)
        for rel in relations:
            # 处理可变长度关系，如FRIEND*1..2
            rel_base = rel.split('*')[0]
            if rel_base not in self.whitelist['relations']:
                return False, f"不允许的关系: {rel_base}"
        
        # 检查属性
        properties = re.findall(r'\b([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)', cypher)
        for var, prop in properties:
            if prop not in self.whitelist['attributes']:
                return False, f"不允许的属性: {prop}"
        
        return True, None
    
    def convert_to_cypher(self, natural_language):
        """
        将自然语言转换为Cypher查询
        参数: natural_language - 自然语言查询
        返回: 生成的Cypher查询
        """
        # 提取查询参数
        params = self.extract_query_params(natural_language)
        
        # 默认参数处理
        user_id = params['user_id'] or '123'  # 默认用户ID
        depth_min = params['depth_min']
        depth_max = params['depth_max']
        department = params['department'] or ['sales']  # 默认销售部门
        relation_type = params['relation_type']
        
        # 构建Cypher查询
        # 处理关系深度
        depth_clause = ''
        if depth_min == depth_max:
            depth_clause = f'*{depth_min}'
        else:
            depth_clause = f'*{depth_min}..{depth_max}'
        
        # 构建部门条件
        dept_clause = ''
        if department:
            dept_str = ', '.join([f"'{d}'" for d in department])
            dept_clause = f" AND v.department IN [{dept_str}]"
        
        # 构建完整的Cypher查询
        cypher = f"MATCH (u:User)-[:{relation_type}{depth_clause}]-(v:User) WHERE u.id='{user_id}'{dept_clause} RETURN DISTINCT v.id;"
        
        return cypher
    
    def process_query(self, natural_language):
        """
        处理自然语言查询，生成Cypher并进行安全检查
        参数: natural_language - 自然语言查询
        返回: (生成的Cypher, 是否安全, 错误信息)
        """
        # 前置检查：检查是否包含危险操作的意图
        dangerous_intents = {'删除', '修改', '插入', '更新', '创建', '清空', '移除', '销毁'}
        for intent in dangerous_intents:
            if intent in natural_language:
                return "", False, f"不支持的操作: {intent}\n此工具仅支持只读查询操作"
        
        # 转换为Cypher
        cypher = self.convert_to_cypher(natural_language)
        
        # 进行安全检查
        is_safe, safe_error = self.is_safe_cypher(cypher)
        if not is_safe:
            return cypher, False, safe_error
        
        # 进行白名单验证
        is_valid_whitelist, whitelist_error = self.validate_whitelist(cypher)
        if not is_valid_whitelist:
            return cypher, False, whitelist_error
        
        return cypher, True, None


def main():
    """主函数，处理命令行参数并执行转换与校验"""
    # 检查并安装依赖
    check_and_install_dependencies()
    
    # 创建转换器实例
    converter = TextToCypherConverter()
    
    # 从命令行参数获取自然语言查询，如果没有则使用示例
    if len(sys.argv) > 1:
        natural_language = ' '.join(sys.argv[1:])
    else:
        # 使用内置示例
        natural_language = "查找用户123在同部门的1到2度朋友"
    
    print(f"NL: {natural_language}")
    
    # 处理查询
    cypher, is_safe, error_msg = converter.process_query(natural_language)
    
    print(f"CYPHER: {cypher}")
    
    # 打印检查结果
    if is_safe:
        print("check: OK")
    else:
        print(f"check: ERROR - {error_msg}")


if __name__ == "__main__":
    main()