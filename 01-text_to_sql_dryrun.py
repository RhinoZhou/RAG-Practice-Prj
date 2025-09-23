#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本到SQL转换与安全校验工具

功能说明：
此程序基于简单模板将自然语言查询映射为SQL语句，并进行schema校验与"干跑"安全检查。
程序会内置数据库表结构信息，通过规则从自然语言中提取列名与过滤条件，拼接成SELECT语句，
然后进行列存在性验证、保留字检测和危险动词检测，最终输出合法的SQL语句或报错信息。

执行流程：
1. 定义数据库schema与允许的表和列
2. 通过规则抽取用户意图与过滤条件
3. 生成SQL查询字符串
4. 执行Dry-run安全校验（禁止DDL/DML操作）
5. 打印生成的SQL或错误信息

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


class TextToSQLConverter:
    """\自然语言到SQL转换与安全校验的主类"""
    
    def __init__(self):
        # 定义数据库schema信息
        self.schema = {
            'sales': {
                'columns': ['order_id', 'customer_id', 'product_id', 'amount', 'quantity', 'order_date', 'status'],
                'description': '销售订单表'
            },
            'products': {
                'columns': ['product_id', 'product_name', 'category', 'price', 'stock_quantity', 'description'],
                'description': '产品信息表'
            },
            'customers': {
                'columns': ['customer_id', 'customer_name', 'email', 'phone', 'address', 'registration_date'],
                'description': '客户信息表'
            }
        }
        
        # 定义危险的SQL操作（DDL/DML）
        self.dangerous_verbs = {
            'drop', 'delete', 'truncate', 'alter', 'update', 'insert', 'create', 
            'rename', 'grant', 'revoke', 'merge', 'execute', 'call'
        }
        
        # 定义自然语言到SQL关键字的映射规则
        self.nlp_rules = {
            # 数量统计相关
            '总金额': 'SUM(amount)',
            '总数': 'COUNT(*)',
            '平均金额': 'AVG(amount)',
            '最大金额': 'MAX(amount)',
            '最小金额': 'MIN(amount)',
            
            # 表名映射
            '订单': 'sales',
            '产品': 'products',
            '客户': 'customers',
            
            # 列名映射
            '产品名称': 'product_name',
            '订单ID': 'order_id',
            '客户姓名': 'customer_name',
            '价格': 'price',
            '数量': 'quantity',
            
            # 条件映射
            '已支付': "status='paid'",
            '已完成': "status='completed'",
            '未支付': "status='unpaid'",
            '已取消': "status='cancelled'",
        }
        
        # 定义聚合函数后的分组关键字
        self.group_by_keywords = {'每个', '各', '每一个'}
    
    def is_safe_sql(self, sql):
        """
        检查SQL语句是否安全（不包含危险操作）
        参数: sql - 要检查的SQL语句
        返回: (布尔值, 错误信息) - 如果安全返回(True, None)，否则返回(False, 错误信息)
        """
        # 将SQL转为小写进行检查
        sql_lower = sql.lower()
        
        # 检查是否包含危险动词
        for verb in self.dangerous_verbs:
            if re.search(rf'\b{verb}\b', sql_lower):
                return False, f"SQL包含危险操作: {verb}"
        
        # 检查是否以SELECT开头（仅允许SELECT查询）
        if not sql_lower.strip().startswith('select'):
            return False, "只允许SELECT查询操作"
        
        # 检查是否包含注释（可能隐藏危险操作）
        if '/*' in sql_lower or '--' in sql_lower:
            return False, "SQL包含注释，可能存在安全风险"
        
        return True, None
    
    def validate_schema(self, sql):
        """
        验证SQL语句是否符合预定义的schema
        参数: sql - 要验证的SQL语句
        返回: (布尔值, 错误信息) - 如果符合schema返回(True, None)，否则返回(False, 错误信息)
        """
        # 提取FROM子句中的表名
        from_match = re.search(r'FROM\s+([a-zA-Z0-9_]+)', sql, re.IGNORECASE)
        if not from_match:
            return False, "SQL缺少FROM子句"
        
        table_name = from_match.group(1).lower()
        
        # 检查表是否存在于schema中
        if table_name not in self.schema:
            return False, f"表不存在: {table_name}"
        
        # 提取SELECT子句中的列名
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return False, "SQL缺少SELECT子句"
        
        columns_str = select_match.group(1)
        
        # 分割列名，处理可能的逗号分隔
        column_parts = [part.strip() for part in columns_str.split(',')]
        
        # 检查每一列
        for part in column_parts:
            # 处理聚合函数
            agg_match = re.match(r'(SUM|AVG|MAX|MIN|COUNT)\(\s*(\*|[a-zA-Z0-9_]+)\s*\)', part, re.IGNORECASE)
            if agg_match:
                # 聚合函数中的列
                agg_col = agg_match.group(2)
                # 如果不是通配符，则检查列是否存在
                if agg_col != '*' and agg_col not in self.schema[table_name]['columns']:
                    return False, f"列不存在: {agg_col} 在表 {table_name} 中不存在"
                continue
            
            # 处理AS别名，如 column_name AS alias
            as_match = re.match(r'([a-zA-Z0-9_\*]+)\s+AS\s+[a-zA-Z0-9_]+', part, re.IGNORECASE)
            if as_match:
                col_name = as_match.group(1)
            else:
                col_name = part
            
            # 检查列是否存在
            if col_name == '*':  # 允许通配符
                continue
            if col_name not in self.schema[table_name]['columns']:
                return False, f"列不存在: {col_name} 在表 {table_name} 中不存在"
        
        return True, None
    
    def convert_to_sql(self, natural_language):
        """
        将自然语言转换为SQL语句
        参数: natural_language - 自然语言查询
        返回: 生成的SQL语句
        """
        # 默认表名和列名
        table_name = 'sales'  # 默认销售表
        columns = []  # 将在后续逻辑中设置
        conditions = []
        group_by_clause = ''
        
        # 检查是否包含特定的表名
        for nl_table, sql_table in self.nlp_rules.items():
            if nl_table in natural_language and sql_table in self.schema:
                table_name = sql_table
                break
        
        # 初始化列列表
        # 检查是否包含聚合函数和普通列名的组合
        detected_columns = []
        
        # 检查聚合函数
        agg_mappings = {
            '总金额': 'SUM(amount)',
            '总数': 'COUNT(*)',
            '平均金额': 'AVG(amount)',
            '最大金额': 'MAX(amount)',
            '最小金额': 'MIN(amount)'
        }
        for nl_agg, sql_agg in agg_mappings.items():
            if nl_agg in natural_language:
                detected_columns.append(sql_agg)
        
        # 检查特定列名
        column_mappings = {
            '产品名称': 'product_name',
            '订单ID': 'order_id',
            '客户姓名': 'customer_name',
            '价格': 'price',
            '数量': 'quantity'
        }
        for nl_col, sql_col in column_mappings.items():
            if nl_col in natural_language:
                detected_columns.append(sql_col)
        
        # 特殊处理'产品表中的产品名称'这样的结构
        if '产品表中的产品名称' in natural_language:
            detected_columns.append('product_name')
        
        # 设置最终列列表
        if detected_columns:
            columns = detected_columns
        elif '所有' in natural_language:
            columns = ['*']
        else:
            columns = ['*']  # 默认选择所有列
        
        # 检查是否需要分组（当有聚合函数时）
        has_agg_function = any(agg in ','.join(columns) for agg in ['SUM', 'AVG', 'MAX', 'MIN', 'COUNT'])
        if has_agg_function:
            for keyword in self.group_by_keywords:
                if keyword in natural_language:
                    if table_name == 'sales':
                        group_by_clause = ' GROUP BY order_id'
                    elif table_name == 'products':
                        group_by_clause = ' GROUP BY product_id'
                    elif table_name == 'customers':
                        group_by_clause = ' GROUP BY customer_id'
                    break
        
        # 添加过滤条件
        for nl_cond, sql_cond in self.nlp_rules.items():
            if nl_cond in natural_language and sql_cond.startswith('status='):
                conditions.append(sql_cond)
        
        # 构建WHERE子句
        where_clause = ''
        if conditions:
            where_clause = ' WHERE ' + ' AND '.join(conditions)
        
        # 构建完整的SQL语句
        columns_str = ', '.join(columns)
        sql = f'SELECT {columns_str} FROM {table_name}{where_clause}{group_by_clause};'
        
        return sql
    
    def process_query(self, natural_language):
        """
        处理自然语言查询，生成SQL并进行安全检查
        参数: natural_language - 自然语言查询
        返回: (生成的SQL, 是否安全, 错误信息)
        """
        # 前置检查：检查是否包含危险操作的意图
        dangerous_intents = {'删除', '修改', '插入', '更新', '创建', '删除表', '清空'}
        for intent in dangerous_intents:
            if intent in natural_language:
                return "", False, f"不支持的操作: {intent}\n此工具仅支持SELECT查询操作"
        
        # 转换为SQL
        sql = self.convert_to_sql(natural_language)
        
        # 进行安全检查
        is_safe, safe_error = self.is_safe_sql(sql)
        if not is_safe:
            return sql, False, safe_error
        
        # 进行schema验证
        is_valid_schema, schema_error = self.validate_schema(sql)
        if not is_valid_schema:
            return sql, False, schema_error
        
        return sql, True, None


def main():
    """主函数，处理命令行参数并执行转换与校验"""
    # 检查并安装依赖
    check_and_install_dependencies()
    
    # 创建转换器实例
    converter = TextToSQLConverter()
    
    # 从命令行参数获取自然语言查询，如果没有则使用示例
    if len(sys.argv) > 1:
        natural_language = ' '.join(sys.argv[1:])
    else:
        # 使用内置示例
        natural_language = "查询已支付订单的总金额"
    
    print(f"NL: {natural_language}")
    
    # 处理查询
    sql, is_safe, error_msg = converter.process_query(natural_language)
    
    print(f"SQL: {sql}")
    
    # 打印检查结果
    if is_safe:
        print("check: OK (dry-run passed)")
    else:
        print(f"check: ERROR - {error_msg}")


if __name__ == "__main__":
    main()