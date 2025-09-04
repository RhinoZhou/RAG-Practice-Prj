#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗方法演示

本脚本演示数据准备阶段中的关键数据清洗技术：
1. 缺失值填补（KNN填补法）
2. 异常值检测（Z-score、IQR、IsolationForest方法）
3. 单位统一（不同单位间的换算策略对比）

这些技术是确保检索/生成阶段数据质量的重要步骤。
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 设置Matplotlib中文字体，解决中文显示问题 - 使用Windows系统确保可用的基本字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 配置路径
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

class DataCleaningMethods:
    """
    数据清洗方法类，提供各种数据清洗技术的实现
    """
    
    def __init__(self):
        """初始化数据清洗方法类"""
        # 设置随机种子以确保结果可复现
        np.random.seed(42)
        
        # 创建示例数据集
        self.sample_data = self._create_sample_data()
    
    def _create_sample_data(self):
        """
        创建包含缺失值、异常值和不同单位的示例数据集
        
        Returns:
            pd.DataFrame: 包含各种数据质量问题的示例数据集
        """
        # 创建一个基础数据集
        n_samples = 200
        
        # 正常数据分布
        age = np.random.normal(40, 10, n_samples).astype(int)
        weight_kg = np.random.normal(70, 10, n_samples)
        height_cm = np.random.normal(170, 10, n_samples)
        blood_pressure = np.random.normal(120, 10, n_samples)
        
        # 创建不同单位的体重数据
        weight_lbs = weight_kg * 2.20462  # kg to lbs
        
        # 创建数据集
        df = pd.DataFrame({
            'age': age,
            'weight_kg': weight_kg,
            'weight_lbs': weight_lbs,
            'height_cm': height_cm,
            'blood_pressure': blood_pressure
        })
        
        # 引入缺失值 (5% 的数据)
        for col in df.columns:
            mask = np.random.rand(n_samples) < 0.05
            df.loc[mask, col] = np.nan
        
        # 引入异常值
        # 年龄异常值
        df.loc[np.random.choice(n_samples, 5), 'age'] = np.random.choice([120, 130, 140, 150], 5)
        # 体重异常值
        df.loc[np.random.choice(n_samples, 5), 'weight_kg'] = np.random.normal(150, 20, 5)
        # 血压异常值
        df.loc[np.random.choice(n_samples, 5), 'blood_pressure'] = np.random.normal(200, 20, 5)
        
        return df
    
    def handle_missing_values(self):
        """
        演示缺失值填补方法，重点使用KNN填补
        
        Returns:
            dict: 包含原始数据和填补后数据的字典
        """
        print("\n=== 缺失值填补演示 ===")
        
        # 复制数据以避免修改原始数据
        data = self.sample_data.copy()
        
        # 显示原始数据中的缺失值
        missing_count = data.isnull().sum()
        print(f"原始数据中的缺失值数量:\n{missing_count}")
        
        # 使用KNN填补缺失值
        print("\n使用KNN填补缺失值...")
        imputer = KNNImputer(n_neighbors=5, weights='uniform')
        
        # 选择数值列进行填补
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data_knn_imputed = data.copy()
        
        # 对数值列进行KNN填补
        data_knn_imputed[numeric_cols] = imputer.fit_transform(data[numeric_cols])
        
        # 显示填补后的缺失值数量
        missing_count_after = data_knn_imputed.isnull().sum()
        print(f"KNN填补后的缺失值数量:\n{missing_count_after}")
        
        # 使用均值填补（作为对比）
        print("\n使用均值填补缺失值...")
        data_mean_imputed = data.copy()
        
        # 对数值列进行均值填补
        for col in numeric_cols:
            if data_mean_imputed[col].isnull().sum() > 0:
                mean_value = data_mean_imputed[col].mean()
                data_mean_imputed[col].fillna(mean_value, inplace=True)
        
        # 显示填补后的缺失值数量
        missing_count_mean_after = data_mean_imputed.isnull().sum()
        print(f"均值填补后的缺失值数量:\n{missing_count_mean_after}")
        
        return {
            'original': data,
            'knn_imputed': data_knn_imputed,
            'mean_imputed': data_mean_imputed
        }
    
    def detect_outliers(self):
        """
        演示异常值检测方法：Z-score、IQR和IsolationForest
        
        Returns:
            dict: 包含检测结果的字典
        """
        print("\n=== 异常值检测演示 ===")
        
        # 复制数据并去除缺失值以便检测异常值
        data = self.sample_data.copy().dropna()
        
        results = {}
        
        # 1. Z-score方法
        print("\n1. 使用Z-score方法检测异常值...")
        z_scores = stats.zscore(data['weight_kg'])
        abs_z_scores = np.abs(z_scores)
        outliers_zscore = abs_z_scores > 3  # 通常使用3作为阈值
        
        n_outliers_zscore = outliers_zscore.sum()
        print(f"Z-score方法检测到的异常值数量: {n_outliers_zscore}")
        print(f"异常值示例:\n{data[outliers_zscore].head()}")
        
        results['zscore'] = {
            'outliers': data[outliers_zscore],
            'count': n_outliers_zscore
        }
        
        # 2. IQR方法
        print("\n2. 使用IQR方法检测异常值...")
        Q1 = data['weight_kg'].quantile(0.25)
        Q3 = data['weight_kg'].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers_iqr = (data['weight_kg'] < (Q1 - 1.5 * IQR)) | (data['weight_kg'] > (Q3 + 1.5 * IQR))
        n_outliers_iqr = outliers_iqr.sum()
        
        print(f"IQR方法检测到的异常值数量: {n_outliers_iqr}")
        print(f"异常值示例:\n{data[outliers_iqr].head()}")
        
        results['iqr'] = {
            'outliers': data[outliers_iqr],
            'count': n_outliers_iqr
        }
        
        # 3. IsolationForest方法
        print("\n3. 使用IsolationForest方法检测异常值...")
        # 选择用于检测异常值的特征
        features = ['age', 'weight_kg', 'height_cm', 'blood_pressure']
        X = data[features]
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练IsolationForest模型
        clf = IsolationForest(contamination=0.05, random_state=42)
        outliers_iforest = clf.fit_predict(X_scaled) == -1
        
        n_outliers_iforest = outliers_iforest.sum()
        print(f"IsolationForest方法检测到的异常值数量: {n_outliers_iforest}")
        print(f"异常值示例:\n{data[outliers_iforest].head()}")
        
        results['isolation_forest'] = {
            'outliers': data[outliers_iforest],
            'count': n_outliers_iforest
        }
        
        return results
    
    def unify_units(self):
        """
        演示单位统一方法，比较不同单位换算策略
        
        Returns:
            dict: 包含单位统一前后数据的字典
        """
        print("\n=== 单位统一演示 ===")
        
        # 复制数据
        data = self.sample_data.copy()
        
        print("\n当前数据中的单位情况:")
        print(f"- weight_kg: 公斤")
        print(f"- weight_lbs: 磅")
        print(f"- height_cm: 厘米")
        
        # 创建一个新的DataFrame用于存储单位统一后的数据
        data_unified = data.copy()
        
        # 1. 将磅转换为公斤
        print("\n1. 将磅(lbs)转换为公斤(kg)...")
        # lbs到kg的转换因子
        lbs_to_kg = 0.453592
        data_unified['weight_kg_converted'] = data_unified['weight_lbs'] * lbs_to_kg
        
        # 2. 将厘米转换为米
        print("2. 将厘米(cm)转换为米(m)...")
        # cm到m的转换因子
        cm_to_m = 0.01
        data_unified['height_m'] = data_unified['height_cm'] * cm_to_m
        
        # 3. 计算BMI（体重指数）
        print("3. 计算BMI（体重指数）...")
        # BMI = 体重(kg) / 身高(m)^2
        data_unified['bmi'] = data_unified['weight_kg'] / (data_unified['height_m'] ** 2)
        
        # 显示转换后的结果
        print("\n单位统一后的结果示例:")
        print(data_unified[['weight_kg', 'weight_lbs', 'weight_kg_converted', 'height_cm', 'height_m', 'bmi']].head())
        
        # 比较原始公斤数据和从磅转换的公斤数据
        print("\n比较原始公斤数据和从磅转换的公斤数据:")
        # 只比较非缺失值
        mask = ~data_unified['weight_kg'].isnull() & ~data_unified['weight_kg_converted'].isnull()
        correlation = data_unified.loc[mask, 'weight_kg'].corr(data_unified.loc[mask, 'weight_kg_converted'])
        print(f"相关性系数: {correlation:.4f}")
        
        # 计算平均绝对误差
        mae = np.mean(np.abs(data_unified.loc[mask, 'weight_kg'] - data_unified.loc[mask, 'weight_kg_converted']))
        print(f"平均绝对误差: {mae:.4f} kg")
        
        return {
            'original': data,
            'unified': data_unified
        }
    
    def visualize_results(self, missing_results, outlier_results, unit_results):
        """
        可视化数据清洗结果
        
        Args:
            missing_results (dict): 缺失值处理结果
            outlier_results (dict): 异常值检测结果
            unit_results (dict): 单位统一结果
        """
        print("\n=== 生成可视化结果 ===")
        
        # 创建图形目录
        fig_dir = os.path.join(OUTPUT_DIR, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # 1. 可视化缺失值填补效果
        plt.figure(figsize=(15, 5))
        
        # 原始数据分布
        plt.subplot(131)
        plt.hist(missing_results['original']['weight_kg'].dropna(), bins=20, alpha=0.7, color='blue')
        plt.title('原始数据 - 体重分布')
        plt.xlabel('体重 (kg)')
        plt.ylabel('频率')
        
        # KNN填补后的数据分布
        plt.subplot(132)
        plt.hist(missing_results['knn_imputed']['weight_kg'], bins=20, alpha=0.7, color='green')
        plt.title('KNN填补后 - 体重分布')
        plt.xlabel('体重 (kg)')
        plt.ylabel('频率')
        
        # 均值填补后的数据分布
        plt.subplot(133)
        plt.hist(missing_results['mean_imputed']['weight_kg'], bins=20, alpha=0.7, color='orange')
        plt.title('均值填补后 - 体重分布')
        plt.xlabel('体重 (kg)')
        plt.ylabel('频率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'missing_value_imputation.png'))
        plt.close()
        
        # 2. 可视化异常值检测结果
        data = outlier_results['zscore']['outliers'].copy()
        
        plt.figure(figsize=(12, 8))
        plt.scatter(data['height_cm'], data['weight_kg'], alpha=0.6, color='blue', label='正常数据')
        
        # 标记Z-score检测的异常值
        outliers_zscore = outlier_results['zscore']['outliers']
        plt.scatter(outliers_zscore['height_cm'], outliers_zscore['weight_kg'], color='red', label='Z-score异常值', s=100, marker='x')
        
        # 标记IQR检测的异常值
        outliers_iqr = outlier_results['iqr']['outliers']
        plt.scatter(outliers_iqr['height_cm'], outliers_iqr['weight_kg'], color='purple', label='IQR异常值', s=100, marker='o', facecolors='none')
        
        # 标记IsolationForest检测的异常值
        outliers_iforest = outlier_results['isolation_forest']['outliers']
        plt.scatter(outliers_iforest['height_cm'], outliers_iforest['weight_kg'], color='orange', label='IsolationForest异常值', s=100, marker='^')
        
        plt.title('不同方法检测的体重-身高异常值对比')
        plt.xlabel('身高 (cm)')
        plt.ylabel('体重 (kg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, 'outlier_detection.png'))
        plt.close()
        
        # 3. 可视化单位统一结果
        data = unit_results['unified'].dropna()
        
        plt.figure(figsize=(12, 6))
        
        # 体重单位转换对比
        plt.subplot(121)
        plt.scatter(data['weight_kg'], data['weight_kg_converted'], alpha=0.6)
        plt.plot([data['weight_kg'].min(), data['weight_kg'].max()], 
                 [data['weight_kg'].min(), data['weight_kg'].max()], 
                 'r--', label='理想转换线')
        plt.title('体重单位转换对比 (kg vs 从lbs转换的kg)')
        plt.xlabel('原始体重 (kg)')
        plt.ylabel('从磅转换的体重 (kg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # BMI分布
        plt.subplot(122)
        plt.hist(data['bmi'], bins=20, alpha=0.7, color='green')
        plt.title('BMI分布')
        plt.xlabel('BMI')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'unit_unification.png'))
        plt.close()
        
        print(f"可视化结果已保存到: {fig_dir}")
    
    def run_all_demonstrations(self):
        """
        运行所有数据清洗方法的演示
        """
        print("开始数据清洗方法演示...")
        
        # 1. 缺失值处理演示
        missing_results = self.handle_missing_values()
        
        # 2. 异常值检测演示
        outlier_results = self.detect_outliers()
        
        # 3. 单位统一演示
        unit_results = self.unify_units()
        
        # 4. 可视化结果
        self.visualize_results(missing_results, outlier_results, unit_results)
        
        # 5. 保存处理后的数据
        self._save_results(missing_results, outlier_results, unit_results)
        
        print("\n数据清洗方法演示完成！")
    
    def _save_results(self, missing_results, outlier_results, unit_results):
        """
        保存数据清洗的结果
        
        Args:
            missing_results (dict): 缺失值处理结果
            outlier_results (dict): 异常值检测结果
            unit_results (dict): 单位统一结果
        """
        # 保存缺失值填补结果
        missing_results['knn_imputed'].to_csv(os.path.join(OUTPUT_DIR, 'knn_imputed_data.csv'), index=False)
        missing_results['mean_imputed'].to_csv(os.path.join(OUTPUT_DIR, 'mean_imputed_data.csv'), index=False)
        
        # 保存异常值结果
        outlier_summary = pd.DataFrame({
            'Method': ['Z-score', 'IQR', 'IsolationForest'],
            'Outlier Count': [
                outlier_results['zscore']['count'],
                outlier_results['iqr']['count'],
                outlier_results['isolation_forest']['count']
            ]
        })
        outlier_summary.to_csv(os.path.join(OUTPUT_DIR, 'outlier_detection_summary.csv'), index=False)
        
        # 保存单位统一结果
        unit_results['unified'].to_csv(os.path.join(OUTPUT_DIR, 'unified_unit_data.csv'), index=False)
        
        print(f"\n处理结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    """主程序入口，运行数据清洗方法演示"""
    # 创建数据清洗方法实例
    cleaner = DataCleaningMethods()
    
    # 运行所有演示
    cleaner.run_all_demonstrations()