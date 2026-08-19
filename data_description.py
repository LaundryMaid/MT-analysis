"""数据集描述与统计分析：读取dev.csv，生成星级分布和评论长度图表"""
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def clean_text(text):
    """去除特殊字符，保留中英文、数字、常用标点"""
    if pd.isna(text):
        return ''
    # 只保留中文、中英文标点、字母数字
    return re.sub(r'[^\u4e00-\u9fa5，。！？；：、a-zA-Z0-9\.,!?;:\s]', '', str(text))

def main():
    os.makedirs('visualization', exist_ok=True)
    
    try:
        df = pd.read_csv('data/dev.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        print("错误: 未找到 dev.csv 文件")
        return
    
    print(f"总评论数: {len(df)}")
    
    # 统计星级分布
    star_counts = df['star'].value_counts().sort_index()
    print("\n星级评分分布:")
    for star, count in star_counts.items():
        print(f"  {star}星: {count}条 ({count/len(df)*100:.1f}%)")
    
    # 评论长度统计
    df['raw_len'] = df['review'].astype(str).apply(len)
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['clean_len'] = df['cleaned_review'].apply(len)
    
    print(f"\n原始评论长度: 均{df['raw_len'].mean():.1f} 中{df['raw_len'].median():.0f}")
    print(f"清洗后长度:  均{df['clean_len'].mean():.1f} 中{df['clean_len'].median():.0f}")
    
    # 图1: 星级分布柱状图
    plt.figure(figsize=(8, 5))
    plt.bar(star_counts.index, star_counts.values, color='steelblue')
    plt.title('各星级评分数量分布')
    plt.xlabel('评论星级')
    plt.ylabel('评论数量 (条)')
    for i, v in enumerate(star_counts.values):
        plt.text(star_counts.index[i], v + 5, str(v), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('visualization/star_distribution.png', dpi=300)
    plt.close()
    
    # 图2: 清洗后长度区间分布（每24字符一档）
    bins = list(range(0, 1024, 24))
    labels = [f"({bins[i]}-{bins[i+1]}]" for i in range(len(bins)-1)]
    df['len_group'] = pd.cut(df['clean_len'], bins=bins, right=True, labels=labels, include_lowest=True)
    group_counts = df['len_group'].value_counts().sort_index()
    group_counts = group_counts[group_counts > 0]
    
    plt.figure(figsize=(14, 6))
    plt.bar(group_counts.index, group_counts.values, color='skyblue', edgecolor='black')
    plt.title('去除特殊字符后评论长度区间分布')
    plt.xlabel('评论字数')
    plt.ylabel('评论数量')
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(group_counts.values):
        plt.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('visualization/cleaned_length_interval.png', dpi=300)
    plt.close()
    
    print("\n图表已保存至 visualization/ 文件夹")

if __name__ == '__main__':
    main()