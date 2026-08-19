"""
信息挖掘与可视化：识别客户类型、同行情况、关注方面，生成统计图表
输出: 3张图表 (客户类型饼图、同行情况柱状图、关注方面柱状图)
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 新/老客户关键词
NEW_KEYWORDS = ['第一次', '初次', '首次', '头一回', '第一次来', '第一次吃']
OLD_KEYWORDS = ['又', '再', '经常', '常来', '老顾客', '回头', '常客', '多次']

# 同行情况关键词词典
COMPANION_DICT = {
    '独自': ['一人', '自己', '独自', '一个人', '只身', '单独'],
    '情侣': ['情侣', '约会', '对象', '女友', '男友', '爱人', '老婆', '老公'],
    '家人': ['家人', '爸妈', '父母', '孩子', '宝宝', '儿子', '女儿', '老爸', '老妈'],
    '同事': ['同事', '上班', '工作', '单位', '同僚', '团队', '搭档'],
    '朋友': ['朋友', '闺蜜', '兄弟', '死党', '哥们', '姐妹', '好友', '老友'],
    '同学': ['同学', '室友', '校友', '同窗', '学友'],
}

# 关注方面 -> ASAP标注列映射
ASPECT_COLUMNS = {
    '口味': ['Food#Taste', 'Food#Portion', 'Food#Appearance', 'Food#Recommend'],
    '环境': ['Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary'],
    '服务': ['Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely'],
    '价格': ['Price#Level', 'Price#Cost_effective', 'Price#Discount'],
    '位置': ['Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find']
}

def extract_customer_type(text):
    """识别客户类型（新/老/未知）"""
    if not text or pd.isna(text):
        return '未知'
    text = str(text)
    if any(k in text for k in NEW_KEYWORDS):
        return '新客户'
    if any(k in text for k in OLD_KEYWORDS):
        return '老客户'
    return '未知'

def extract_companion(text):
    """识别同行人类型（匹配数最多者优先）"""
    if not text or pd.isna(text):
        return '未知'
    text = str(text)
    type_scores = {}
    for comp, words in COMPANION_DICT.items():
        matched = sum(1 for word in words if word in text)
        if matched > 0:
            type_scores[comp] = matched
    if not type_scores:
        return '未知'
    max_score = max(type_scores.values())
    priority_order = ['独自', '情侣', '家人', '同事', '朋友', '同学']
    candidates = [comp for comp, score in type_scores.items() if score == max_score]
    for comp in priority_order:
        if comp in candidates:
            return comp
    return candidates[0]

def extract_aspects(row):
    """基于ASAP标注列判断评论关注的方面"""
    aspects = []
    for category, columns in ASPECT_COLUMNS.items():
        for col in columns:
            val = row.get(col, -2)
            if pd.notna(val) and int(val) != -2:
                aspects.append(category)
                break
    return aspects

def main():
    """读取数据 -> 信息挖掘 -> 统计可视化"""
    os.makedirs('visualization', exist_ok=True)
    
    try:
        df = pd.read_csv('data/dev.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        print("错误: 未找到 dev.csv 文件")
        return
    
    customer_types = []
    companions = []
    all_aspects = []
    total_valid = 0
    
    for idx, row in df.iterrows():
        review = row.get('review', '')
        if pd.isna(review) or not str(review).strip():
            continue
        total_valid += 1
        review = str(review).strip()
        
        ctype = extract_customer_type(review)
        comp = extract_companion(review)
        aspects = extract_aspects(row)
        
        customer_types.append(ctype)
        if comp != '未知':
            companions.append(comp)
        all_aspects.extend(aspects)
    
    ctype_counts = Counter(customer_types)
    comp_counts = Counter(companions)
    aspect_counts = Counter(all_aspects)
    
    # 打印统计结果
    print("=== 客户类型分布 ===")
    for k, v in ctype_counts.items():
        print(f"{k}: {v}条")
    print("\n=== 同行情况分布（已排除未知） ===")
    for k, v in comp_counts.most_common():
        print(f"{k}: {v}条")
    print(f" (另有 {total_valid - sum(comp_counts.values())} 条无法识别)")
    print("\n=== 关注方面分布 ===")
    for k, v in aspect_counts.most_common():
        print(f"{k}: {v}次 ({v/total_valid*100:.1f}%)")
    
    # 图1: 客户类型饼图
    plt.figure(figsize=(6, 6))
    labels = list(ctype_counts.keys())
    sizes = list(ctype_counts.values())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('客户类型分布')
    plt.tight_layout()
    plt.savefig('visualization/customer_type_pie.png', dpi=300)
    plt.close()
    
    # 图2: 同行情况柱状图（排除未知）
    comp_sorted = comp_counts.most_common()
    if comp_sorted:
        comp_labels = [item[0] for item in comp_sorted]
        comp_values = [item[1] for item in comp_sorted]
        plt.figure(figsize=(8, 5))
        plt.bar(comp_labels, comp_values, color='skyblue')
        plt.title('同行情况分布')
        plt.xlabel('同行类型')
        plt.ylabel('评论数量(条)')
        for i, v in enumerate(comp_values):
            plt.text(i, v + 1, str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('visualization/companion_bar.png', dpi=300)
        plt.close()
    
    # 图3: 关注方面柱状图（百分比）
    aspect_sorted = aspect_counts.most_common()
    if aspect_sorted:
        aspect_labels = [item[0] for item in aspect_sorted]
        aspect_values = [item[1] / total_valid * 100 for item in aspect_sorted]
        plt.figure(figsize=(8, 5))
        plt.bar(aspect_labels, aspect_values, color='lightcoral')
        plt.title('关注方面分布')
        plt.xlabel('关注方面')
        plt.ylabel('关注程度(%)')
        for i, v in enumerate(aspect_values):
            plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
        plt.ylim(0, max(aspect_values) * 1.15)
        plt.tight_layout()
        plt.savefig('visualization/aspect_bar.png', dpi=300)
        plt.close()
    
    print("\n图表已保存至 visualization/ 文件夹")

if __name__ == '__main__':
    main()