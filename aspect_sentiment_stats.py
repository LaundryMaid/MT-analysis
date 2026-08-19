"""
情感标注统计分析：对 dev.csv 的18列情感标注按5大类进行统计可视化
输出: 5张柱状图（正面/中性/负面占比）
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DEV_PATH = 'data/dev.csv'
OUTPUT_DIR = 'visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5个大类及其对应的ASAP标注列
ASPECT_GROUPS = {
    '位置': [
        ('Location#Transportation', '交通便利性'),
        ('Location#Downtown',       '市中心'),
        ('Location#Easy_to_find',   '易找到'),
    ],
    '服务': [
        ('Service#Queue',       '排队'),
        ('Service#Hospitality', '招待'),
        ('Service#Parking',     '停车'),
        ('Service#Timely',      '及时性'),
    ],
    '价格': [
        ('Price#Level',          '价格水平'),
        ('Price#Cost_effective', '性价比'),
        ('Price#Discount',       '折扣'),
    ],
    '环境': [
        ('Ambience#Decoration', '装修'),
        ('Ambience#Noise',      '噪音'),
        ('Ambience#Space',      '空间'),
        ('Ambience#Sanitary',   '卫生'),
    ],
    '食物': [
        ('Food#Portion',    '份量'),
        ('Food#Taste',      '口味'),
        ('Food#Appearance', '外观'),
        ('Food#Recommend',  '推荐度'),
    ],
}

COLOR_POOL = ['#E63946', '#F4B400', '#1E88E5', '#43A047']
LABEL_NAMES = ['负面(-1)', '中性(0)', '正面(1)']
SENT_KEYS = [-1, 0, 1]

def count_aspect(df, col):
    """统计某标注列的情感分布（排除-2未提及）"""
    s = df[col].apply(lambda x: int(x) if pd.notna(x) else -2)
    total = int((s != -2).sum())
    counts = {k: int((s == k).sum()) for k in SENT_KEYS}
    pcts = [counts[k] / total * 100 if total > 0 else 0 for k in SENT_KEYS]
    return total, counts, pcts

def plot_group(group_name, items, df):
    """为一个大类生成情感分布柱状图"""
    n = len(items)
    x = np.arange(len(LABEL_NAMES))
    width = 0.8 / n
    
    fig, ax = plt.subplots(figsize=(10, 6))
    max_pct = 0
    for i, (col, label) in enumerate(items):
        total, counts, pcts = count_aspect(df, col)
        offset = (i - (n - 1) / 2) * width
        color = COLOR_POOL[i % len(COLOR_POOL)]
        bars = ax.bar(x + offset, pcts, width, label=f'{label} ({total})',
                      color=color, edgecolor='white')
        for j, v in enumerate(pcts):
            if v > 0:
                ax.text(x[j] + offset, v + 0.5, f'{v:.1f}%', ha='center',
                        fontsize=8, color=color)
        max_pct = max(max_pct, max(pcts) if pcts else 0)
        print(f"  {label}: 提及={total}, 负面={counts[-1]}, 中性={counts[0]}, 正面={counts[1]}")
    
    ax.set_xlabel('情感极性', fontsize=13)
    ax.set_ylabel('评价数目占比 (%)', fontsize=13)
    ax.set_title(f'{group_name} - 情感标注分布', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.legend(loc='upper right', fontsize=10, title='方面(提及数)')
    ax.set_ylim(0, max_pct * 1.25 + 5)
    plt.tight_layout()
    fname = f'dev_sentiment_{group_name}.png'
    plt.savefig(f'{OUTPUT_DIR}/{fname}', dpi=300)
    plt.close()
    print(f"  -> 已保存: {OUTPUT_DIR}/{fname}")

def main():
    df = pd.read_csv(DEV_PATH, encoding='utf-8-sig')
    print(f"加载 {len(df)} 条评论")
    
    print("\n" + "=" * 60)
    print("  按5个大类统计情感标注分布")
    print("=" * 60)
    for group_name, items in ASPECT_GROUPS.items():
        print(f"\n[{group_name}] ({len(items)} 个方面)")
        plot_group(group_name, items, df)
    print(f"\n[Done] 共生成5张图, 保存到 {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()