"""
评论拆分模块：将长评论按标点切分为句子片段，并标记每个片段涉及的方面
输出: data/segmented_reviews.csv
"""
import pandas as pd
import re
import os

# 方面关键词词典
ASPECT_DICT = {
    '口味': ['味道', '口味', '好吃', '鲜美', '口感', '入味', '清淡', '油腻', '咸', '甜', '辣', '香', '食材', '原料', '新鲜'],
    '环境': ['环境', '装修', '氛围', '布置', '雅致', '吵闹', '安静', '干净', '整洁', '空间', '位置'],
    '服务': ['服务', '服务员', '热情', '态度', '上菜', '响应', '贴心', '周到', '耐心', '友好'],
    '价格': ['价格', '价钱', '性价比', '便宜', '贵', '实惠', '划算', '收费', '不值', '值得'],
    '卫生': ['卫生', '干净', '脏', '苍蝇', '虫', '餐具'],
    '位置': ['位置', '交通', '停车', '方便', '好找', '地段', '周边', '坐落']
}

def extract_aspects(text):
    """从文本中提取提及的方面（基于关键词匹配）"""
    if not text or pd.isna(text):
        return []
    aspects = set()
    for aspect, words in ASPECT_DICT.items():
        if any(w in str(text) for w in words):
            aspects.add(aspect)
    return list(aspects)

def split_segments(text):
    """按标点切分句子片段，并为每个片段关联方面"""
    if not text or pd.isna(text):
        return []
    # 按句号、感叹号等切分
    sentences = re.split(r'[。！？；]', str(text))
    segments = []
    for sent in sentences:
        sent = sent.strip()
        if sent:
            segments.append({'text': sent, 'aspects': extract_aspects(sent)})
    return segments

def main():
    """读取评论 -> 拆分句子 -> 标记方面 -> 保存结果"""
    os.makedirs('data', exist_ok=True)
    
    try:
        df = pd.read_csv('data/dev.csv', encoding='utf-8-sig')
    except FileNotFoundError:
        print("错误: 未找到 dev.csv 文件")
        return
    
    print(f"总评论数: {len(df)}")
    print("开始拆分评论...")
    
    segmented_data = []
    for idx, row in df.iterrows():
        review = row.get('review', '')
        star = row.get('star', 3.0)
        if pd.isna(review) or not str(review).strip():
            continue
        
        segments = split_segments(str(review).strip())
        segment_texts = [seg['text'] for seg in segments]
        segment_aspects = [','.join(seg['aspects']) for seg in segments]
        
        # 聚合所有片段到一行
        segmented_data.append({
            'review_id': idx,
            'star': star,
            'original_review': str(review).strip(),
            'segment_count': len(segments),
            'segments': ' ||| '.join(segment_texts),
            'segment_aspects': ' ||| '.join(segment_aspects)
        })
        
        if (idx + 1) % 500 == 0:
            print(f"已处理 {idx+1} 条")
    
    out_df = pd.DataFrame(segmented_data)
    out_df.to_csv('data/segmented_reviews.csv', index=False, encoding='utf-8-sig')
    print(f"\n拆分完成, 共处理 {len(out_df)} 条评论, 结果已保存到 data/segmented_reviews.csv")

if __name__ == '__main__':
    main()