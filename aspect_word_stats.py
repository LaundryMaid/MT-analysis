"""
评论文本分词统计：对 dev.csv 按5大类进行分词词频统计，生成CSV文件
输出: word_stats/*.csv (词语、出现次数、占比)
"""
import pandas as pd, os, re, warnings, csv
warnings.filterwarnings('ignore')
import jieba
import jieba.posseg as pseg
from collections import defaultdict

DEV_PATH = 'data/dev.csv'
OUTPUT_DIR = 'word_stats'
STOPWORDS_DIR = 'data/stopwords'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 五大类词汇分类词典（一词多义时归入多个类别）
WORD_CATEGORIES = {
    '位置': ['位置', '地点', '地址', '交通', '地铁', '公交', '商场', '市中心', '商圈',
             '附近', '周边', '对面', '旁边', '好找', '便利', '近', '远', '热闹', '地标'],
    '服务': ['服务', '态度', '热情', '冷漠', '周到', '贴心', '耐心', '专业', '礼貌',
             '服务员', '老板', '员工', '店员', '上菜', '等位', '排队', '及时', '迅速'],
    '环境': ['环境', '装修', '装饰', '氛围', '风格', '噪音', '吵', '安静', '卫生', '干净',
             '整洁', '脏', '宽敞', '狭窄', '空间', '座位', '包间', '大厅', '舒适', '温馨'],
    '食物': ['菜', '菜品', '味道', '口味', '好吃', '难吃', '美味', '鲜美', '香', '嫩', '脆',
             '份量', '新鲜', '特色', '羊肉串', '牛肉', '鱼', '汤', '面', '饭', '好吃', '推荐'],
    '价格': ['价格', '价钱', '费用', '人均', '贵', '便宜', '划算', '实惠', '性价比', '值',
             '优惠', '折扣', '团购', '优惠券', '公道', '坑', '宰客', '物美价廉'],
}

# 标点符号集合
PUNCTUATIONS = set('，。！？、：；""\'\'（）《》【】[]{}()<>"*,.;:!?\u3000 \t\n\r')
ENG_NUM_PATTERN = re.compile(r'^[a-zA-Z0-9\s\.\-]+$')

def load_official_stopwords():
    """加载5套官方停用词表，合并去重"""
    stopwords = set()
    files = ['hit_stopwords.txt', 'baidu_stopwords.txt', 'cn_stopwords.txt',
             'scu_stopwords.txt', 'jieba_stop_words.txt']
    loaded = []
    for fname in files:
        fpath = os.path.join(STOPWORDS_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [warn] 停用词表缺失: {fpath}")
            continue
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                w = line.strip()
                if w:
                    stopwords.add(w)
        loaded.append(f"{fname.split('_')[0]}({len([1 for _ in open(fpath, 'r', encoding='utf-8', errors='ignore').readlines() if _.strip()])})")
    print(f"  加载停用词: {', '.join(loaded)} -> 合计 {len(stopwords)} 个唯一词")
    return stopwords

def is_useful_word(word, flag, stopwords):
    """判断词是否保留：过滤停用词、纯数字/英文、单字、标点"""
    if not word:
        return False
    # 停用词过滤
    if word in stopwords:
        return False
    # 核心词典词优先保留
    if word in WORD_CATEGORIES:
        return True
    # 单字过滤
    if len(word) <= 1:
        return False
    # 纯数字/英文过滤
    if ENG_NUM_PATTERN.match(word):
        return False
    # 标点过滤
    if word in PUNCTUATIONS or re.match(r'^[\W_]+$', word, re.UNICODE):
        return False
    # 词性过滤：只保留名词、动词、形容词、副词
    useful_flags = {'n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v', 'a', 'ad', 'd', 'i', 'l', 'j'}
    if flag and flag[0] not in useful_flags:
        return False
    return True

def categorize_word(word):
    """返回该词所属的所有类别列表"""
    return WORD_CATEGORIES.get(word, [])

def main():
    """读取数据 -> 分词统计 -> 生成CSV"""
    print(f"加载 {DEV_PATH}...")
    df = pd.read_csv(DEV_PATH, encoding='utf-8-sig')
    print(f"  共 {len(df)} 条评论")
    
    # 加载停用词
    print("\n加载官方停用词表...")
    stopwords = load_official_stopwords()
    
    # 初始化结巴分词并加载自定义词典
    jieba.initialize()
    for w in WORD_CATEGORIES.keys():
        if len(w) >= 2:
            jieba.add_word(w)
    
    # 各类别词频统计容器
    cat_words = {cat: defaultdict(int) for cat in ['位置', '服务', '环境', '食物', '价格']}
    cat_total = {cat: 0 for cat in cat_words}
    
    processed = 0
    for _, row in df.iterrows():
        text = str(row.get('review', '') or '')
        if not text:
            continue
        # 分词并标注词性
        for word, flag in pseg.cut(text):
            if not is_useful_word(word, flag, stopwords):
                continue
            cats = categorize_word(word)
            if not cats:
                continue
            for cat in cats:
                if cat in cat_words:
                    cat_words[cat][word] += 1
                    cat_total[cat] += 1
        processed += 1
        if processed % 1000 == 0:
            print(f"  已处理 {processed}/{len(df)} 条")
    
    # 输出统计结果到CSV
    print("\n" + "=" * 60)
    print("  分词统计结果（按词频降序）")
    print("=" * 60)
    for cat in ['位置', '服务', '环境', '食物', '价格']:
        word_freq = cat_words[cat]
        total = cat_total[cat]
        sorted_items = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
        fname = f'word_stats_{cat}.csv'
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['词语', '出现次数', '占比(%)'])
            for word, cnt in sorted_items:
                pct = cnt / total * 100 if total > 0 else 0
                writer.writerow([word, cnt, f'{pct:.2f}'])
        print(f"\n[{cat}] 总词频={total}, 不同词数={len(word_freq)}")
        print(f"  文件: {fpath}")
        print(f"  Top 10:")
        for word, cnt in sorted_items[:10]:
            pct = cnt / total * 100 if total > 0 else 0
            print(f"    {word}: {cnt} ({pct:.2f}%)")
    
    print(f"\n[Done] 5个CSV已保存到 {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()