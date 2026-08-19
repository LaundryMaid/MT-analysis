"""
Voting模型评分预测：TF-IDF + 情感词典特征 → XGBoost+LightGBM+LR+RF 软投票
  1. train.csv训练，test.csv评估口味/环境/服务三方面
  2. dev.csv分析位置/价格评分分布
"""
import pandas as pd, numpy as np, os, re, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import lightgbm as lgb

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TRAIN_PATH = 'data/asap-master/data/train.csv'
TEST_PATH = 'data/asap-master/data/test.csv'
DEV_PATH = 'data/dev.csv'
OUTPUT_DIR = 'visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练集方面映射
ASPECT_MAP = {
    '口味': ['Food#Taste', 'Food#Portion', 'Food#Appearance', 'Food#Recommend'],
    '环境': ['Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary'],
    '服务': ['Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely']
}
# dev.csv方面映射
DEV_ASPECT_MAP = {
    '位置': ['Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find'],
    '价格': ['Price#Level', 'Price#Cost_effective', 'Price#Discount'],
}

# 情感词典
POS_WORDS = {
    '好吃':5,'美味':5,'鲜美':5,'香':4,'鲜':4,'嫩':4,'脆':4,'滑':4,'爽':4,
    '甜':4,'完美':5,'绝佳':5,'惊艳':5,'超赞':5,'极品':5,'一流':5,
    '入味':4,'地道':4,'正宗':4,'新鲜':4,'棒':4,'赞':4,'出色':4,'不错':3,
    '干净':4,'整洁':4,'舒适':4,'优雅':5,'安静':4,'温馨':4,'宽敞':4,'明亮':4,
    '气派':5,'精致':5,'舒服':4,'宜人':4,'高档':5,'豪华':5,'雅致':5,'清新':4,
    '热情':5,'周到':5,'贴心':5,'友好':4,'专业':4,'及时':4,'快':3,
    '耐心':4,'主动':4,'微笑':4,'负责':4,'细心':4,'有礼貌':4,'给力':4,
    '值得':4,'推荐':4,'招牌':4,'特色':4,
}
NEG_WORDS = {
    '难吃':1,'腥':2,'腻':2,'油腻':2,'咸':2,'淡':2,'无味':1,'奇怪':2,
    '不新鲜':1,'老':2,'柴':2,'硬':2,'差':2,'糟糕':1,'坑':2,'失望':2,
    '脏':1,'乱':2,'吵':2,'嘈杂':2,'昏暗':2,'拥挤':2,'狭小':2,'破旧':2,
    '冷漠':1,'敷衍':1,'不耐烦':1,'不专业':1,'恶劣':1,'垃圾':1,'可怕':1,
    '慢':2,'错':2,'漏':2,'凶':2,'生气':2,'愤怒':1,'投诉':2,
}
INTENSIFIERS = {'很':1.5,'非常':1.8,'特别':1.8,'极其':2.0,'超级':1.8,'超':1.5,'太':1.5,'真的':1.2,'有点':0.7,'稍微':0.7,'十分':1.8,'相当':1.5,'无比':2.0}
NEGATIONS = {'不','没','无','非','不是','没有','不太','不够','并不'}

def compute_sentiment(text):
    """计算情感得分，返回12维特征向量"""
    if pd.isna(text): text = ''
    words = list(jieba.cut(str(text)))
    scores, pos_c, neg_c = [], 0, 0
    i = 0
    while i < len(words):
        w, mult, negated = words[i], 1.0, False
        if w in INTENSIFIERS:
            mult = INTENSIFIERS[w]; i += 1
            if i < len(words): w = words[i]
            else: break
        if w in NEGATIONS:
            negated = True; i += 1
            if i < len(words): w = words[i]
            else: break
        if w in POS_WORDS:
            s = POS_WORDS[w] * mult
            if negated: s = max(1, 6 - s)
            scores.append(s); pos_c += 1
        elif w in NEG_WORDS:
            s = NEG_WORDS[w] * mult
            if negated: s = min(5, 6 - s)
            scores.append(s); neg_c += 1
        i += 1
    if not scores: scores = [3.0]
    avg = np.mean(scores)
    return [avg, np.median(scores), np.std(scores), min(scores), max(scores),
            pos_c, neg_c, pos_c+neg_c, pos_c/max(len(words),1), neg_c/max(len(words),1),
            len(set(words))/max(len(words),1), len(words),
            -1 if neg_c>pos_c else (1 if pos_c>neg_c else 0)]

def _stat_feats(texts):
    """提取文本统计特征：长度、句子数、标点等"""
    feats = []
    for t in texts:
        words = list(jieba.cut(str(t)))
        sents = [s.strip() for s in re.split(r'[。！？!?.\n]+', str(t)) if s.strip()]
        feats.append([len(t), len(words), len(sents), len(words)/max(len(sents),1),
                      len(set(words))/max(len(words),1),
                      t.count('！')+t.count('!'), t.count('？')+t.count('?'), t.count('。')])
    return np.array(feats)

def extract_features(texts):
    """提取全部特征：TF-IDF + 情感特征 + 统计特征"""
    sent = np.array([compute_sentiment(t) for t in texts])
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)
    tfidf = vec.fit_transform(texts)
    stat = _stat_feats(texts)
    s1, s2 = StandardScaler(), StandardScaler()
    return hstack([tfidf, csr_matrix(s1.fit_transform(sent)), csr_matrix(s2.fit_transform(stat))]), vec, s1, s2

def transform_features(texts, vec, s1, s2):
    """用已训练的转换器提取特征"""
    sent = np.array([compute_sentiment(t) for t in texts])
    tfidf = vec.transform(texts)
    stat = _stat_feats(texts)
    return hstack([tfidf, csr_matrix(s1.transform(sent)), csr_matrix(s2.transform(stat))])

def get_aspect_mask(df, cols):
    """获取某方面的有效数据掩码"""
    mask = pd.Series([False]*len(df), index=df.index)
    for _, row in df.iterrows():
        for col in cols:
            if pd.notna(row.get(col, -2)) and int(row.get(col, -2)) != -2:
                mask[row.name] = True; break
    return mask

def metrics(yt, yp):
    """计算评估指标：准确率、宏F1、 exact match、1分以内"""
    acc = accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average='macro')
    w1 = sum(1 for t,p in zip(yt,yp) if abs(t-p)<=1)/len(yt)*100
    return acc, mf1, acc*100, w1

def plot_compare(aspect, yt, yp, name):
    """绘制实际vs模型评分分布对比图"""
    x = np.arange(5); w = 0.35
    tc = [yt.count(s) for s in [1,2,3,4,5]]
    pc = [yp.count(s) for s in [1,2,3,4,5]]
    total = sum(tc)
    tp = [c/total*100 for c in tc]; pp = [c/total*100 for c in pc]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x-w/2, tp, w, label='消费者实际评分', color='royalblue')
    ax.bar(x+w/2, pp, w, label=f'{name}', color='crimson')
    ax.set_xticks(x); ax.set_xticklabels(['1分-极差','2分-较差','3分-一般','4分-良好','5分-优秀'])
    ax.set_ylabel('评价数目占比(%)'); ax.set_title(f'{aspect}')
    ax.legend(loc='upper right'); ax.set_ylim(0, max(max(tp),max(pp))*1.3+5)
    for i,(v1,v2) in enumerate(zip(tp,pp)):
        if v1>0: ax.text(x[i]-w/2, v1+0.5, f'{v1:.1f}%', ha='center', fontsize=8, color='navy')
        if v2>0: ax.text(x[i]+w/2, v2+0.5, f'{v2:.1f}%', ha='center', fontsize=8, color='darkred')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{aspect}_{name}_eval.png', dpi=300); plt.close()

# ============================ 主流程 ============================
print("加载数据...")
train_df = pd.read_csv(TRAIN_PATH, encoding='utf-8-sig')
test_df = pd.read_csv(TEST_PATH, encoding='utf-8-sig')
jieba.initialize()

print("提取特征...")
train_texts = train_df['review'].fillna('').tolist()
test_texts = test_df['review'].fillna('').tolist()
y_train = np.array(train_df['star'].astype(int).tolist())
y_test = np.array(test_df['star'].astype(int).tolist())

X_train, vec, s1, s2 = extract_features(train_texts)
X_test = transform_features(test_texts, vec, s1, s2)
print(f"  训练: {len(train_texts)}, 测试: {len(test_texts)}, 特征: {X_train.shape[1]}")

# 各方面掩码
aspect_masks_test = {a: get_aspect_mask(test_df, c).values for a, c in ASPECT_MAP.items()}

# 训练Voting模型
print("\n训练 Voting (XGBoost + LightGBM + LR + RF)...")
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
nc = len(le.classes_)
counts = np.bincount(y_train_enc)
spw = [counts.max()/max(c,1) for c in counts]

xgb_m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.08,
                           scale_pos_weight=spw, use_label_encoder=False,
                           eval_metric='mlogloss', random_state=42, verbosity=0,
                           objective='multi:softprob', num_class=nc,
                           subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)
lgb_m = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.08,
                            class_weight='balanced', random_state=42, verbosity=-1,
                            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=0.1)
lr_m = LogisticRegression(C=2.0, max_iter=5000, class_weight='balanced', random_state=42)
rf_m = RandomForestClassifier(n_estimators=300, max_depth=30, class_weight='balanced',
                                random_state=42, n_jobs=-1, min_samples_leaf=3)

voting = VotingClassifier(
    estimators=[('xgb', xgb_m), ('lgb', lgb_m), ('lr', lr_m), ('rf', rf_m)],
    voting='soft', n_jobs=-1
)
voting.fit(X_train, y_train)

# ============================ test.csv 评估 ============================
print("\n" + "="*60)
print("  test.csv 分方面评估（模型）")
print("="*60)

for aspect, cols in ASPECT_MAP.items():
    mask = aspect_masks_test[aspect]
    yp = voting.predict(X_test[mask]).tolist()
    yt = y_test[mask].tolist()
    acc, mf1, exact, w1 = metrics(yt, yp)
    plot_compare(aspect, yt, yp, '模型评分')
    flag = '[OK]' if exact >= 65 else '[X]'
    print(f"  {flag} {aspect}: acc={acc:.4f} exact={exact:.1f}% within1={w1:.1f}%")

# ============================ dev.csv 分析 ============================
print("\n" + "="*60)
print("  dev.csv 分析: 位置/价格 评分分布（star列 + Voting模型）")
print("="*60)

dev_df = pd.read_csv(DEV_PATH, encoding='utf-8-sig')
dev_texts = dev_df['review'].fillna('').tolist()
X_dev = transform_features(dev_texts, vec, s1, s2)
y_dev_pred = voting.predict(X_dev).tolist()

# 实际评分直接取自star列（取整1-5）
dev_stars = dev_df['star'].fillna(3).apply(lambda x: max(1, min(5, int(round(float(x)))))).tolist()
print(f"  dev.csv: {len(dev_df)} 条, Voting预测分布: {[(s, y_dev_pred.count(s)) for s in sorted(set(y_dev_pred))]}")
print(f"  dev.csv star分布: {[(s, dev_stars.count(s)) for s in sorted(set(dev_stars))]}")

aspect_pred, aspect_actual = {}, {}
for aspect_name, cols in DEV_ASPECT_MAP.items():
    indices = []
    for idx, (_, row) in enumerate(dev_df.iterrows()):
        labels = [int(row.get(c, -2)) for c in cols if pd.notna(row.get(c, -2)) and int(row.get(c, -2)) != -2]
        if labels:
            indices.append(idx)
    aspect_pred[aspect_name] = [y_dev_pred[i] for i in indices]
    aspect_actual[aspect_name] = [dev_stars[i] for i in indices]
    print(f"  {aspect_name}: {len(indices)} 条提及")

# 颜色与标签配置
x = np.arange(5); w = 0.35
colors = {'位置': 'royalblue', '价格': 'forestgreen'}
labels_map = {'位置': '位置评分', '价格': '价格评分'}
score_range = [1, 2, 3, 4, 5]
xtick_labels = ['1分-极差', '2分-较差', '3分-一般', '4分-良好', '5分-优秀']


def plot_dev_aspect(data, title, filename):
    """单类数据柱状图：蓝色=位置, 绿色=价格"""
    fig, ax = plt.subplots(figsize=(10, 6))
    max_pct = 0
    for i, (aspect_name, scores) in enumerate(data.items()):
        total = len(scores)
        if total == 0:
            continue
        pcts = [scores.count(s) / total * 100 for s in score_range]
        offset = (i - 0.5) * w
        ax.bar(x + offset, pcts, w, label=labels_map[aspect_name],
               color=colors[aspect_name], edgecolor='white')
        for j, v in enumerate(pcts):
            if v > 0:
                ax.text(x[j] + offset, v + 0.5, f'{v:.1f}%', ha='center',
                        fontsize=8, color=colors[aspect_name])
        max_pct = max(max_pct, max(pcts))
    ax.set_xlabel('评分等级', fontsize=13)
    ax.set_ylabel('评价数目占比 (%)', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(xtick_labels)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, max_pct * 1.3 + 5)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=300); plt.close()


def plot_dev_compare(actual, pred, title, filename):
    """位置/价格 实际vs模型 对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    max_pct = 0
    width = 0.2
    for i, (aspect_name, _) in enumerate(actual.items()):
        a_scores = actual[aspect_name]
        p_scores = pred[aspect_name]
        ta, tp = len(a_scores), len(p_scores)
        if ta == 0 or tp == 0:
            continue
        a_pcts = [a_scores.count(s) / ta * 100 for s in score_range]
        p_pcts = [p_scores.count(s) / tp * 100 for s in score_range]
        actual_color = colors[aspect_name]
        from matplotlib.colors import to_rgba
        pred_color = (*to_rgba(colors[aspect_name])[:3], 0.5)
        ax.bar(x + (i - 0.5) * width * 2, a_pcts, width,
               label=f'{aspect_name}实际(star)', color=actual_color, edgecolor='white')
        ax.bar(x + (i - 0.5) * width * 2 + width, p_pcts, width,
               label=f'{aspect_name}模型', color=pred_color, edgecolor='white',
               hatch='//')
        max_pct = max(max_pct, max(a_pcts), max(p_pcts))
    ax.set_xlabel('评分等级', fontsize=13)
    ax.set_ylabel('评价数目占比 (%)', fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(xtick_labels)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max_pct * 1.3 + 5)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{filename}', dpi=300); plt.close()


# 1) 基于star列的实际评分分布
plot_dev_aspect(aspect_actual,
                'dev.csv - 位置/价格 实际评分分布(star列)',
                'dev_star_location_price.png')

# 2) Voting模型预测评分分布
plot_dev_aspect(aspect_pred,
                'dev.csv - 位置/价格 模型预测评分分布',
                'dev_voting_location_price.png')

# 3) 对比图：实际(star) vs Voting模型预测
plot_dev_compare(aspect_actual, aspect_pred,
                 'dev.csv - 位置/价格 实际(star) vs 模型',
                 'dev_compare_location_price.png')

print(f"\n[Done] charts saved to {OUTPUT_DIR}/")