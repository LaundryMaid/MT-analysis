"""
智能回复模型：分析评论 + 商家信息 → 生成契合的回复
  - 评论分析：分词、情感、方面识别
  - 调用智谱 GLM-4-Flash API 生成回复，无API时降级为本地模板
依赖: pip install requests jieba
"""
import os, csv, json, re, warnings, random, sys
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import jieba
import jieba.posseg as pseg


def strip_emoji(text):
    """过滤emoji，避免GBK编码报错"""
    if not text:
        return text
    return re.sub(
        r'[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF'
        r'\U0001F1E0-\U0001F1FF\U00002B00-\U00002BFF\U0001F900-\U0001F9FF]',
        '', text
    )


def _load_api_key():
    """按优先级读取API key: 环境变量 → api_key.txt → config.json"""
    key = os.environ.get('ZHIPU_API_KEY', '').strip()
    if key:
        return key
    base = os.path.dirname(os.path.abspath(__file__))
    for fname in ['api_key.txt', 'config.json']:
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            continue
        try:
            if fname.endswith('.txt'):
                with open(fpath, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            else:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return (data.get('zhipu_api_key') or data.get('api_key') or '').strip()
        except Exception:
            pass
    return ''


# ============================ 配置 ============================
CONFIG = {
    'api_key': _load_api_key(),
    'base_url': 'https://open.bigmodel.cn/api/paas/v4',
    'model': 'glm-4-flash',
    'temperature': 0.7,
    'max_tokens': 350,
    'timeout': 30,
}

# ============================ 内置商家 ============================
MERCHANTS = [
    {
        'name': '老北京炭烤羊肉串', 'rating': 4.6, 'reputation': '口碑优秀',
        'features': ['炭烤羊肉串', '北方烧烤', '家常菜'],
        'specialties': ['羊肉串', '烤翅中', '烤羊腿', '酸梅汤'],
        'service_style': '热情周到,上菜迅速', 'price_range': '人均 ¥50-80',
        'characteristics': ['干净卫生', '人流量大', '适合家庭聚餐', '适合生日聚会'],
        'best_for': ['家庭聚餐', '朋友小聚', '生日聚会'],
    },
    {
        'name': '蜀香居川菜馆', 'rating': 4.3, 'reputation': '口碑良好',
        'features': ['正宗川菜', '麻辣风味', '江湖菜'],
        'specialties': ['水煮鱼', '麻婆豆腐', '宫保鸡丁', '辣子鸡'],
        'service_style': '店员礼貌,环境舒适', 'price_range': '人均 ¥60-90',
        'characteristics': ['装修雅致', '分量足', '口味正宗', '空调给力'],
        'best_for': ['朋友聚餐', '辣味爱好者'],
    },
    {
        'name': '湘味小厨', 'rating': 4.1, 'reputation': '口碑良好',
        'features': ['湖南菜', '家常小炒', '蒸菜'],
        'specialties': ['剁椒鱼头', '小炒黄牛肉', '湘西腊肉'],
        'service_style': '老板热情,服务到位', 'price_range': '人均 ¥40-70',
        'characteristics': ['性价比高', '味道地道', '分量足', '排队人多'],
        'best_for': ['工作餐', '家庭用餐'],
    },
]

# ============================ 词典 ============================
ASPECT_KEYWORDS = {
    '食物': ['羊肉串', '牛肉', '鸡肉', '鸭肉', '鱼', '虾', '蟹', '汤', '面', '饭',
             '粥', '点心', '甜点', '水果', '饮料', '茶', '酒', '咖啡', '奶茶',
             '果汁', '菜', '菜品', '菜肴', '菜单', '味道', '口味', '口感', '好吃',
             '难吃', '美味', '鲜美', '香', '鲜', '嫩', '脆', '份量', '量', '足',
             '够', '新鲜', '特色', '招牌菜', '凉菜', '热菜', '主食', '小吃',
             '烧烤', '火锅', '串串', '烤鱼', '炒饭', '面条', '米饭'],
    '服务': ['服务', '态度', '热情', '冷漠', '周到', '贴心', '耐心', '专业',
             '礼貌', '服务员', '老板', '员工', '店员', '上菜', '等位', '排队',
             '预约', '外卖', '打包', '及时', '迅速', '效率', '慢', '快'],
    '环境': ['环境', '装修', '装饰', '风格', '氛围', '气氛', '灯光', '音乐',
             '声音', '噪音', '吵', '安静', '卫生', '干净', '整洁', '脏', '乱',
             '宽敞', '狭窄', '空间', '座位', '包间', '大厅', '空调', '舒适',
             '舒服', '温馨', '雅致', '气派', '豪华', '人多', '人少'],
    '价格': ['价格', '价钱', '费用', '消费', '人均', '价位', '贵', '便宜',
             '划算', '实惠', '性价比', '值', '优惠', '折扣', '打折', '团购',
             '优惠券', '免费', '赠送', '公道', '合理', '坑', '宰客'],
    '位置': ['位置', '地点', '地址', '交通', '地铁', '公交', '商场', '市中心',
             '商圈', '附近', '周边', '对面', '旁边', '好找', '难找', '便利',
             '近', '远', '繁华', '偏僻', '热闹', '人流量', '地标'],
}

POS_WORDS = {'好吃':5, '美味':5, '鲜美':5, '香':4, '鲜':4, '嫩':4, '脆':4, '爽':4,
             '完美':5, '绝佳':5, '惊艳':5, '超赞':5, '极品':5, '一流':5, '棒':4,
             '赞':4, '出色':4, '不错':4, '干净':4, '整洁':4, '舒适':4, '优雅':5,
             '安静':4, '温馨':4, '宽敞':4, '明亮':4, '精致':5, '气派':5, '豪华':5,
             '热情':5, '周到':5, '贴心':5, '友好':4, '专业':4, '及时':4, '耐心':4,
             '值得':4, '推荐':4, '特色':4, '招牌':4, '新鲜':4, '地道':4, '正宗':4,
             '便宜':4, '划算':4, '实惠':4, '性价比':5, '物美价廉':5, '优惠':4,
             '方便':4, '便利':4, '好找':4, '近':3, '热闹':3, '人多':3,
             '迅速':4, '快':3, '舒服':4, '宜人':4, '出色':4,
             '入味':4, '给力':4, '细致':4, '用心':4, '良心':4}

NEG_WORDS = {'难吃':1, '腥':2, '腻':2, '油腻':2, '咸':2, '淡':2, '无味':1,
             '奇怪':2, '不新鲜':1, '老':2, '柴':2, '硬':2, '差':2, '糟糕':1,
             '失望':2, '脏':1, '乱':2, '吵':2, '嘈杂':2, '昏暗':2, '拥挤':2,
             '狭小':2, '破旧':2, '冷漠':1, '敷衍':1, '不耐烦':1, '不专业':1,
             '恶劣':1, '垃圾':1, '慢':2, '凶':2, '生气':2, '投诉':2,
             '贵':2, '坑':2, '宰客':1, '不值':1, '不划算':1, '暴利':1,
             '远':2, '偏僻':2, '难找':2, '不新鲜':1}

SCENE_KEYWORDS = {
    '生日': ['生日', '过生日', '庆生'],
    '聚会': ['聚会', '聚餐', '团建', '庆功'],
    '家庭': ['带孩子', '亲子', '一家', '家庭', '小孩', '老人'],
    '朋友': ['朋友', '同学', '同事'],
    '约会': ['约会', '情侣', '浪漫'],
    '工作餐': ['工作餐', '出差', '加班', '商务'],
    '节日': ['节日', '圣诞', '元旦', '春节', '中秋', '国庆'],
}

INTENSIFIERS = {'很':1.5, '非常':1.8, '特别':1.8, '极其':2.0, '超级':1.8,
                '超':1.5, '太':1.5, '真的':1.2, '有点':0.7, '稍微':0.7,
                '十分':1.8, '相当':1.5, '无比':2.0}
NEGATIONS = {'不', '没', '无', '非', '不是', '没有', '不太', '不够', '并不'}


# ============================ 评论分析 ============================
def analyze_review(text, merchant_name=''):
    """分析单条评论：分词、方面、情感、情境、关键词"""
    if not text:
        return {}
    words = list(jieba.cut(str(text)))

    # 方面识别
    aspects_mentioned = set()
    aspect_keywords = {}
    for aspect, kws in ASPECT_KEYWORDS.items():
        matched = [w for w in words if w in kws]
        if matched:
            aspects_mentioned.add(aspect)
            aspect_keywords[aspect] = matched

    # 情境识别
    scenes = []
    for scene, kws in SCENE_KEYWORDS.items():
        if any(kw in text for kw in kws):
            scenes.append(scene)

    # 情感打分
    scores = []
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
            scores.append(s)
        elif w in NEG_WORDS:
            s = NEG_WORDS[w] * mult
            if negated: s = min(5, 6 - s)
            scores.append(s)
        i += 1

    avg_score = sum(scores) / len(scores) if scores else 3.0
    if avg_score >= 4.0:
        sentiment = '正面'
    elif avg_score >= 3.0:
        sentiment = '中性'
    else:
        sentiment = '负面'

    # 关键词提取
    keywords = []
    for w, flag in pseg.cut(str(text)):
        if len(w) >= 2 and flag and flag[0] in {'n', 'vn', 'a', 'v', 'i', 'nz'} \
                and w not in {'我们', '你们', '他们', '这个', '那个', '什么', '怎么'}:
            keywords.append(w)
    seen = set()
    keywords = [k for k in keywords if not (k in seen or seen.add(k))][:8]

    return {
        'text': text, 'words': words,
        'aspects': sorted(aspects_mentioned), 'aspect_keywords': aspect_keywords,
        'scenes': scenes, 'sentiment': sentiment,
        'score': round(avg_score, 2), 'keywords': keywords,
        'merchant_name': merchant_name,
    }


# ============================ 商家推荐 ============================
def get_merchant_recommendation(merchant):
    """根据商家口碑给出回复策略"""
    rating = merchant.get('rating', 4.0)
    if rating >= 4.5:
        return {'level': '优秀', 'strategy': '强化推荐,引导复购与口碑传播',
                'phrases': ['期待您的再次光临', '欢迎常来', '值得推荐给身边朋友'],
                'tone': '自信温暖'}
    elif rating >= 4.0:
        return {'level': '良好', 'strategy': '保持优势,弱化短板,引导尝试新品',
                'phrases': ['欢迎下次尝试我们的特色菜', '期待再次相遇', '欢迎您带朋友来'],
                'tone': '诚恳热情'}
    elif rating >= 3.5:
        return {'level': '一般', 'strategy': '诚恳感谢,主动承诺改进,弱化推荐',
                'phrases': ['我们会持续改进', '感谢您的反馈,会做得更好', '欢迎再次体验改进后的服务'],
                'tone': '诚恳谦逊'}
    else:
        return {'level': '待提升', 'strategy': '致歉优先,真诚改进,不主动推荐',
                'phrases': ['非常抱歉给您带来不佳体验', '我们会认真改进', '欢迎您再次给予我们机会'],
                'tone': '致歉诚恳'}


# ============================ Prompt构造 ============================
def build_prompt(review_text, merchant, analysis, rec):
    """构造API请求的prompt"""
    merchant_desc = (
        f"【商家信息】\n"
        f"店名: {merchant['name']}\n"
        f"口碑: {merchant['reputation']} (评分 {merchant['rating']})\n"
        f"主营: {', '.join(merchant['features'])}\n"
        f"招牌菜: {', '.join(merchant['specialties'])}\n"
        f"服务风格: {merchant['service_style']}\n"
        f"价格区间: {merchant['price_range']}\n"
        f"商家特点: {', '.join(merchant['characteristics'])}\n"
        f"适合场景: {', '.join(merchant['best_for'])}"
    )

    analysis_desc = (
        f"【评论分析】\n"
        f"提及方面: {', '.join(analysis['aspects']) if analysis['aspects'] else '无明显方面'}\n"
        f"情境标签: {', '.join(analysis['scenes']) if analysis['scenes'] else '普通用餐'}\n"
        f"情感倾向: {analysis['sentiment']} (评分 {analysis['score']}/5)\n"
        f"关键词: {', '.join(analysis['keywords']) if analysis['keywords'] else '无'}"
    )

    rec_desc = (
        f"【商家口碑等级: {rec['level']}】\n"
        f"推荐策略: {rec['strategy']}\n"
        f"语气基调: {rec['tone']}"
    )

    user_review = f"【顾客评论】\n{review_text}"

    instructions = (
        "【任务要求】\n"
        "1. 准确理解评论内容，识别关键信息（菜品、情境、情感）\n"
        "2. 回复要呼应顾客提到的菜品/服务/环境/情境\n"
        "3. 语气与评论情感契合（好评温暖感谢，中评诚恳改进，差评致歉）\n"
        "4. 根据商家口碑等级决定推荐强度\n"
        "5. 有特殊情境（生日/聚会/家庭等）时体现关怀\n"
        "6. 回复80-150字，自然流畅，不堆砌套话\n"
        "7. 紧扣本条评论的具体内容\n"
        "8. 只输出回复正文，不要加引号或解释"
    )

    return f"{merchant_desc}\n\n{analysis_desc}\n\n{rec_desc}\n\n{user_review}\n\n{instructions}"


# ============================ API调用 ============================
def call_zhipu_api(prompt):
    """调用智谱GLM-4-Flash API"""
    try:
        import requests
    except ImportError:
        return None, 'requests库未安装，请运行: pip install requests'

    if not CONFIG['api_key']:
        return None, 'API key未配置（设置环境变量ZHIPU_API_KEY或创建api_key.txt）'

    headers = {
        'Authorization': f'Bearer {CONFIG["api_key"]}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': CONFIG['model'],
        'messages': [
            {'role': 'system', 'content': '你是一位资深餐厅客服，擅长为不同口碑的商家撰写契合评论的智能回复。'},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': CONFIG['temperature'],
        'max_tokens': CONFIG['max_tokens'],
    }
    try:
        resp = requests.post(
            f"{CONFIG['base_url']}/chat/completions",
            headers=headers, json=data, timeout=CONFIG['timeout']
        )
        if resp.status_code != 200:
            return None, f"API错误{resp.status_code}: {resp.text[:200]}"
        content = resp.json()['choices'][0]['message']['content'].strip()
        return strip_emoji(content), None
    except Exception as e:
        return None, f"请求异常: {e}"


# ============================ 本地模板 ============================
def local_template_reply(review_text, merchant, analysis, rec):
    """基于规则模板生成回复（无API时的降级方案）"""
    aspects = analysis['aspects']
    scenes = analysis['scenes']
    sentiment = analysis['sentiment']
    keywords = analysis['keywords']
    mname = merchant['name']

    # 开头
    greetings = [f"感谢您光临{mname}"]
    scene_greetings = {
        '生日': "能为您或家人朋友庆祝生日倍感荣幸，祝生日快乐",
        '聚会': "愿我们的菜品为您与朋友相聚增添一份温暖",
        '家庭': "感谢您带家人选择我们，让全家人满意是我们的目标",
        '朋友': "感谢您与朋友一同光临，与好友共享美食是种幸福",
        '约会': "愿这里成为您与Ta的甜蜜回忆",
        '节日': "节日快乐，感谢您选择与我们一起度过",
        '工作餐': "工作辛苦了，愿这顿饭让您稍作休息",
    }
    greetings.append(scene_greetings.get(scenes[0] if scenes else '', "感谢您抽出宝贵时间与我们分享用餐体验"))

    # 中段
    body_parts = []
    for spec in merchant['specialties']:
        if spec in review_text or spec in keywords:
            body_parts.append(f"{spec}能得到您的认可，我们十分欣慰，会继续保持")
            break

    aspect_templates = {
        '食物': {
            '正面': "菜品的味道是我们最看重的，您的好评是厨师团队最大的动力",
            '负面': "菜品未能让您满意，我们会认真检视，不断改进",
            '中性': "您的反馈将帮助我们提升菜品质量",
        },
        '环境': {
            '正面': "环境的舒适是我们一直的追求，期待下次为您带来更好体验",
            '负面': "环境方面我们会认真改进",
        },
        '服务': {
            '正面': "服务员的热情周到是我们的招牌，期待再次为您提供贴心服务",
            '负面': "服务方面我们会加强培训，力求更加专业",
        },
        '价格': {
            '正面': "我们会继续坚持高性价比，让每位顾客都吃得满意",
            '负面': "关于价格，我们会认真反思定位，力求让品质配得上消费",
        },
        '位置': {
            '正面': "门店位置确实方便，欢迎您随时过来",
            '负面': "位置方面给您带来不便，我们会在指引上做得更清楚",
        },
    }
    for aspect in aspects:
        if aspect in aspect_templates:
            if aspect == '环境' and ('干净' in review_text or '卫生' in review_text):
                body_parts.append("我们会持续保持干净卫生的用餐环境")
            elif aspect == '环境':
                body_parts.append(aspect_templates['环境'][sentiment] if sentiment in aspect_templates['环境'] else aspect_templates['环境']['正面'])
            elif aspect == '价格' and ('便宜' in review_text or '划算' in review_text):
                body_parts.append(aspect_templates['价格']['正面'])
            elif aspect == '价格' and '贵' in review_text:
                body_parts.append(aspect_templates['价格']['负面'])
            elif aspect == '位置' and ('好找' in review_text or '便利' in review_text):
                body_parts.append(aspect_templates['位置']['正面'])
            elif aspect == '位置' and ('难找' in review_text or '远' in review_text):
                body_parts.append(aspect_templates['位置']['负面'])
            elif sentiment in aspect_templates.get(aspect, {}):
                body_parts.append(aspect_templates[aspect][sentiment])

    # 结尾
    ending_map = {
        '优秀': ["期待您再次光临，也欢迎您带朋友来体验", "再次感谢，祝您生活愉快"],
        '良好': ["欢迎您下次再来尝试我们的其他特色菜", "再次感谢，期待与您的再次相遇"],
        '一般': ["我们会持续改进，期待下次给您带来更好体验"],
        '待提升': ["再次致歉，我们会认真改进，欢迎您再次给予我们机会"],
    }
    endings = ending_map.get(rec['level'], ending_map['良好'])

    parts = greetings + body_parts + [random.choice(endings)]
    return '，'.join(parts) + '。'


# ============================ 主流程 ============================
def generate_reply(review_text, merchant, use_api=True):
    """生成智能回复：优先API，否则本地模板"""
    analysis = analyze_review(review_text, merchant['name'])
    rec = get_merchant_recommendation(merchant)

    if use_api and CONFIG['api_key']:
        prompt = build_prompt(review_text, merchant, analysis, rec)
        reply, err = call_zhipu_api(prompt)
        if err:
            print(f"  [warn] API调用失败，降级本地模板: {err}")
            reply = local_template_reply(review_text, merchant, analysis, rec)
            source = '本地模板(API失败降级)'
        else:
            source = '智谱GLM-4-Flash API'
    else:
        reply = local_template_reply(review_text, merchant, analysis, rec)
        source = '本地规则模板'

    return {'reply': reply, 'source': source, 'analysis': analysis, 'recommendation': rec}


def demo():
    """演示：对示例评论和dev.csv生成回复"""
    output_path = 'aspect_reply_demo.txt'
    log_lines = []

    def log(msg=''):
        print(msg)
        log_lines.append(str(msg))

    log("=" * 70)
    log("  智能回复模型演示（智谱GLM-4-Flash）")
    log("=" * 70)

    if not CONFIG['api_key']:
        log("[提示] 未配置ZHIPU_API_KEY，将使用本地规则模板")
        log("       配置方法: 创建api_key.txt文件，内容为API key")
    else:
        log(f"[配置] 已加载API key: {CONFIG['api_key'][:6]}...{CONFIG['api_key'][-4:]}")
        log(f"       模型: {CONFIG['model']}, Base URL: {CONFIG['base_url']}")
    log()

    sample_reviews = [
        ('带孩子过生日来吃饭，羊肉串超级好吃，干净卫生，下次还来。', MERCHANTS[0]),
        ('水煮鱼味道很正宗, 麻辣够劲, 但是上菜有点慢, 整体还算满意。', MERCHANTS[1]),
        ('价格实惠份量足, 一家人吃得开心, 老板很热情, 推荐剁椒鱼头。', MERCHANTS[2]),
        ('环境嘈杂排队太久, 菜品味道一般, 性价比不高, 不会再来。', MERCHANTS[0]),
    ]

    for i, (review, merchant) in enumerate(sample_reviews, 1):
        log(f"\n{'-'*70}")
        log(f"[案例{i}] 商家: {merchant['name']} (口碑{merchant['rating']})")
        log(f"评论: {review}")
        result = generate_reply(review, merchant, use_api=True)
        log(f"分析: 方面={result['analysis']['aspects']}, "
            f"情境={result['analysis']['scenes']}, "
            f"情感={result['analysis']['sentiment']}({result['analysis']['score']}/5), "
            f"关键词={result['analysis']['keywords']}")
        log(f"口碑等级: {result['recommendation']['level']}, "
            f"策略={result['recommendation']['strategy']}")
        log(f"回复来源: {result['source']}")
        log(f"智能回复: {result['reply']}")
        log()

    # 批量处理dev.csv
    log("=" * 70)
    log("  批量处理dev.csv评论（示例前5条）")
    log("=" * 70)
    try:
        import pandas as pd
        df = pd.read_csv('data/dev.csv', encoding='utf-8-sig')
        merchant = MERCHANTS[0]
        for idx, row in df.head(5).iterrows():
            review = str(row.get('review', '') or '')[:80] + ('...' if len(str(row.get('review', ''))) > 80 else '')
            star = row.get('star', '?')
            log(f"\n[评论#{idx}] 评分={star}")
            log(f"评论: {review}")
            result = generate_reply(str(row.get('review', '') or ''), merchant, use_api=True)
            log(f"回复: {result['reply'][:300]}")
            log(f"  (来源: {result['source']})")
    except Exception as e:
        log(f"批量处理失败: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print(f"\n[输出已保存] {output_path}")


if __name__ == '__main__':
    demo()