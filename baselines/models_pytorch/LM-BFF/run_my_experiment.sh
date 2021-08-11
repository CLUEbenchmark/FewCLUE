SEED=13
# TASK=bustm
TASK=tnews
# TASK=iflytek
# TASK=csl
# TASK=csldcp
# TASK=eprstmt
# TASK=ocnli
# TASK=cluewsc

MODEL=hfl/chinese-roberta-wwm-ext
# MODEL=hfl/chinese-roberta-wwm-ext-large
TYPE=prompt

K=16
# 训练集的长度设置
# eprstmt TASK_EXTRA="--max_seq_len 70 --first_sent_limit 64"
# bustm TASK_EXTRA="--max_seq_len 140 --first_sent_limit 64 --other_sent_limit 64"
# ocnli TASK_EXTRA="--max_seq_len 140 --first_sent_limit 64 --other_sent_limit 64"
# tnews TASK_EXTRA="--max_seq_len 70 --first_sent_limit 64"
# iflytek TASK_EXTRA="--max_seq_len 512 --first_sent_limit 500"
# csl TASK_EXTRA="--max_seq_len 512 --first_sent_limit 500"
# csldcp TASK_EXTRA="--max_seq_len 512 --first_sent_limit 500"
# cluewsc TASK_EXTRA="--max_seq_len 300 --first_sent_limit 256"

# 在此设置 max len
# 稍小的max len加快训练速度
TASK_EXTRA="--max_seq_len 140 --first_sent_limit 64 --other_sent_limit 64"
# TASK_EXTRA="--max_seq_len 270 --first_sent_limit 256"
# TASK_EXTRA="--max_seq_len 70 --first_sent_limit 64"
# TASK_EXTRA="--max_seq_len 140 --first_sent_limit 128"
# TASK_EXTRA="--max_seq_len 512 --first_sent_limit 500"
# TASK_EXTRA="--max_seq_len 300 --first_sent_limit 256"

LR=1e-5

MAX_STEP=1000
# MAX_STEP=300
# Validation steps
EVAL_STEP=300

# TEMPLATE=*cls**sent_0*_。_*mask*点!*sep+*
# 字符串表示的dict没有任何空格！
MAPPING="{100:'故事',101:'文化',102:'娱乐',103:'体育',104:'财经',106:'房产',107:'汽车',108:'教育',109:'科技',110:'军事',112:'旅游',113:'国际',114:'股票',115:'农业',116:'电竞'}"
# cluewsc MAPPING="{False:'否',True:'是'}"
# csldcp MAPPING="{'材料科学与工程':'材料','作物学':'作物','口腔医学':'口腔','药学':'药学','教育学':'教育','水利工程':'水利','理论经济学':'理经','食品科学与工程':'食品','畜牧学/兽医学':'兽医','体育学':'体育','核科学与技术':'核能','力学':'力学','园艺学':'园艺','水产':'水产','法学':'法学','地质学/地质资源与地质工程':'地质','石油与天然气工程':'能源','农林经济管理':'农林','信息与通信工程':'通信','图书馆、情报与档案管理':'情报','政治学':'政治','电气工程':'电气','海洋科学':'海洋','民族学':'民族','航空宇航科学与技术':'航空','化学/化学工程与技术':'化工','哲学':'哲学','公共卫生与预防医学':'卫生','艺术学':'艺术','农业工程':'农工','船舶与海洋工程':'船舶','计算机科学与技术':'计科','冶金工程':'冶金','交通运输工程':'交通','动力工程及工程热物理':'动力','纺织科学与工程':'纺织','建筑学':'建筑','环境科学与工程':'环境','公共管理':'公管','数学':'数学','物理学':'物理','林学/林业工程':'林业','心理学':'心理','历史学':'历史','工商管理':'工商','应用经济学':'应经','中医学/中药学':'中医','天文学':'天文','机械工程':'机械','土木工程':'土木','光学工程':'光学','地理学':'地理','农业资源利用':'农资','生物学/生物科学与工程':'生物','兵器科学与技术':'兵器','矿业工程':'矿业','大气科学':'大气','基础医学/临床医学':'医学','电子科学与技术':'电子','测绘科学与技术':'测绘','控制科学与工程':'控制','军事学':'军事','中国语言文学':'语言','新闻传播学':'新闻','社会学':'社会','地球物理学':'地球','植物保护':'植物'}"
# bustm csl MAPPING="{0:'否',1:'是'}"
# eprstmt MAPPING="{'Negative':'差','Positive':'好'}"
# iflytek MAPPING="{0:'打车',100:'美颜',101:'影像',102:'摄影',103:'相机',104:'绘画',105:'二手',106:'电商',107:'团购',108:'外卖',109:'票务',10:'社区',110:'超市',111:'购物',112:'笔记',113:'办公',114:'日程',115:'女性',116:'经营',117:'收款',118:'其他',11:'赚钱',12:'魔幻',13:'仙侠',14:'卡牌',15:'飞行',16:'射击',17:'休闲',18:'动作',19:'体育',1:'地图',20:'棋牌',21:'养成',22:'策略',23:'竞技',24:'辅助',25:'约会',26:'通讯',27:'工作',28:'论坛',29:'婚恋',2:'免费',30:'情侣',31:'社交',32:'生活',33:'博客',34:'新闻',35:'漫画',36:'小说',37:'技术',38:'教辅',39:'问答',3:'租车',40:'搞笑',41:'杂志',42:'百科',43:'影视',44:'求职',45:'兼职',46:'视频',47:'短视',48:'音乐',49:'直播',4:'同城',50:'电台',51:'唱歌',52:'两性',53:'小学',54:'职考',55:'公务',56:'英语',57:'在线',58:'教育',59:'成人',5:'快递',60:'艺术',61:'语言',62:'旅游',63:'预定',64:'民航',65:'铁路',66:'酒店',67:'行程',68:'民宿',69:'出国',6:'婚庆',70:'工具',71:'亲子',72:'母婴',73:'驾校',74:'违章',75:'汽车',76:'买车',77:'养车',78:'行车',79:'租房',7:'家政',80:'买房',81:'装修',82:'电子',83:'挂号',84:'养生',85:'医疗',86:'减肥',87:'美妆',88:'菜谱',89:'餐饮',8:'交通',90:'资讯',91:'运动',92:'支付',93:'保险',94:'股票',95:'借贷',96:'理财',97:'彩票',98:'记账',99:'银行',9:'政务'}"
# tnews MAPPING="{100:'故事',101:'文化',102:'娱乐',103:'体育',104:'财经',106:'房产',107:'汽车',108:'教育',109:'科技',110:'军事',112:'旅游',113:'国际',114:'股票',115:'农业',116:'电竞'}"
# ocnli MAPPING="{'contradiction':'不','neutral':'或','entailment':'是'}"


for split in 1 2 3 4 "few_all"
do
    echo "Running experiment on split $split"
    
    TEMPLATE_PATH=my_auto_template/$split/$TASK/$K-$SEED.sort.txt
    TRIAL_IDTF=$RANDOM
    DATA_DIR=data/k-shot/$split/$TASK/$K-$SEED

    # FILTER_MODEL=distiluse-base-multilingual-cased-v1

    python run.py \
      --task_name $TASK \
      --data_dir $DATA_DIR \
      --overwrite_output_dir \
      --do_train \
      --do_eval \
      --do_predict \
      --evaluate_during_training \
      --model_name_or_path $MODEL \
      --few_shot_type $TYPE \
      --num_k $K \
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 16 \
      --learning_rate $LR \
      --max_steps $MAX_STEP \
      --eval_steps $EVAL_STEP \
      --output_dir result/$TASK-$TYPE-$split-$K-$SEED-$MODEL-$TRIAL_IDTF \
      --seed $SEED \
      --template_path $TEMPLATE_PATH \
      --template_id 0 \
      --mapping $MAPPING \
      $TASK_EXTRA
      # --max_seq_length $MAX_LEN \
      # --first_sent_limit $FIRST_LIMIT \
      # --double_demo \
      # --demo_filter \
      # --demo_filter_model sbert-$FILTER_MODEL
      # --logging_steps $EVAL_STEP \
done