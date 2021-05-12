#-*- coding:utf-8 -*-

tnews_labels ={'news_tech':'科技','news_entertainment':'娱乐','news_car':'汽车','news_travel':'旅游','news_finance':'财经',
              'news_edu':'教育','news_world':'国际','news_house':'房产','news_game':'电竞','news_military':'军事',
              'news_story':'故事','news_culture':'文化','news_sports':'体育','news_agriculture':'农业', 'news_stock':'股票'}
tnews_label_descriptions= {key:"这是一条"+value+"新闻" for key,value in tnews_labels.items()}

eprstmt_labels ={'Negative':'负面','Positive':'正面'}
eprstmt_label_descriptions= {key:"这表达了"+value+"的情感" for key,value in eprstmt_labels.items()}

