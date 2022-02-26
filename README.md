# 基于BriVL模型实现图片检索绝句诗
## 1. [预训练模型](https://wudaoai.cn/model/detail/BriVL#download)
下载预模型需要提交在线申请  
下载的预训练模型建议存放为`../BriVL-pretrain-model/BriVL-1.0-5500w.pth`
## 2. 本实现参考了代码[BriVlL推理源码](https://github.com/BAAI-WuDao/BriVlL)
## 3. 绝句诗来源于[CCPC](http://github.com/THUNLP-AIPoet/Datasets/tree/master/CCPC)
已放在data/poetry/CCPC路径下
## 4. 应用
```
# a. 安装依赖
pip install -r requirements.txt
# b. 提取诗句特征
python extract_poetry_line_feature.py
# c. 图像检索诗句示例
python image_retrieve_peotry_demo.py
```

## 5. 说明
 a. 可以把可工程修改为web/微信小程序  
 b. 如果有图像库，亦可修改为使用诗歌检索图像  
 c. CCPC绝句诗总共510728句，可把诗句范围缩小至诗歌经典诗句/诗篇
