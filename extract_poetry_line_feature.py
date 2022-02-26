import numpy as np

from poetry_handler import read_raw_ccpc, to_lines
from pathlib import Path
from easydict import EasyDict
from utils.config import cfg_from_yaml_file
from infer import load_model, encode_texts
import torch
from tqdm import tqdm


if __name__ == '__main__':

    # all poetry data
    data = []
    test_data = read_raw_ccpc('data/poetry/CCPC/ccpc_test_v1.0.json')
    data.extend(test_data)
    train_data = read_raw_ccpc('data/poetry/CCPC/ccpc_train_v1.0.json')
    data.extend(train_data)
    valid_data = read_raw_ccpc('data/poetry/CCPC/ccpc_valid_v1.0.json')
    data.extend(valid_data)

    # lines = to_lines(data)
    lines = ['日照香炉生紫烟', '遥看瀑布挂前川', '飞流直下三千尺', '疑是银河落九天',
             '千呼万唤始出来', '犹抱琵琶半遮面', '转轴拨弦三两声', '未成曲调先有情',
             '天生丽质难自弃', '一朝选在君王侧', '回眸一笑百媚生', '六宫粉黛无颜色'
             ]

    # model config
    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / './').resolve()
    cfg_from_yaml_file('./cfg/test_xyb.yml', cfg)

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE.GPU)

    model = load_model(cfg, '../BriVL-pretrain-model/BriVL-1.0-5500w.pth')

    texts_emb = encode_texts(cfg, lines, model)

    line_emb_list = [[line, emb] for line, emb in tqdm(zip(lines, texts_emb))]
    np.save('logs/saves/poetry.npy', line_emb_list)


