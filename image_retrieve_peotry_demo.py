import numpy as np

from poetry_handler import read_raw_ccpc, get_poetry_by_line
from pathlib import Path
from easydict import EasyDict
from utils.config import cfg_from_yaml_file
from infer import load_model, encode_images
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

    line_emb_list = np.load('./logs/saves/poetry.npy', allow_pickle=True)
    lines = line_emb_list[:, 0]
    texts_emb = line_emb_list[:, 1]
    texts_emb = np.array(texts_emb.tolist(), dtype=np.float64)

    # input searching image
    images_pth = ['data/imgs/img.png',
                  'data/imgs/img_1.png',
                  'data/imgs/img_2.png']

    # model config
    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / './').resolve()
    cfg_from_yaml_file('./cfg/test_xyb.yml', cfg)

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE.GPU)

    model = load_model(cfg, '../BriVL-pretrain-model/BriVL-1.0-5500w.pth')
    images_emb = encode_images(cfg, images_pth, model)

    img = torch.from_numpy(images_emb)
    text = torch.from_numpy(texts_emb)

    if torch.cuda.is_available():
        img = img.cuda()
        text = text.cuda()

    cos_sim_scores = torch.zeros((len(images_pth), len(text)), dtype=torch.float32)  # .cuda()
    if torch.cuda.is_available():
        cos_sim_scores = cos_sim_scores.cuda()

    print('Pair-to-pair: calculating scores')
    for i in tqdm(range(len(images_pth))):  # row: image  col: text
        cos_sim_scores[i, :] = torch.sum(img[i] * text, -1)

    for path, scores in tqdm(zip(images_pth, cos_sim_scores)):
        max_img_idx = torch.argmax(scores)
        print("Path:", path)
        print("Score:", scores[max_img_idx])
        print("Line:", lines[max_img_idx])
        print("poetry:", get_poetry_by_line(lines[max_img_idx], data), "\n")

