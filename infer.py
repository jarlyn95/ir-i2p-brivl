import torch
from models import build_network
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils.loss import getLanMask
from pathlib import Path
from easydict import EasyDict
from utils.config import cfg_from_yaml_file
from transformers import AutoTokenizer
from torch import Tensor


def load_model(cfg, checkpoint_pth='../BriVL-pretrain-model/BriVL-1.0-5500w.pth'):
    model = build_network(cfg.MODEL)
    model_component = torch.load(checkpoint_pth, map_location=torch.device('cpu'))
    model.learnable.load_state_dict(model_component['learnable'])  ####### only save learnable
    model = torch.nn.DataParallel(model, device_ids=[cfg.DEVICE.GPU])
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def encode_images(cfg, images_pth, model):
    dataset = ImageDataset(images_pth, cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=cfg.DATASET.WORKERS,
        pin_memory=True,
        drop_last=False
    )
    with torch.no_grad():
        np_text, np_img = None, None
        for idx, batch in enumerate(tqdm(dataloader)):
            # data
            imgs = batch[0]
            img_lens = batch[1].view(-1)
            image_boxs = batch[2]
            imgMask = getLanMask(img_lens, cfg.MODEL.MAX_IMG_LEN)
            if torch.cuda.is_available():
                imgMask = imgMask.cuda()
                imgs = imgs.cuda()
                image_boxs = image_boxs.cuda()  # <BSZ, 36, 4>

            # text_lens = text_lens.cuda() ############
            img = model(imgs, None, imgMask, None, None, image_boxs, is_training=False, schema='image')

            if np_img is None:
                np_img = img.cpu().numpy()  # <bsz, featdim>

            else:
                np_img = np.concatenate((np_img, img.cpu().numpy()), axis=0)

    return np_img


def encode_texts(cfg, texts, model):
    dataset = TextDataset(texts, cfg=cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        # num_workers=cfg.DATASET.WORKERS,
        pin_memory=True,
        drop_last=False
    )
    with torch.no_grad():
        np_text = None
        for idx, batch in enumerate(tqdm(dataloader)):
            texts = batch[0]
            text_lens = batch[1]
            textMask = getLanMask(text_lens, max_len=cfg.MODEL.MAX_TEXT_LEN)
            if torch.cuda.is_available():
                textMask = textMask.cuda()
                texts = texts.cuda()
                text_lens = text_lens.cuda()
            text = model(None, texts, None, textMask, text_lens, None, is_training=False, schema='text')

            if np_text is None:
                np_text = text.cpu().numpy()
            else:
                np_text = np.concatenate((np_text, text.cpu().numpy()), axis=0)
    return np_text


class TextDataset(Dataset):
    def __init__(self, texts, cfg):
        self.texts = texts
        self.max_length = cfg.MODEL.MAX_TEXT_LEN
        self.text_transform = AutoTokenizer.from_pretrained(cfg.MODEL.ENCODER)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]

        text_info = self.text_transform(text, padding='max_length', truncation=True,
                                        max_length=self.max_length, return_tensors='pt')
        text = text_info.input_ids.reshape(-1)
        text_len = torch.sum(text_info.attention_mask)
        return text, text_len


class ImageDataset(Dataset):
    def __init__(self, images_pth, cfg):
        self.images_pth = images_pth
        self.new_size = cfg.MODEL.IMG_SIZE
        self.cfg = cfg
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        self.visual_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.new_size, self.new_size)),
            normalize])

    def __len__(self):
        return len(self.images_pth)

    def __getitem__(self, index):
        # new_size = self.cfg.MODEL.IMG_SIZE
        img_path = self.images_pth[index]
        image = Image.open(img_path).convert('RGB')
        # width, height = image.size

        img_box_s = [torch.from_numpy(np.array([0, 0, self.new_size, self.new_size]).astype(np.float32))]

        valid_len = len(img_box_s)
        img_len = torch.full((1,), valid_len, dtype=torch.long)
        if valid_len < self.cfg.MODEL.MAX_IMG_LEN:
            for i in range(self.cfg.MODEL.MAX_IMG_LEN - valid_len):
                img_box_s.append(torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)))
        image_boxs = torch.stack(img_box_s, 0)

        image = self.visual_transform(image)

        return image, img_len, image_boxs


def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


if __name__ == '__main__':

    cfg = EasyDict()
    cfg.ROOT_DIR = (Path(__file__).resolve().parent / './').resolve()
    cfg_from_yaml_file('./cfg/test_xyb.yml', cfg)

    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.DEVICE.GPU)

    model = load_model(cfg, '../BriVL-pretrain-model/BriVL-1.0-5500w.pth')

    texts = ['日照香炉生紫烟', '遥看瀑布挂前川', '飞流直下三千尺', '疑是银河落九天',
             '千呼万唤始出来', '犹抱琵琶半遮面', '转轴拨弦三两声', '未成曲调先有情',
             '天生丽质难自弃', '一朝选在君王侧', '回眸一笑百媚生', '六宫粉黛无颜色'
             ]

    texts_emb = encode_texts(cfg, texts, model)
    print(texts_emb.shape)

    images_pth = ['data/imgs/img.png',
                  'data/imgs/img_1.png',
                  'data/imgs/img_2.png']

    images_emb = encode_images(cfg, images_pth, model)
    print(images_emb.shape)

    img = torch.from_numpy(images_emb)
    text = torch.from_numpy(texts_emb)

    if torch.cuda.is_available():
        img = img.cuda()
        text = text.cuda()

    # cos_sim_scores = cos_sim(img, text)

    cos_sim_scores = torch.zeros((len(images_pth), len(text)), dtype=torch.float32)  # .cuda()
    print('Pair-to-pair: calculating scores')
    for i in tqdm(range(len(images_pth))):  # row: image  col: text
        cos_sim_scores[i, :] = torch.sum(img[i] * text, -1)
        # print(cos_sim_scores[i, :])

    for path, scores in tqdm(zip(images_pth, cos_sim_scores)):
        print("Path:", path)
        print('Scores:', scores)
        sort = torch.argsort(scores, descending=True)
        print("Sort:", sort)
        print("Sorted texts:", [texts[i] for i in sort], '\n')
