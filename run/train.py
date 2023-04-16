from dataset import *
import argparse
from models import *
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import multilabel_auroc
from torchmetrics.classification import MultilabelPrecision
import collections
import warnings
# python3 /Users/yuxiaoliu/miniconda3/envs/si699-music-tagging/lib/python3.10/site-packages/tensorboard/main.py --logdir=runs
import logging
import json
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import csv
from transformers import AutoConfig, AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from sklearn.metrics import *
from models import *
import collections
import torch
import yaml
import json
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import multilabel_auroc
from torchmetrics.classification import MultilabelPrecision

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tag_file', type=str, default='data/autotagging_moodtheme.tsv')
parser.add_argument('--npy_root', type=str, default='data/waveform')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--model', type=str, default='samplecnn')
parser.add_argument('--transform', type=str, default='raw')
parser.add_argument('--is_map', type=bool, default=True)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on:", device)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
torch.manual_seed(config['seed'])


def train(model, epoch, criterion, optimizer, train_loader, is_title=False):
    losses = []
    ground_truth = []
    prediction = []
    model.train()
    for waveform, input_ids, attention_mask, label in tqdm(train_loader):
        waveform, label = waveform.to(device), label.to(device)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        if is_title:
            output = model(waveform, input_ids, attention_mask)
        else:
            output = model(waveform)
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach())
        ground_truth.append(label)
        prediction.append(output)
    get_eval_metrics(prediction, ground_truth, 'train', epoch, losses)


@torch.no_grad()
def validate(model, epoch, criterion, val_loader, is_title=False):
    losses = []
    ground_truth = []
    prediction = []
    model.eval()
    for waveform, input_ids, attention_mask, label in tqdm(val_loader):
        waveform, label = waveform.to(device), label.to(device)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        if is_title:
            output = model(waveform, input_ids, attention_mask)
        else:
            output = model(waveform)
        loss = criterion(output, label.float())
        losses.append(loss.cpu().detach())
        ground_truth.append(label)
        prediction.append(output)
    pre = get_eval_metrics(prediction, ground_truth, 'val', epoch, losses)
    return pre


def get_eval_metrics(outputs, labels, run_type, epoch, losses):
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    assert outputs.shape == labels.shape
    # 1. number of correctly predicted tags divided by the total number of tags
    prob_classes = []
    for i in range(labels.size(0)):
        label = labels[i]
        k = label.sum()
        _, idx = outputs[i].topk(k=k)
        predict = torch.zeros_like(outputs[i])
        predict[idx] = 1
        prob_classes.append(predict)
    prob_classes = torch.stack(prob_classes)
    matched_1s = torch.mul(prob_classes, labels)
    correct_tag_percentage = matched_1s.sum() / labels.sum()

    # 2. Auroc
    n_classes = 16
    if not args.is_map:
        n_classes = 59
    auroc = multilabel_auroc(outputs, labels, num_labels=n_classes, average="macro", thresholds=None).item()

    # 3. avg precision
    metric = MultilabelPrecision(average='macro', num_labels=n_classes, thresholds=None).to(device)
    pre = metric(outputs, labels).item()

    # write tensorboard and logging file
    writer.add_scalar("Loss/{}".format(run_type), np.mean(losses), epoch)
    writer.add_scalar("Auroc/{}".format(run_type), auroc, epoch)
    writer.add_scalar("Pre/{}".format(run_type), pre, epoch)
    writer.add_scalar("Avg_percent/{}".format(run_type), correct_tag_percentage, epoch)
    print("{} - epoch: {}, loss: {}, auroc: {}, pre: {}, avg percent: {}".format(
        run_type, epoch, np.mean(losses), auroc, pre, correct_tag_percentage))
    logging.info("{} - epoch: {}, loss: {}, auroc: {}, pre: {}, avg percent: {}".format(
        run_type, epoch, np.mean(losses), auroc, pre, correct_tag_percentage))
    return correct_tag_percentage


def get_model(tags):
    n_classes = len(tags)
    if args.model =='samplecnn':
        model = SampleCNN(n_classes, config).to(device)
    elif args.model == 'crnn':
        model = CRNN(n_classes, config).to(device)
    elif args.model =='fcn':
        model = FCN(n_classes, config).to(device)
    elif args.model == 'musicnn':
        model = Musicnn(n_classes, config).to(device)
    elif args.model == 'shortchunkcnn_res':
        model = ShortChunkCNN_Res(n_classes, config).to(device)
    elif args.model == 'cnnsa':
        model = CNNSA(n_classes, config).to(device)
    elif args.model == 'musicnn_title':
        model = MusicnnwithTitle(n_classes, config).to(device)
    elif args.model == 'baseline2':
        model = Baseline2(n_classes, config).to(device)
    elif args.model == 'wav2vec':
        model_config = AutoConfig.from_pretrained(
            "facebook/wav2vec2-base-960h",
            num_labels=n_classes,
            label2id={label: i for i, label in enumerate(tags)},
            id2label={i: label for i, label in enumerate(tags)},
            finetuning_task="wav2vec2_clf",
        )
        model = Wav2Vec2ForSpeechClassification(model_config).to(device)
    else:
        model = SampleCNN(n_classes, config).to(device)
    return model


def get_tags(tag_file, npy_root, isMap):
    id2title_dict = {}
    with open('data/raw.meta.tsv') as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)
        for row in reader:
            id2title_dict[row[0]] = row[3]

    if isMap:
        f = open('tag_categorize.json')
        data = json.load(f)
        categorize = {}
        for k, v in data.items():
            for i in v[1:-1].split(', '):
                categorize[i] = k
    tracks = {}
    total_tags = []
    with open(tag_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            if not os.path.exists(os.path.join(npy_root, row[3].replace('.mp3', '.npy'))):
                print(os.path.join(npy_root, row[3].replace('.mp3', '.npy')))
                continue
            track_id = row[3].split('.')[0]
            tags = []
            for tag in row[5:]:
                if isMap:
                    tags.append(categorize[tag.split('---')[-1]])
                else:
                    tags.append(tag.split('---')[-1])
            tracks[track_id] = (list(set(tags)), id2title_dict[row[0]])
            total_tags += list(set(tags))
    print("Distribution of tags:", collections.Counter(total_tags))
    plt.figure(figsize=(10,3))
    plt.xticks(rotation=90)
    plt.hist(total_tags)
    plt.savefig('dist.png')
    return tracks, list(set(total_tags))


def baseline1(val_loader, tags):
    whole_filenames = sorted(glob.glob(os.path.join(args.npy_root, "*/*.npy")))
    train_size = int(len(whole_filenames) * 0.8)
    random.shuffle(whole_filenames)
    train_filenames = whole_filenames[:train_size]
    train_ids = []
    for filename in train_filenames:
        train_ids.append(filename.split('/')[-2] + '/' + filename.split('/')[-1])
    if args.is_map:
        f = open('tag_categorize.json')
        data = json.load(f)
        categorize = {}
        for k, v in data.items():
            for i in v[1:-1].split(', '):
                categorize[i] = k
    train_total_tags = []
    with open(args.tag_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            if row[3].replace('.mp3', '.npy') not in train_ids:
                # if not in train set
                continue
            if not os.path.exists(os.path.join(args.npy_root, row[3].replace('.mp3', '.npy'))):
                print(os.path.join(args.npy_root, row[3].replace('.mp3', '.npy')))
                continue
            tmp = []
            for tag in row[5:]:
                if args.is_map:
                    tmp.append(categorize[tag.split('---')[-1]])
                else:
                    tmp.append(tag.split('---')[-1])
            train_total_tags += list(set(tmp))

    train_dist_tags = collections.Counter(train_total_tags)
    print(train_dist_tags)
    total = 0
    for v in train_dist_tags.values():
        total += v
    probs = []
    for t in tags:
        probs.append(train_dist_tags[t]/total)
    labels, outputs = [], []
    for _, _, _, label in val_loader:  
        labels.append(label)
        for _ in range(label.size(0)):
            outputs.append(probs)
    
    outputs = torch.Tensor(outputs)
    labels = torch.cat(labels, dim=0)
    assert outputs.shape == labels.shape, "{}, {}".format(outputs.shape, labels.shape)
    # 1. number of correctly predicted tags divided by the total number of tags
    prob_classes = []
    for i in range(labels.size(0)):
        label = labels[i]
        k = label.sum()
        _, idx = outputs[i].topk(k=k)
        predict = torch.zeros_like(outputs[i])
        predict[idx] = 1
        prob_classes.append(predict)
    prob_classes = torch.stack(prob_classes)
    matched_1s = torch.mul(prob_classes, labels)
    correct_tag_percentage = matched_1s.sum() / labels.sum()

    # 2. Auroc
    auroc = multilabel_auroc(outputs, labels, num_labels=len(tags), average="macro", thresholds=None).item()

    # 3. avg precision
    metric = MultilabelPrecision(average='macro', num_labels=len(tags), thresholds=None).to(device)
    pre = metric(outputs, labels).item()

    print("auroc: {}, pre: {}, avg percent: {}".format(auroc, pre, correct_tag_percentage))


if __name__ == '__main__':
    logging.basicConfig(filename="log/log_{}_{}_{}".format(args.model, args.learning_rate, args.batch_size),
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        filemode='w',
                        level=logging.INFO)
    logging.info("Preparing dataset...")
    tracks_dict, tags = get_tags(args.tag_file, args.npy_root, args.is_map)

    train_dataset = MyDataset(tracks_dict, args.npy_root, config, tags, "train", args.transform)
    val_dataset = MyDataset(tracks_dict, args.npy_root, config, tags, "valid", args.transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    model = get_model(tags)

    # Baseline 1
    baseline1(val_loader, tags)

    # Binary cross-entropy with logits loss combines a Sigmoid layer and the BCELoss in one single class.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Training and validating model...")
    writer = SummaryWriter('runs/{}_{}_{}'.format(args.model, args.learning_rate, args.batch_size))
    best_pre = float('-inf')
    for epoch in range(args.num_epochs):
        train(model, epoch, criterion, optimizer, train_loader)
        pre = validate(model, epoch, criterion, val_loader)
        if pre > best_pre:
            print("Best avg precision:", pre)
            best_pre = pre
            torch.save(model.state_dict(), 'model/{}_best_score_{}_{}.pt'.format(args.model, args.learning_rate, len(tags)))
    writer.close()
