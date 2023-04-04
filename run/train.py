from dataset import *
import argparse
from models import *
import yaml
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.classification import multilabel_auroc
from torchmetrics.classification import MultilabelPrecision
import collections
import warnings
warnings.filterwarnings('ignore', message='No positive class found in y_true') # positive class is rare in y_true
# python3 /Users/yuxiaoliu/miniconda3/envs/si699-music-tagging/lib/python3.10/site-packages/tensorboard/main.py --logdir=runs
import logging

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tag_file', type=str, default='data/autotagging_moodtheme.tsv')
parser.add_argument('--npy_root', type=str, default='data/theme_npy')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--model', type=str, default='samplecnn')
parser.add_argument('--transform', type=str, default='melspec')
parser.add_argument('--threshold', type=float, default=0.5)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on:", device)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
torch.manual_seed(config['seed'])


def train(model, epoch, criterion, optimizer, train_loader):
    losses = []
    ground_truth = []
    prediction = []
    model.train()
    for input, label in tqdm(train_loader):
        input, label = input.to(device), label.to(device)
        output = model(input)
        # print("label:", label.shape, "output", output.shape)
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        ground_truth.append(label)
        prediction.append(output)
    get_eval_metrics(prediction, ground_truth, 'train', epoch, losses)
    return model


@torch.no_grad()
def validate(model, epoch, learning_rate, criterion, val_loader, best_pre):
    losses = []
    ground_truth = []
    prediction = []
    model.eval()
    for input, label in tqdm(val_loader):
        input, label = input.to(device), label.to(device)
        # input = input.unsqueeze(1)
        output = model(input)
        loss = criterion(output, label.float())
        losses.append(loss.detach())
        ground_truth.append(label)
        prediction.append(output)
    pre = get_eval_metrics(prediction, ground_truth, 'val', epoch, losses)
    if pre > best_pre:
        print("Best precision:", pre)
        best_pre = pre
        torch.save(model, 'model/{}_best_score_{}.pt'.format(args.model, learning_rate))
    return best_pre


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
    auroc = multilabel_auroc(outputs, labels, num_labels=59, average="macro", thresholds=None).item()

    # 3. avg precision
    precision = MultilabelPrecision(average='macro', num_labels=59, thresholds=None)
    pre = precision(outputs, labels).item()

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
    n_labels = len(tags)
    print("Number of classes to predict:", n_labels)
    if args.model =='samplecnn':
        model = SampleCNN(n_labels, config).to(device)
    elif args.model == 'crnn':
        model = CRNN(n_labels, config).to(device)
    elif args.model =='fcn':
        model = FCN(n_labels, config).to(device)
    elif args.model == 'musicnn':
        model = Musicnn(n_labels, config).to(device)
    elif args.model == 'shortchunkcnn_res':
        model = ShortChunkCNN_Res(n_labels, config).to(device)
    elif args.model == 'cnnsa':
        model = CNNSA(n_labels, config).to(device)
    elif args.model == 'ast':
        model = ASTClassifier(n_labels, config).to(device)
    elif args.model == 'wav2vec':
        model_config = AutoConfig.from_pretrained(
            "facebook/wav2vec2-base-960h",
            num_labels=n_labels,
            label2id={label: i for i, label in enumerate(tags)},
            id2label={i: label for i, label in enumerate(tags)},
            finetuning_task="wav2vec2_clf",
        )
        model = Wav2Vec2ForSpeechClassification(model_config).to(device)
    else:
        model = SampleCNN(n_labels, config).to(device)
    return model


def get_tags(tag_file):
    collected_tags = []
    with open(tag_file, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            result = row[0].split('\t')
            for splitted in result:
                if 'mood/theme---' in splitted:
                    collected_tags.append(splitted.split('---')[-1])
    collected_tags = collections.Counter(collected_tags)
    return list(collected_tags.keys())


if __name__ == '__main__':
    logging.basicConfig(filename="log/log_{}_{}_{}".format(args.model, args.learning_rate, args.batch_size),
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        filemode='w',
                        level=logging.INFO)
    logging.info("Preparing dataset...")
    tags = get_tags(args.tag_file)

    train_dataset = MyDataset(args.tag_file, args.npy_root, config, tags, "train", args.transform)
    val_dataset = MyDataset(args.tag_file, args.npy_root, config, tags, "valid", args.transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    model = get_model(tags)
    # Binary cross-entropy with logits loss combines a Sigmoid layer and the BCELoss in one single class.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Training and validating model...")
    writer = SummaryWriter('runs/{}_{}_{}'.format(args.model, args.learning_rate, args.batch_size))
    best_pre = float('-inf')
    for epoch in range(args.num_epochs):
        train(model, epoch, criterion, optimizer, train_loader)
        best_pre = validate(model, epoch, args.learning_rate, criterion, val_loader, best_pre)

    # torch.save(model, 'model/{}.pt'.format(args.model))
    writer.close()
