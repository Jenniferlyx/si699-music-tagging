from dataset import *
import argparse
from models import *
import yaml
from torch.utils.tensorboard import SummaryWriter
# python3 /Users/yuxiaoliu/miniconda3/envs/si699-music-tagging/lib/python3.10/site-packages/tensorboard/main.py --logdir=runs
import matplotlib.pyplot as plt
import logging
import multiprocessing
logging.basicConfig(filename="log",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    filemode='w',
                    level=logging.INFO)
sem = multiprocessing.Semaphore(1)
sem.release()

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tag_file', type=str, default='data/autotagging_top50tags.tsv')
parser.add_argument('--npy_root', type=str, default='data/npy')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=20)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()
print("Run on:", device)


def train(model, epoch, criterion, optimizer, train_loader):
    accs, losses = [], []
    model.train()
    for _, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        # input = input.unsqueeze(1)
        output = model(input)
        # print("label:", label.shape, "output", output.shape)
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
        acc = accuracy(output, label)
        accs.append(acc)
    logging.info("Train - epoch: {}, loss: {}, acc: {}".format(epoch, sum(losses) / len(losses), sum(accs) / len(accs)))
    writer.add_scalar("Loss/train", sum(losses) / len(losses), epoch)
    writer.add_scalar("Acc/train", sum(accs) / len(accs))
    return model


@torch.no_grad()
def validate(model, epoch, criterion, val_loader):
    losses, accs = [], []
    model.eval()
    for _, (input, label) in enumerate(val_loader):
        input, label = input.to(device), label.to(device)
        # input = input.unsqueeze(1)
        output = model(input)

        loss = criterion(output, label.float())
        losses.append(loss)
        acc = accuracy(output, label)
        accs.append(acc)
    logging.info("Validate - epoch: {}, loss: {}, acc: {}".format(epoch, sum(losses) / len(losses), sum(accs) / len(accs)))
    writer.add_scalar("Loss/val", sum(losses) / len(losses), epoch)
    writer.add_scalar("Acc/val", sum(accs) / len(accs))


def accuracy(output, labels):
    plt.hist(output.detach().numpy())
    plt.savefig('dist.png')
    plt.close()
    classes = output
    classes[classes > 0.5] = 1
    classes[classes <= 0.5] = 0
    return (classes == labels).sum() / len(classes.reshape(-1))


def save_to_onnx(model):
    dummy_input = torch.randn(1, 96, 960000)
    torch.onnx.export(model,
                      dummy_input,
                      "model/crnn.onnx",
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11  # the ONNX version to export the model to
                      )


if __name__ == '__main__':
    with open('run/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Preparing dataset...")
    train_dataset = MyDataset(args.tag_file, args.npy_root, config, "train")
    val_dataset = MyDataset(args.tag_file, args.npy_root, config, "valid")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    n_classes = 50
    model = Musicnn(n_classes, config).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Training and validating model...")
    for epoch in tqdm(range(args.num_epochs)):
        train(model, epoch, criterion, optimizer, train_loader)
        validate(model, epoch, criterion, val_loader)

    save_to_onnx(model)
    writer.close()
