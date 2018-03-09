# https://github.com/spro/char-rnn.pytorch

# stdlib imports
import argparse
import os

# thirdparty imports
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import torch.nn as nn

# local imports
from model import imgCNN


def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, required=True)
    argparser.add_argument('--valid_set', type=str, required=True)
    argparser.add_argument('--test_set', type=str, required=True)
    argparser.add_argument('--learning_rate', type=float, default=0.0001)
    argparser.add_argument('--n_epochs', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=300)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--model_file', type=str, default='None')
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), args.train_set),
                                         transform=transforms.ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)
    valid_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), args.valid_set),
                                         transform=transforms.ToTensor())
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  drop_last=True)
    test_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), args.test_set),
                                        transform=transforms.ToTensor())
    test_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 drop_last=True)

    if args.model_file == 'None':
        epoch_from = 1
        decoder = imgCNN()
    else:
        epoch_from = 1
        decoder = imgCNN()

    if args.cuda:
        print('running with GPU...')
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)

    try:
        print('Training for maximum {} epochs...'.format(args.n_epochs))
        for epoch in range(epoch_from, args.n_epochs + 1):
            num_samples_t, training_loss, train_acc = 0, 0, 0
            for (input, target) in train_dataloader:
                input = Variable(input)
                target = Variable(target)
                if args.cuda:
                    input = input.cuda()
                    target = target.cuda()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = decoder(input)
                train_acc += sum(outputs.topk(1)[1].view(-1) == target).data[0]
                loss = criterion(outputs, target)
                training_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                num_samples_t += args.batch_size

            num_samples_v, valid_acc = 0, 0
            for (input, target) in valid_dataloader:
                input = Variable(input)
                target = Variable(target)
                if args.cuda:
                    input = input.cuda()
                    target = target.cuda()

                outputs = decoder(input)
                valid_acc += sum(outputs.topk(1)[1].view(-1) == target).data[0]
                num_samples_v += args.batch_size

            pcnt = epoch / args.n_epochs * 100
            log = ('epoch #{} ({:.1f}%) - training loss {:.2f} - training acc: {:.2f}% - valid acc {:.2f}%')
            print(log.format(epoch, pcnt, training_loss, train_acc/num_samples_t * 100, valid_acc/num_samples_v * 100))

    except KeyboardInterrupt:
        print("Saving before quit...")


if __name__ == '__main__':
    main()
