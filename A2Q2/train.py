# https://github.com/spro/char-rnn.pytorch

# stdlib imports
import argparse

# thirdparty imports
from torch.utils.data import DataLoader

# local imports
from dataset import imgDataset

def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, required=True)
    argparser.add_argument('--valid_set', type=str, required=True)
    argparser.add_argument('--batch_size', type=int, default=300)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--cpu', action='store_true')
    args = argparser.parse_args()

    train_dataset = imgDataset(args.train_set)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)

    valid_dataset = imgDataset(args.valid_set)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)

if __name__ == '__main__':
    main()