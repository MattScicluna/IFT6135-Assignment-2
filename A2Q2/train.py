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
from model import imCNN2 as imgCNN
from lr_scheduler import ReduceLROnPlateau
#from torch.optim import lr_scheduler

def main():
    # Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, required=True)
    argparser.add_argument('--valid_set', type=str, required=True)
    argparser.add_argument('--test_set', type=str, required=True)
    argparser.add_argument('--learning_rate', type=float, default=0.01)
    argparser.add_argument('--n_epochs', type=int, default=50)
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--weight_decay', type=int, default=5e-4)
    argparser.add_argument('--momentum', type=int, default=0.9)
    argparser.add_argument('--num_workers', type=int, default=8)
    argparser.add_argument('--optim_type', type=str, default='SGD')
    argparser.add_argument('--model_file', type=str, default='None')
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #  Pytorch ImageNet implementation
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), args.train_set),
                                         transform=data_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  drop_last=True)
    valid_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), args.valid_set),
                                         transform=data_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  drop_last=True)
    test_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), args.test_set),
                                        transform=data_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 drop_last=True)

    train_acc_vec, valid_acc_vec, train_loss_vec, valid_loss_vec = [], [], [], []

    if args.model_file == 'None':
        epoch_from = 1
        decoder = imgCNN()

        # Optimizer
        if args.optim_type == 'SGD':
            optimizer = torch.optim.SGD(decoder.parameters(), lr=args.learning_rate,
                                        momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    else:
        model = torch.load(args.model_file)
        epoch_from = model['epoch']
        state_dict = model['state_dict']
        decoder = imgCNN()
        decoder.load_state_dict(state_dict)
        if args.optim_type == 'SGD':
            optimizer = torch.optim.SGD(decoder.parameters(), lr=args.learning_rate,
                                        momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_state = model['optimizer']
        optimizer.load_state_dict(optimizer_state)

        train_acc_vec = model['train_acc_vec']
        valid_acc_vec = model['valid_acc_vec']
        train_loss_vec = model['train_loss_vec']
        valid_loss_vec = model['valid_loss_vec']

        print('model successfully loaded! Resuming training...')

    if args.cuda:
        print('running with GPU...')
        decoder.cuda()

    #  Scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr=1e-6)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Loss
    criterion = nn.CrossEntropyLoss()

    try:
        print('Training for maximum {} epochs...'.format(args.n_epochs))
        for epoch in range(epoch_from, args.n_epochs + 1):

            num_samples_t, training_loss, train_acc = 0, 0, 0
            decoder.train()
            for (input, target) in train_dataloader:
                input = Variable(input)
                target = Variable(target)
                if args.cuda:
                    input = input.cuda()
                    target = target.cuda()

                # Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = decoder(input)
                output_vals = outputs.max(1)[1].type_as(target)
                train_acc += output_vals.eq(target).sum().data[0]
                loss = criterion(outputs, target)
                training_loss += loss.data[0]
                loss.backward()
                optimizer.step()
                num_samples_t += args.batch_size

            num_samples_v, valid_acc, valid_loss = 0, 0, 0
            decoder.eval()
            for (input, target) in valid_dataloader:
                input = Variable(input)
                target = Variable(target)
                if args.cuda:
                    input = input.cuda()
                    target = target.cuda()

                outputs = decoder(input)
                loss = criterion(outputs, target)
                valid_loss += loss.data[0]
                output_vals = outputs.max(1)[1].type_as(target)
                valid_acc += output_vals.eq(target).sum().data[0]
                num_samples_v += args.batch_size

            scheduler.step(valid_loss/num_samples_v)
            #scheduler.step(epoch)

            pcnt = epoch / args.n_epochs * 100

            #  Display current results
            log = ('epoch #{} ({:.1f}%) - training loss {:.4f} - training acc: {:.2f}% - valid acc {:.2f}%')
            print(log.format(epoch,
                             pcnt,
                             training_loss/num_samples_t,
                             train_acc/num_samples_t * 100,
                             valid_acc/num_samples_v * 100))

            #  Save results
            train_acc_vec.append(train_acc/num_samples_t * 100)
            train_loss_vec.append(training_loss/num_samples_t)
            valid_acc_vec.append(valid_acc/num_samples_v * 100)
            valid_loss_vec.append(valid_loss/num_samples_v)

            # Stop when there is overfitting
            if epoch > 50:
                if (valid_loss_vec[-2]-valid_loss_vec[-1] > 0):
                    valid_acc_vec.pop()
                    train_acc_vec.pop()
                    print('overfitting! stopping...')
                    break

            state = {
                'epoch': epoch,
                'state_dict': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_acc_vec': train_acc_vec,
                'valid_acc_vec': valid_acc_vec,
                'train_loss_vec': train_loss_vec,
                'valid_loss_vec': valid_loss_vec
            }
            torch.save(state, 'models/most-recent-model')

    except KeyboardInterrupt:
        print("Saving before quit...")
        state = {
            'epoch': epoch,
            'state_dict': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_acc_vec': train_acc_vec,
            'valid_acc_vec': valid_acc_vec,
            'train_loss_vec': train_loss_vec,
            'valid_loss_vec': valid_loss_vec
        }
        torch.save(state, 'models/epoch-{0}-trainloss-{1}'.format(epoch, train_acc_vec[-1]))

    # Evaluate test set
    decoder.eval()
    test_acc, test_loss, num_samples_test = 0, 0, 0
    for (input, target) in test_dataloader:
        input = Variable(input)
        target = Variable(target)
        if args.cuda:
            input = input.cuda()
            target = target.cuda()

        outputs = decoder(input)
        loss = criterion(outputs, target)
        test_loss += loss.data[0]
        output_vals = outputs.max(1)[1].type_as(target)
        test_acc += output_vals.eq(target).sum().data[0]
        num_samples_test += args.batch_size
    log = ('test loss {:.4f} - test acc: {:.2f}%')
    print(log.format(test_loss / num_samples_test, test_acc / num_samples_test * 100))

    plt.figure()
    plt.title('Accuracy per Epoch')
    plt.plot(train_acc_vec, color='red', label='training set')
    plt.plot(valid_acc_vec, color='blue', label='validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.savefig('figs/acc')
    plt.axis([0, 100, 0, 100])
    # plt.show()

    plt.figure()
    plt.title('Loss per Epoch')
    plt.plot(train_loss_vec, color='red', label='training set')
    plt.plot(valid_loss_vec, color='blue', label='validation set')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('figs/loss')
    plt.axis([0, 100, 0, 100])
    #plt.show()

if __name__ == '__main__':
    main()
