import shutil
import os
import argparse


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_set', type=str, required=True)
    argparser.add_argument('--test_set', type=str, required=True)
    argparser.add_argument('--valid_prop', type=float, default=0.3)
    args = argparser.parse_args()

    # ensure path location exists
    cat_train_loc = 'datasets/train/cats/'
    dog_train_loc = 'datasets/train/dogs/'
    cat_valid_loc = 'datasets/valid/cats/'
    dog_valid_loc = 'datasets/valid/dogs/'
    cat_test_loc = 'datasets/test/cats/'
    dog_test_loc = 'datasets/test/dogs/'

    try:
        os.makedirs(cat_train_loc)
        os.makedirs(dog_train_loc)
        source = args.train_set
        files = os.listdir(source)
        for file in files:
            if file.split('.')[0] == 'Cat':
                shutil.move(os.path.join(source, file), cat_train_loc)
            else:
                shutil.move(os.path.join(source, file), dog_train_loc)
    except:
        pass
    try:
        os.makedirs(cat_valid_loc)
        os.makedirs(dog_valid_loc)
        source = cat_train_loc
        files = os.listdir(cat_train_loc)
        transfer_until = int(args.valid_prop * len(files))
        for ind, file in enumerate(files):
            shutil.move(os.path.join(source, file), cat_valid_loc)
            if ind > transfer_until:
                break

        source = dog_train_loc
        files = os.listdir(source)
        transfer_until = int(args.valid_prop * len(files))
        for ind, file in enumerate(files):
            shutil.move(os.path.join(source, file), dog_valid_loc)
            if ind > transfer_until:
                break
    except:
        pass
    try:
        os.makedirs(cat_test_loc)
        os.makedirs(dog_test_loc)
        source = args.test_set
        files = os.listdir(source)
        for file in files:
            if file.split('.')[0] == 'Cat':
                shutil.move(os.path.join(source, file), cat_test_loc)
            else:
                shutil.move(os.path.join(source, file), dog_test_loc)
    except:
        pass

if __name__ == '__main__':
    main()
