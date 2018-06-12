#!/usr/bin/env bash
if [ ! -d "datasets" ]; then
  echo "Files not found. Downloading!"
  python download_cat_dog.py
  python make_dataset_folders.py --valid_prop 0.3 --train_set "datasets/train_64x64" --test_set "datasets/valid_64x64"
fi

python train.py --train_set "datasets/train/" --valid_set "datasets/valid/" --test_set "datasets/test/" --cuda
