#!/usr/bin/env bash
if [ ! -d "datasets" ]; then
  echo "Files not found. Downloading!"
  python download_cat_dog.py
fi

python train.py --train_set "datasets/train_64x64/" --valid_set "datasets/valid_64x64/"
