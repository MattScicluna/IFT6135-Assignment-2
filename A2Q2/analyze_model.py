# stdlib imports
import argparse
import os

# thirdparty imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from torch.autograd import Variable
from skimage import io

# local imports
from model import imCNN2 as imgCNN


def get_prediction(index, savefig):
    decoder.eval()
    softmax = nn.Softmax()
    input = Variable(test_dataset[index][0].unsqueeze(0))
    true_label = test_dataset[index][1]
    if true_label == 0:
        true_label_word = 'Cat'
    else:
        true_label_word = 'Dog'

    output = softmax(decoder(input)).data[0]
    if output[0] > 0.5:
        prediction_word = 'Cat'
    else:
        prediction_word = 'Dog'

    score = round(output[np.argmax(output)], 2)
    txt = 'True Label: {0} \n Prediction: {1} \n Score: {2}'.format(true_label_word, prediction_word, score)

    path = test_dataset.imgs[index][0]
    image = io.imread(path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    if savefig:
        plt.savefig('result_from_pred')
    print(txt)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #  Pytorch ImageNet implementation
])

model_file = 'models/CModelTrained'
model = torch.load(model_file)
epoch_from = model['epoch']
state_dict = model['state_dict']
decoder = imgCNN()
decoder.load_state_dict(state_dict)

train_acc_vec = model['train_acc_vec']
valid_acc_vec = model['valid_acc_vec']
train_loss_vec = model['train_loss_vec']
valid_loss_vec = model['valid_loss_vec']

#  Get number of params
model_parameters = filter(lambda p: p.requires_grad, decoder.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('the model has: {} parameters'.format(params))

test_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), 'datasets/test/'),
                                    transform=data_transform)


print('done')

outcome_probs = []
decoder.eval()
for (img, label) in test_dataset:
    softmax = nn.Softmax()
    input = Variable(img.unsqueeze(0))
    output = softmax(decoder(input)).data[0]
    outcome_probs.append(output[label])
outcome_probs = np.array(outcome_probs)

#  Visualize worst results
worst_results = np.where(outcome_probs<0.2)[0]
get_prediction(worst_results[0], savefig=True)

#  Ambiguous results
ambiguous_results = np.where(np.equal((outcome_probs < 0.51), (outcome_probs > 0.49)))[0]
get_prediction(ambiguous_results[0], savefig=True)

print(outcome_probs)

# see each filter
# decoder.conv[0].weight.data

#index=0
#decoder.eval()
#softmax = nn.Softmax()
#input = Variable(test_dataset[index][0].unsqueeze(0))
#layer_viz = decoder.conv(input).data[0]
#plt.imshow(layer_viz[1].numpy())
