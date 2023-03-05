import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torchvision

from sklearn.neighbors import NearestNeighbors


class BackendMobileNetv3:

    def __init__(self):
        self.dataset = pd.read_csv("Book/dataset/main_dataset.csv")

        torch.cuda.empty_cache()
        self.model = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.eval()

        self.imageTransform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])

        self.features = pickle.load(open('book_cover/static/features-bookcovers-mobilenetv3large1.pickle', 'rb'))
        self.filenames = pickle.load(open('book_cover/static/filenames-book_covers-mobilenetv3large1.pickle', 'rb'))

        self.neighbors = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean').fit(self.features)

    def query(self, queryFilename):
        query_features = self.model(self.imageTransform(
            Image.open(queryFilename).convert('RGB')
        ).unsqueeze(0)).view(-1).detach().numpy()

        distances, indices = self.neighbors.kneighbors([query_features])
        # results = list(set([self.filenames[indices[0][i]].split('/')[-1].split('.')[0] for i in range(len(indices[0]))]))
        results = [self.filenames[indices[0][i]].split('/')[-1] for i in range(len(indices[0]))]

        # deduplicating results
        deduplicated_results = {}

        columns = ['title', 'primary_isbn13']
        for res in results:
            isbn = res.split('.')[0]
            deduplicated_results[res] = self.dataset.loc[self.dataset['primary_isbn13'] == isbn][columns].iloc[
                0].values.tolist()

        titles, title_isbn_filepath = [], []
        t = 0
        for k, v in deduplicated_results.items():
            if v[0] in titles:
                continue
            else:
                titles.append(v[0])
                title_isbn_filepath.append([v[0], v[1], k])

        # print(title_isbn_filepath)
        return title_isbn_filepath, query_features


class MobileNetRewired(torch.nn.Module):
    def __init__(self):
        super(MobileNetRewired, self).__init__()
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mobilent = torchvision.models.mobilenet_v3_large(pretrained=True)
        self.features_conv = self.mobilent.features

        self.ad_average = torch.nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = self.mobilent.classifier
        self.gradient = None

    def activation_hook(self, grad):
        self.gradient = grad

    def forward(self, x):
        x = self.features_conv(x)
        h = x.register_hook(self.activation_hook)

        x = self.ad_average(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    def get_activation_grad(self):
        return self.gradient

    def get_activation(self, x):
        return self.features_conv(x)


def get_heatmap(filename, model=MobileNetRewired()):
    model.eval()
    image_tensor = model.preprocess(Image.open(filename).convert('RGB')).unsqueeze(0)
    pred = model(image_tensor)
    pred[:, pred.argmax(dim=1)].backward()
    gradients = model.get_activation_grad()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = model.get_activation(image_tensor).detach()

    for i in range(960):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    img = cv2.imread(filename)
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.5 + img * 0.5

    heatmap_file = filename.split('.')[0] + '_heatmap.jpg'
    cv2.imwrite(heatmap_file, superimposed_img)

    return heatmap_file


if __name__ == '__main__':
    # b = BackendDelf()
    # b2 = BackendResnet()
    b3 = BackendMobileNetv3()
    b3.query('Book/uploads/9780307346612.jpg')