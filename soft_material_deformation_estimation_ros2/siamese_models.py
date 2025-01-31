import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import torch.nn.functional as F
from enum import Enum
from typing import Union
from VisionTransformer import ViT


class ModelType(Enum):
    regression = "regression"
    classification = "general"


class SiameseMultiHeadNetwork(nn.Module):
    def __init__(self,
                 features_extractor: nn.Module,
                 model_type: ModelType,
                 output_layers: Union[nn.ModuleList, nn.Sequential]
                 ):
        super(SiameseMultiHeadNetwork, self).__init__()
        self.features_extractor = features_extractor
        self.output_layers = output_layers
        self.model_type = model_type
        if self.model_type == ModelType.classification:
            self.forward = self.forward_classification
        elif self.model_type == ModelType.regression:
            self.forward = self.forward_regression

    def extract_features(self, x):
        x = self.features_extractor(x)
        return x

    def forward_classification(self, input1, input2):
        output1 = self.extract_features(input1)
        output2 = self.extract_features(input2)
        distance = output1 - output2
        return torch.stack([classifier(distance) for classifier in self.output_layers], dim=1)

    def forward_regression(self, input1, input2):
        output1 = self.extract_features(input1)
        output2 = self.extract_features(input2)
        distance = output1 - output2
        return self.output_layers(distance)

    @torch.no_grad()
    def predict(self, input1, input2):
        return F.softmax(self.forward(input1, input2), dim=-1)

    @torch.no_grad()
    def fast_predict_classification(self, const_feature_input, input2):
        output2 = self.extract_features(input2)
        distance = const_feature_input - output2
        return F.softmax(torch.stack([classifier(distance) for classifier in self.output_layers], dim=1), dim=-1)

class SiameseResNet(SiameseMultiHeadNetwork):
    def __init__(self, model_type: ModelType):
        features_extractor = models.resnet18()
        features_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=True)
        features_extractor.fc = nn.Identity()

        if model_type == ModelType.classification:
            classifiers = nn.ModuleList([nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 5)) for _ in range(4)])

            super(SiameseResNet, self).__init__(features_extractor=features_extractor,
                                                output_layers=classifiers,
                                                model_type=model_type)
        elif model_type == ModelType.regression:
            regression_layers = nn.Sequential(nn.Linear(512, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128),
                                              nn.ReLU(),
                                              nn.Linear(128, 4))

            super(SiameseResNet, self).__init__(features_extractor=features_extractor,
                                                output_layers=regression_layers,
                                                model_type=model_type)


class SiameseDenseNet(SiameseMultiHeadNetwork):
    def __init__(self, model_type: ModelType):
        features_extractor = models.densenet121()
        features_extractor.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if model_type == ModelType.classification:
            classifiers = nn.ModuleList([nn.Sequential(nn.Dropout(p=0.25),
                                                       nn.Linear(features_extractor.classifier.in_features, 256),
                                                       nn.ReLU(inplace=True),
                                                       nn.Dropout(p=0.25),
                                                       nn.Linear(256, 128),
                                                       nn.ReLU(inplace=True),
                                                       nn.Linear(128, 5))
                                         for _ in range(4)])
            features_extractor.classifier = nn.Identity()

            super(SiameseDenseNet, self).__init__(features_extractor=features_extractor,
                                                  output_layers=classifiers,
                                                  model_type=model_type)

        elif model_type == ModelType.regression:
            regression_layers = nn.Sequential(nn.Linear(features_extractor.classifier.in_features, 256),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(256, 128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 4))
            features_extractor.classifier = nn.Identity()

            super(SiameseDenseNet, self).__init__(features_extractor=features_extractor,
                                                  output_layers=regression_layers,
                                                  model_type=model_type)


class SiameseVGG(SiameseMultiHeadNetwork):
    def __init__(self, model_type: ModelType):
        features_extractor = models.vgg11()
        original_first_layer = features_extractor.features[0]
        features_extractor.features[0] = nn.Conv2d(in_channels=1,
                                                   out_channels=original_first_layer.out_channels,
                                                   kernel_size=original_first_layer.kernel_size,
                                                   stride=original_first_layer.stride,
                                                   padding=original_first_layer.padding)

        if model_type == ModelType.classification:
            in_features = features_extractor.classifier[0].in_features

            # Replace the classifier
            features_extractor.classifier = nn.Identity()

            classifiers = nn.ModuleList([nn.Sequential(nn.Linear(in_features, 128),
                                                       nn.ReLU(inplace=True),
                                                       nn.Dropout(),
                                                       nn.Linear(128, 5)) for _ in range(4)])

            super(SiameseVGG, self).__init__(features_extractor=features_extractor,
                                             output_layers=classifiers,
                                             model_type=model_type)
        elif model_type == ModelType.regression:
            regression_layers = nn.Sequential(nn.Linear(512, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128),
                                              nn.ReLU(),
                                              nn.Linear(128, 4))

            super(SiameseResNet, self).__init__(features_extractor=features_extractor,
                                                output_layers=regression_layers,
                                                model_type=model_type)


class SiameseViT(SiameseMultiHeadNetwork):
    def __init__(self,
                 model_type: ModelType,
                 n_patches: int,
                 n_blocks: int,
                 hidden_dimension: int,
                 n_heads: int,
                 ):
        features_extractor = ViT(chw=(1, 224, 224),
                                 n_patches=n_patches,
                                 n_blocks=n_blocks,
                                 hidden_dimension=hidden_dimension,
                                 n_heads=n_heads,)

        if model_type == ModelType.classification:
            out_layers = nn.ModuleList([nn.Sequential(nn.Dropout(p=0.25),
                                                                 nn.Linear(features_extractor.hidden_dimension, 256),
                                                                 nn.ReLU(inplace=True),
                                                                 nn.Dropout(p=0.25),
                                                                 nn.Linear(256, 128),
                                                                 nn.ReLU(inplace=True),
                                                                 nn.Linear(128, 5))
                                                   for _ in range(4)])

        elif model_type == ModelType.regression:
            out_layers = nn.Sequential(nn.Dropout(p=0.25),
                                              nn.Linear(features_extractor.hidden_dimension, 256),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=0.25),
                                              nn.Linear(256, 128),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(128, 4))

        super(SiameseViT, self).__init__(model_type=model_type,
                                         features_extractor=features_extractor,
                                         output_layers=out_layers)
