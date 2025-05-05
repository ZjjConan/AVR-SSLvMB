import torch
import torch.nn as nn

from .blocks import ConvConceptBlock, Classifier, ConvBNAct
from .resnet4b import resnet4b
from .resnet import resnet18

class SSLvMB(nn.Module):
# Self-Supervised Learning via Mental Bootstrapping

    def __init__(self, model="resnet4b", per_choice_loc_type="fix", args=None):
        super().__init__()

        # frame extractor
        # ---------------
        self.in_channels = args.in_channels
        self.ou_channels = args.ou_channels
        self.per_choice_loc_type = per_choice_loc_type

        self.unsupervised_training = args.unsupervised_training
        self.row2col = torch.tensor([0, 3, 6, 1, 4, 7, 2, 5, 8])
        self.rmv_3fm = torch.tensor([3, 4, 5, 6, 7, 8, 0, 1])
        self.rmv_6fm = torch.tensor([0, 1, 2, 6, 7, 8, 3, 4])
        
        if model == "resnet4b":
            self.feature_extractor = resnet4b(args)
            conv_in_channels = self.feature_extractor.ou_channels
        
        elif model == "resnet18":
            self.feature_extractor = resnet18(dropout_rate=args.block_drop)
            conv_in_channels = self.feature_extractor.ou_channels
    
            # modified for RPM
            self.feature_extractor.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False)
            self.feature_extractor.layer4.conv1.stride = 1
            self.feature_extractor = nn.Sequential(
                self.feature_extractor.conv1,
                self.feature_extractor.bn1,
                self.feature_extractor.relu,
                self.feature_extractor.maxpool,
                self.feature_extractor.layer1,
                self.feature_extractor.layer2,
                self.feature_extractor.layer3,
                self.feature_extractor.layer4
            )

        conv_re_channels = 64
        conv_md_channels = 64

        self.feature_reducer = ConvBNAct(
            conv_in_channels, conv_re_channels, 
            conv_dim=2, kernel_size=1, padding=0, 
            stride=1, bias=False, use_act=False
        )

        # convolutional rule extractor
        # ---------------
        conv_in_channels = conv_re_channels
        self.concept_extractor = []
        for i in range(args.num_extra_stages):

            downsample = ConvBNAct(
                conv_in_channels, conv_in_channels, 
                conv_dim=2, kernel_size=1, padding=0, 
                stride=1, bias=False, use_act=False
            )
            
            self.concept_extractor.append(
                ConvConceptBlock(
                    conv_in_channels, 
                    conv_in_channels,
                    md_channels=conv_md_channels,
                    downsample=downsample, 
                    dropout=args.block_drop,
                    permute=True if i == 0 else False
                )
            )

        self.concept_extractor = nn.Sequential(*self.concept_extractor)

        # predictor
        self.featr_dims = 1024
        self.featr_pool = torch.nn.AdaptiveAvgPool1d(self.featr_dims)

        self.predictor = Classifier(
            self.featr_dims, args.ou_channels, 
            norm_layer = nn.BatchNorm1d, 
            dropout = args.classifier_drop, 
            classifier_hidreduce = args.classifier_hidreduce
        )


    def _replace_choices(self, ctx_features, cho_features, labels):

        batches, frames = ctx_features.shape[0:2]
        features = ctx_features.clone().unsqueeze(1).repeat(1,10,1,1,1)
        
        # (concept generation) cross-replacement
        if self.per_choice_loc_type == "fix":
            replace_index = torch.randint(0, frames, (1,))
            features[:, 1:9, replace_index] = cho_features.unsqueeze(2)
        
        elif self.per_choice_loc_type == "random":
            replace_index = torch.randint(0, frames, (1, batches), dtype=torch.long)
            replace_index = replace_index.tolist()[0]
            features[range(batches), 1:9, replace_index] = cho_features

        # self-context concept reuse
        perms = torch.multinomial(
            torch.ones((1,frames)), frames, replacement=True
        )
        features[:,9] = features[:,9,perms[0]]
        
        return features

    def _forward_remove_iframe(self, row_features, col_features, index):

        row_features = row_features[:, index]
        col_features = col_features[:, index]
        
        features = torch.cat((row_features, col_features), dim=1)
        concepts = self.concept_extractor(features)
        batches, channels, numbers, spatials = concepts.size()

        concepts = concepts.reshape(batches, 1, channels*numbers*spatials)
        concepts = self.featr_pool(concepts).reshape(batches, self.featr_dims)
        scores = self.predictor(concepts)

        return scores


    def _forward_to_rpms(self, ctx_features, cho_features, context_n=8):
        features = torch.stack(
            [torch.cat((ctx_features, cho_features[:,i].unsqueeze(1)), dim=1)
            for i in range(context_n)],
            dim=1
        )
        batches, numbers, frames, channels, spatials = features.size()
        return features.reshape(batches*numbers, frames, channels, spatials)

    def _forward_to_cols(self, features, index):
        return features[:, index]


    def _forward_feature(self, x):
        batches, frames, height, width = x.size()
        if self.in_channels == 1:
            x = x.reshape(batches*frames, 1, height, width)
        features = self.feature_extractor(x)
        features = self.feature_reducer(features)
        features = features.reshape(batches, frames, features.size(1), -1)
        return features


    def _forward_ultrain(self, ctx_features, cho_features):

        batches, _, channels, spatials = ctx_features.size()

        labels = torch.zeros(batches, dtype=torch.long).cuda()

        row_features = self._replace_choices(ctx_features, cho_features, labels)

        num_matrix_per_rpm = row_features.size(1)
        row_features = row_features.reshape(batches*num_matrix_per_rpm, -1, channels, spatials)
        col_features = self._forward_to_cols(row_features, self.row2col[:-1])
        
        features = torch.cat((row_features, col_features), dim=1)
        concepts = self.concept_extractor(features)
        concepts = concepts.reshape(batches, num_matrix_per_rpm, -1)
        concepts = self.featr_pool(concepts).reshape(batches*num_matrix_per_rpm, self.featr_dims)
        
        scores = self.predictor(concepts)

        return scores, labels


    @torch.no_grad()
    def _forward_ultest(self, ctx_features, cho_features):

        row_features = self._forward_to_rpms(ctx_features, cho_features)
        col_features = self._forward_to_cols(row_features, self.row2col)

        scores_rm3 = self._forward_remove_iframe(row_features, col_features, self.rmv_3fm)
        scores_rm6 = self._forward_remove_iframe(row_features, col_features, self.rmv_6fm)

        scores = scores_rm3 + scores_rm6

        return scores
        
    
    def forward(self, x):

        # get feature individually
        img_features = self._forward_feature(x)
        # split to 8 context features and 8 choices features
        # ctx_features [b 8 c hw]
        # cho_features [b 8 c hw]
        ctx_features, cho_features = img_features[:,:8], img_features[:,8:]

        if self.unsupervised_training and self.training:
            scores, labels = self._forward_ultrain(ctx_features, cho_features)
            return scores.reshape(x.size(0), -1), labels
        else:
            scores = self._forward_ultest(ctx_features, cho_features)
            return scores.reshape(x.size(0), -1)

def sslvmb(args):
    return SSLvMB(model="resnet4b", per_choice_loc_type="fix", args=args)

def sslvmb_cr(args):
    return SSLvMB(model="resnet4b", per_choice_loc_type="random", args=args)