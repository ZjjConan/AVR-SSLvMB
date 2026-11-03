import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CConvBlock, Classifier, ConvBNAct
from .resnet4b import resnet4b
from .models import register

class SSLvMB(nn.Module):
# Self-Supervised Learning via Mental Bootstrapping

    def __init__(
            self, 
            model="resnet4b", 
            in_channels=1, 
            ou_channels=1, 
            block_drop=0.1, 
            classifier_drop=0.1, 
            num_extra_stages=3,
            classifier_hidreduce=4
        ):
        super().__init__()


        self.n_way = 2
        self.n_shot = 6

        # frame extractor
        # ---------------
        self.in_channels = in_channels
        self.ou_channels = ou_channels

        self.feature_extractor = resnet4b(
            in_channels=in_channels, block_drop=block_drop, strides=[2,2,2,1]
        )
        conv_in_channels = self.feature_extractor.ou_channels
        
        conv_re_channels = 64
        conv_md_channels = 64

        self.feature_reducer = ConvBNAct(
            conv_in_channels, conv_re_channels, 
            conv_dim=2, kernel_size=1, padding=0, 
            stride=1, bias=False, use_act=False
        )

        # concept induction module
        # ---------------
        conv_in_channels = conv_re_channels
        self.concept_extractor = []
        for i in range(num_extra_stages):

            downsample = ConvBNAct(
                conv_in_channels, conv_in_channels, 
                conv_dim=2, kernel_size=1, padding=0, 
                stride=1, bias=False, use_act=False
            )
            
            self.concept_extractor.append(
                CConvBlock(
                    conv_in_channels, 
                    conv_in_channels,
                    md_channels=conv_md_channels,
                    downsample=downsample, 
                    dropout=block_drop,
                    permute=False
                )
            )

        self.concept_extractor = nn.Sequential(*self.concept_extractor)

        # predictor
        self.featr_dims = 1024
        self.featr_pool = torch.nn.AdaptiveAvgPool1d(self.featr_dims)

        self.predictor = Classifier(
            self.featr_dims, 2, 
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            classifier_hidreduce = classifier_hidreduce
        )


    def _select_pseudo_query(self, ctx_features):
        batches, frames = ctx_features.shape[0:2]

        sel_index = torch.randint(0, frames, (1,))
        query = ctx_features[:, sel_index]
        masks = torch.ones_like(ctx_features)
        masks[:, sel_index] = 0
        masks = masks * frames/(frames-1)
        ctx_features = ctx_features * masks

        return query, ctx_features

    def _forward_feature(self, x):
        features = self.feature_extractor(x)
        features = self.feature_reducer(features)
        return features

    def forward(self, x_shot, x_query, **kwargs):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        ep_per_batch, n_way, n_shot = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        assert n_shot == self.n_shot and n_way == self.n_way
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)

        x_tot = torch.cat([x_shot, x_query], dim=0)
        x_tot = self._forward_feature(x_tot)

        n, c, h, w = x_tot.size()

        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(ep_per_batch, n_way, n_shot, c, h*w)  # [ep_per_batch, n_way, n_shot, chans, spatial]
        x_shot_pos, x_shot_neg = x_shot[:, 0], x_shot[:, 1]  # [ep_per_batch, n_way, n_shot, chans, spatial]
        x_query = x_query.view(*query_shape, c, h*w)  # [ep_per_batch, n_way * n_query, chans, spatial]

        if self.training:
            x_query_pos, x_shot_pos = self._select_pseudo_query(x_shot_pos)
            x_query_neg, x_shot_neg = self._select_pseudo_query(x_shot_neg)
            x_query = torch.cat([x_query_pos, x_query_neg], dim=1)

        num_query_samples = x_query.size(1)  # n_way * n_query
        x_tot = []
        for i in range(num_query_samples):
            x_query_single = x_query[:, i:i + 1]  # [ep_per_batch, 1, chans, spatial]
                
            # positive
            x_pos = torch.cat([x_shot_pos, x_query_single], dim=1)  
            # negative
            x_neg = torch.cat([x_shot_neg, x_query_single], dim=1)  # [ep_per_batch, n_shot, chans, spatial]

            x_bps = torch.cat((x_pos.unsqueeze(1), x_neg.unsqueeze(1)), dim=1)

            x_tot.append(x_bps.unsqueeze(1))
                
        x_tot = torch.cat(x_tot, dim=1) # [ep_per_batch, num_query, n_way, n_shot + 1, chans, spatial]

        # unsupervised learning with n_shot, supervised learning with n_shot + 1
        x_tot = x_tot.reshape(ep_per_batch*num_query_samples*n_way, n_shot+1, c, h*w)

        b, n, c, l = x_tot.size()
        x_tot = x_tot.permute(0,2,1,3)
        x_tot = self.concept_extractor(x_tot)

        x_tot = x_tot.reshape(ep_per_batch*num_query_samples, -1)
        x_tot = F.adaptive_avg_pool1d(x_tot, self.featr_dims)
        x_tot = x_tot.reshape(ep_per_batch*num_query_samples, self.featr_dims)

        logits = self.predictor(x_tot).view(
            ep_per_batch, num_query_samples, n_way
        )

        if self.training:
            labels = torch.zeros(ep_per_batch, num_query_samples, dtype=torch.long).to(x_tot.device)
            labels[:, 1] = 1
            return logits, labels
        else:
            return logits

        
@register('sslvmb-bd')
def sslvmb_bd(encoder, encoder_args={}):
    return SSLvMB(model="resnet4b")