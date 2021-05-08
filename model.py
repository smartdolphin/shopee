import math
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, AdaptiveAvgPool2d, Linear
from torch.nn.parameter import Parameter
from loss import FocalLoss


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5,
                 crit="bce", reduction="mean", easy_margin=False, ls_eps=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.reduction = reduction
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps
        
        if crit == "focal":
            self.crit = FocalLoss(gamma=args.focal_loss_gamma)
        elif crit == "bce":
            self.crit = nn.CrossEntropyLoss(reduction="none")   

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([30.], requires_grad=True, device='cuda'))
        else:
            self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        cosine = F.linear(F.normalize(logits), F.normalize(self.weight))
        sine = torch.sqrt(((1.0 + 1e-7) - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        onehot = torch.zeros_like(cosine)
        onehot.scatter_(1, labels.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = (onehot * phi) + ((1.0 - onehot) * cosine)
        output *= self.s

        loss = self.crit(output, labels)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return output, loss


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class ShopeeModel(nn.Module):
    def __init__(
        self, model_name='eca_nfnet_l1',
        n_classes=11014,
        fc_dim=512,
        margin=0.5,
        scale=30,
        crit='bce',
        ls_eps=0,
        use_fc=True,
        pretrained=True):

        super(ShopeeModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        self.gem = GeM()

        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif 'efficientnet' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        
        elif 'nfnet' in model_name:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        self.use_fc = use_fc

        if use_fc:          
            self.neck = nn.Sequential(
                nn.Linear(final_in_features, fc_dim),
                nn.BatchNorm1d(fc_dim),
                torch.nn.PReLU(),
            )
            final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            s = scale,
            m = margin,
            crit=crit,
            easy_margin = False,
            ls_eps = ls_eps
        )

    def forward(self, image, label):
        x = self.backbone(image)
        x = self.gem(x).squeeze()
        if self.use_fc:
            x = self.neck(x)
        logits = self.final(x, label)
        return logits
