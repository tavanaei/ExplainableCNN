import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,G_kernels,D_kernels,G_strides,D_strides,G_patch,D_patch,fcc_layers,G_pool='max',D_pool='max',dropout=0.3,prob=False,size=32,res=False):
        super(Model,self).__init__()
        self._n_class = fcc_layers[-1]
        self.__padG = G_patch//2
        self.__padD = D_patch//2
        self.code = None
        self.maps = None
        self.res = res
        decode = []
        encode = []
        discri = []
        fcc = []
        channel = G_kernels[0]
        if G_pool=='average':
            self.__poolG = nn.AvgPool2d
        else:
            self.__poolG = nn.MaxPool2d
        if D_pool=='average':
            self.__poolD = nn.AvgPool2d
        else:
            self.__poolD = nn.MaxPool2d

        for i in range(len(G_kernels)-1):
            decode.append(nn.Conv2d(G_kernels[i],G_kernels[i+1],G_patch,padding=self.__padG,stride=1))
            #decode.append(nn.BatchNorm2d(G_kernels[i+1]))
            decode.append(nn.ReLU())
            decode.append(self.__poolG(G_strides[i]))

        G_kernels[0]=1
        for i in range(len(G_kernels)-1,0,-1):
            #encode.append(nn.ConvTranspose2d(G_kernels[i],G_kernels[i-1],G_patch,padding=self.__padG,stride=G_strides[i-1],output_padding=G_strides[i-1]//2))
            encode.append(nn.UpsamplingBilinear2d(scale_factor=G_strides[i-1]))
            encode.append(nn.Conv2d(G_kernels[i],G_kernels[i-1],G_patch,padding=self.__padG,stride=1))
            if i==1:
                encode.append(nn.Tanh())
            else:
                #encode.append(nn.BatchNorm2d(G_kernels[i-1]))
                encode.append(nn.ReLU())
        
        for i in range(len(D_kernels)-1):
            discri.append(nn.Conv2d(D_kernels[i],D_kernels[i+1],D_patch,padding=self.__padD,stride=1))
            discri.append(nn.BatchNorm2d(D_kernels[i+1]))
            discri.append(nn.ReLU())
            if D_strides[i]>1:
                discri.append(self.__poolD(D_strides[i]))
            #else:
            #    discri.append(nn.Dropout2d(dropout))

        for i in range(len(fcc_layers)-2):
            fcc.append(nn.Dropout(dropout))
            fcc.append(nn.Linear(fcc_layers[i],fcc_layers[i+1]))
            fcc.append(nn.ReLU())
        #fcc.append(nn.Dropout(dropout))
        fcc.append(nn.Linear(fcc_layers[-2],self._n_class))
        
        self.decoder = nn.Sequential(*decode)
        self.encoder = nn.Sequential(*encode)
        self.feature = nn.Sequential(*discri)
        self.classifier = nn.Sequential(*fcc)
        
        self.residual = nn.Sequential(*[nn.Conv2d(3,1,1,stride=1,padding=0),nn.Tanh()])

    def forward(self,x):
        raw_im = x.clone()
        self.code = self.decoder(x)
        self.maps = self.encoder(self.code)
        
        """
        This Section is for adding the CNV(1x1) next to the generator model
        It improves the model's accuracy rate but detains the heatmap quality slightly
        More detail: Paper, Discussion section
        """
        #if self.res:
            #res_x = self.residual(x)
            #new_x = torch.cat((self.maps,res_x),1)
            #feature_maps = self.feature(new_x)
        
        #else:
        feature_maps = self.feature(self.maps)
        
        y = feature_maps.view(feature_maps.size(0),-1)
        out = self.classifier(y)

        return out


        




        



        
