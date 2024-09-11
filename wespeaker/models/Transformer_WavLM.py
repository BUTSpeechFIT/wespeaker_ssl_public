import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

# from einops import rearrange, repeat
from torch.nn.utils import remove_weight_norm

from wespeaker.models.ssl.modules import GradMultiply
from wespeaker.models.ssl_backend import *
from wespeaker.models.ssl.WavLM import *

class WavLM_MHFA(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim, group, cnn_scale=0.0, layer_drop=0.05, feature_grad_mult=0.05, randninit=False):
        super(WavLM_MHFA, self).__init__()
        checkpoint = torch.load(model_path)
        
        print(pooling)
        checkpoint['cfg']['encoder_layerdrop']=layer_drop
        checkpoint['cfg']['feature_grad_mult']=cnn_scale
        
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        if not randninit:
            self.loadParameters(checkpoint['model'])

        inputs_dim = checkpoint['cfg']['encoder_embed_dim']
        nb_layer = checkpoint['cfg']['encoder_layers'] + 1

        if pooling == 'MHFA':
            self.back_end = MHFA(inputs_dim=inputs_dim,head_nb=head_nb, outputs_dim=embed_dim, nb_layer=nb_layer)
        else:  # model_name error !!!
            print(pooling + " not found !!!")
            exit(1)

        self.feature_grad_mult = feature_grad_mult
        if self.feature_grad_mult == 0.0:
            self.fixed_condition = True
        else:
            self.fixed_condition = False


    def forward(self, raw_wav):
        
        # To avoid OOM, here we take 20s as max length
        max_length = 16000 * 20
        wav_segment = raw_wav[:, :max_length]

        if self.fixed_condition:
            with torch.no_grad():
                rep, layer_results = self.model.extract_features(wav_segment, output_layer=100)
        else:
            rep, layer_results = self.model.extract_features(wav_segment, output_layer=100)

        
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1) # [Batch, Dim, Frame_len, Nb_Layer]
        x = GradMultiply.apply(x, self.feature_grad_mult)
        
        spk_embedding = self.back_end(x)
        
        return spk_embedding


    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            

            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

if __name__ == "__main__":
    from thop import profile
    # from ptflops import get_model_complexity_info
    model_path = '/mnt/matylda6/pengjy/share_ssl/WavLM-Base+.pt'
    pooling = 'MHFA'
    embed_dim = 256
    head_nb = 128
    group = 1
    model = WavLM_MHFA(model_path, pooling, head_nb, embed_dim, group,cnn_scale=0.0,layer_drop=0.00)
    flops, params = profile(model.eval(), inputs=(torch.randn(1, 16000*2),))

    print("FLOPS: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
