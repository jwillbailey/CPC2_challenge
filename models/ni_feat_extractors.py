import torch
import torch.nn.functional as F
from speechbrain.processing.features import STFT, spectral_magnitude
from torch import Tensor, nn

# try: #look in two places for the HuBERT wrapper
from models.huBERT_wrapper import (HuBERTWrapper_extractor, HuBERTWrapper_full,
                                   WhisperWrapper_encoder, WhisperWrapper_full,
                                   WhisperWrapperBase)
from models.wav2vec2_wrapper import (Wav2Vec2Wrapper_encoder_only,
                                     Wav2Vec2Wrapper_no_helper)

# from models.ni_predictors import PoolAttFF


class Spec_feats(nn.Module):

    def __init__(self):
        super().__init__()

        self.stft = STFT(hop_length=16,win_length=32,sample_rate=16000,n_fft=512,window_fn=torch.hamming_window)


    def forward(self, x):

        feats = self.stft(x)
        feats = spectral_magnitude(feats, power=0.5)
        out_feats = torch.log1p(feats)

        return out_feats

class XLSRCombo_feats(nn.Module):

    def __init__(self):
        super().__init__()

        self.feat_extract = Wav2Vec2Wrapper_no_helper()

    def forward(self, x):
        mod_out = self.feat_extract(x)
        out_feats_full =  mod_out['last_hidden_state']#.permute(0,2,1)
        out_feats_extact = mod_out['extract_features']

        return out_feats_full, out_feats_extact



class XLSRFull_feats(nn.Module):

    def __init__(self):
        super().__init__()

        self.feat_extract = Wav2Vec2Wrapper_no_helper()

    def forward(self, x):

        return self.feat_extract(x)['last_hidden_state']#.permute(0,2,1)


class XLSREncoder_feats(nn.Module):

    def __init__(self):
        super().__init__()

        self.feat_extract = Wav2Vec2Wrapper_encoder_only()
        self.i = 0

    def forward(self, x):

        x = self.feat_extract(x).permute(0,2,1)

        return x

        
class HuBERTFull_feats(nn.Module):

    def __init__(self):
        super().__init__()

        self.feat_extract = HuBERTWrapper_full()

    def forward(self, x):

        return self.feat_extract(x)#.permute(0,2,1)

class HuBERTEncoder_feats(nn.Module):

    def __init__(self):
        super().__init__()

        self.feat_extract = HuBERTWrapper_extractor()

    def forward(self, x):

        return self.feat_extract(x).permute(0,2,1)


class WhisperEncoder_feats(nn.Module):

    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None):
        super().__init__()

        self.feat_extract = WhisperWrapper_encoder(
            layer = layer,
            pretrained_model = pretrained_model,
            use_feat_extractor = use_feat_extractor
        )

    def forward(self, x):

        x = self.feat_extract(x)#.permute(0,2,1)
        # print(f"whisperencoder_feats: {x.size()}")

        return x
    
    
class WhisperFull_feats(nn.Module):

    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None):
        super().__init__()
        
        self.feat_extract = WhisperWrapper_full(
            layer = layer,
            pretrained_model = pretrained_model,
            use_feat_extractor = use_feat_extractor
        )

    def forward(self, x):

        x = self.feat_extract(x)#.permute(0,2,1)
        # print(f"whisperencoder_feats: {x.size()}")

        return x
    

class WhisperBase_feats(nn.Module):

    def __init__(self, layer = None, use_feat_extractor = False, pretrained_model = None):
        super().__init__()
        
        self.feat_extract = WhisperWrapperBase(
            layer = layer,
            pretrained_model = pretrained_model,
            use_feat_extractor = use_feat_extractor
        )

    def forward(self, x):

        x = self.feat_extract(x)#.permute(0,2,1)
        # print(f"whisperencoder_feats: {x.size()}")

        return x