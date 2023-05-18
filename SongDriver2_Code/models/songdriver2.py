import torch
import torch.nn as nn
import math
from config import Config
from models.transformer import DualTransformer
from models.emo_model import EmoModel, EmoModelNoRhyPtn, EmoModelNoChordColor, EmoModelNoContour, EmoModelNoStructEmbed

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SongDriver2(nn.Module):
    def __init__(
            self, emo_ckpt_pth=None, chord_yaml='config_m2c.yaml', 
            melody_yaml='config_n2m.yaml', emo_fusion_type='replace', 
            d_emb_emo=512, va_dim=16, unlabeled=False, remove_music_feat=None
        ):
        super(SongDriver2, self).__init__()
        args_chord = Config(chord_yaml)
        args_melody = Config(melody_yaml)
        self.emo_fusion_type = emo_fusion_type
        # Melody to Chord, Notes
        self.dual_generator = DualTransformer(args_melody)
        # Emotion Model
        if not remove_music_feat:
            self.emo_model = EmoModel(d_emb_emo, va_dim=va_dim)
        elif remove_music_feat == 'chord':
            self.emo_model = EmoModelNoChordColor(d_emb_emo, va_dim=va_dim)
        elif remove_music_feat == 'rhy':
            self.emo_model = EmoModelNoRhyPtn(d_emb_emo, va_dim=va_dim)
        elif remove_music_feat == 'contour':
            self.emo_model = EmoModelNoContour(d_emb_emo, va_dim=va_dim)
        elif remove_music_feat == 'struct':
            self.emo_model = EmoModelNoStructEmbed(d_emb_emo, va_dim=va_dim)

        # load checkpoint
        if emo_ckpt_pth:
            ckpt = torch.load(emo_ckpt_pth)
            self.emo_model.load_state_dict(ckpt['model'])
            print('load emotion model from {}'.format(emo_ckpt_pth))
        # Emotion Embedding
        self.emotion_embed = nn.Linear(2, d_emb_emo)
        self.pos_emotion_embed = PositionalEncoding(d_emb_emo)
        self.emotion_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_emb_emo, 2),
            num_layers=4,
        )
        if emo_fusion_type == 'concat':
            self.emo_out = nn.Linear(d_emb_emo * 2, d_emb_emo)
        elif emo_fusion_type == 'replace':
            self.emo_out = nn.Linear(d_emb_emo * 3 + 32, d_emb_emo)
        else:
            self.emo_out = nn.Identity()
        self.unlabeled = unlabeled
        print('MuThera Unlabeled: ', self.unlabeled)

    def forward(self, now_input, pre_output=None, alpha=1.0):
        '''
        4 musical features
        B X 256 chord_color
        B X 256 rhy_patn
        B X (4 + 2) struct
        B X (3 + 3) contour
        '''
        if self.unlabeled:
            cur_VA = self.emo_model(now_input)
        else:
            cur_VA = now_input['emotion'].float()
        cur_VA = cur_VA.detach()
        if not pre_output:
            predict_VA = 0
            emotions = self.pos_emotion_embed(self.emotion_embed(cur_VA))
            emotions = self.emotion_encoder(emotions).mean(dim=1)
        else:
            # Predict VA B X 256 X 2
            predict_VA, features = self.emo_model(pre_output)
            # Recognize Emotion
            if self.emo_fusion_type == 'replace':
                # first get av
                cur_emotions = self.pos_emotion_embed(self.emotion_embed(cur_VA))
                cur_emotions = self.emotion_encoder(cur_emotions).mean(dim=1)
                emotions = torch.cat((cur_emotions, features), dim=-1)
                emotions = self.emo_out(emotions)
            elif self.emo_fusion_type == 'concat':
                # (1 - a) * true_va + a * pre_va for soft fusion
                fusion_va = predict_VA if self.unlabeled else (1 - alpha) * pre_output['emotion'] + alpha * predict_VA 
                # Embed Previous and Current Emotion
                cur_emotions = self.pos_emotion_embed(self.emotion_embed(cur_VA))
                cur_emotions = self.emotion_encoder(cur_emotions).mean(dim=1)
                pre_emotions = self.pos_emotion_embed(self.emotion_embed(fusion_va))
                pre_emotions = self.emotion_encoder(pre_emotions).mean(dim=1)
                # concat emotion
                emotions = torch.cat((cur_emotions, pre_emotions), dim=-1)
                emotions = self.emo_out(emotions)
            elif self.emo_fusion_type == 'median':
                # (1 - a) * true_va + a * pre_va for soft fusion
                fusion_va = predict_VA if self.unlabeled else (1 - alpha) * pre_output['emotion'] + alpha * predict_VA 
                emotions = (fusion_va + cur_VA) / 2
                emotions = self.pos_emotion_embed(self.emotion_embed(emotions))
                emotions = self.emotion_encoder(emotions).mean(dim=1)
            else:
                raise NotImplementedError
        # 生成旋律Melody 和弦Chord
        generate_note, generate_chord = self.dual_generator(now_input['melody'], now_input['note_in'], emotions)

        '''
        返回生成和弦、旋律, 当前情感(作为下一次计算的pre_emotion)
        '''
        out = {
            # 'music_feat': now_input['music_feat'],
            'tone': now_input['tone'],
            'chord': generate_chord,
            # 'chord_in': now_input['chord_in'],
            'note': generate_note,
            'note_in':  now_input['note_in'],
            'predict_VA': predict_VA,
            # 与下一时刻的predict_VA计算损失，因为predict_VA是预测上一时刻的VA
            'emotion': cur_VA
        }
        return out
