import torch
import math
import torch.nn as nn



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

# MLP (multi-head perceptron)
class VAModel(nn.Module):
    def __init__(self, input_dim, embed_dim=512, va_dim=8):
        super(VAModel, self).__init__()
        # AV value
        self.v_predictor = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, va_dim)
        )
        self.a_predictor = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, va_dim)
        )
    def forward(self, x):
        # B X 256
        valence = self.v_predictor(x).unsqueeze(2)
        arousal = self.a_predictor(x).unsqueeze(2)
        # B X 256 X 2
        emotion = torch.cat([valence, arousal], dim=-1)
        return emotion
    

class EmoModel(nn.Module):
    def __init__(self, d_emb_emo=256, va_dim=16):
        super(EmoModel, self).__init__()
        # embedding
        self.chord_color_embed = nn.Linear(256, 32)
        self.rhy_ptn_embed = nn.Linear(256, 32, bias=False)
        self.struct_embed_one = nn.Linear(4, 16, bias=False)
        self.struct_embed_two = nn.Linear(2, 16)
        self.contour_embed_one = nn.Linear(3, 16)
        self.contour_embed_two = nn.Linear(3, 16)
        self.music_feat_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        # Previous chord and notes
        self.chord_embed = nn.Embedding(17327, d_emb_emo)
        self.pos_chord = PositionalEncoding(d_emb_emo)
        self.chord_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_emb_emo, 2),
            num_layers=4,
        )
        self.note_embed = nn.Embedding(113, d_emb_emo)
        self.pos_note = PositionalEncoding(d_emb_emo)
        self.note_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_emb_emo, 2),
            num_layers=4,
        )
        # AV value
        self.emo_model = VAModel(input_dim=d_emb_emo * 2 + 32, embed_dim=d_emb_emo, va_dim=va_dim)


    def forward(self, datas):
        '''
        4 musical features
        B X 256 chord_color
        B X 256 rhy_patn
        B X (4 + 2) struct
        B X (3 + 3) contour
        '''
        # Music Feature Embedding
        music_feat = datas['music_feat']
        rhy_ptn = self.rhy_ptn_embed(music_feat[..., :256])
        chord_color = self.chord_color_embed(music_feat[..., 256:512])
        struct_one = self.struct_embed_one(music_feat[..., 512:516])
        struct_two = self.struct_embed_two(music_feat[..., 516:518])
        contour_one = self.contour_embed_one(music_feat[..., 518:521])
        contour_two = self.contour_embed_two(music_feat[..., 521:524])
        music_feat = torch.cat([rhy_ptn, chord_color, struct_one, struct_two, contour_one, contour_two], dim=-1)
        embed_music_feat = self.music_feat_out(music_feat)
        # Note & Chord Embedding
        chord = datas['chord_true']
        note = datas['note_true']
        embed_chord = self.pos_chord((self.chord_embed(chord)))
        embed_chord = self.chord_encoder(embed_chord).mean(dim=1)
        embed_note = self.pos_note((self.note_embed(note)))
        embed_note = self.note_encoder(embed_note).mean(dim=1)
        features = torch.cat([embed_music_feat, embed_chord, embed_note], dim=-1)
        # Predict VA B X 256 X 2
        predict_VA = self.emo_model(features)
        return predict_VA, features

class EmoModelPure(nn.Module):
    def __init__(self, d_emb_emo=256, va_dim=16):
        super(EmoModelPure, self).__init__()
        # embedding
        self.chord_color_embed = nn.Linear(256, 32)
        self.rhy_ptn_embed = nn.Linear(256, 32, bias=False)
        self.struct_embed_one = nn.Linear(4, 16, bias=False)
        self.struct_embed_two = nn.Linear(2, 16)
        self.contour_embed_one = nn.Linear(3, 16)
        self.contour_embed_two = nn.Linear(3, 16)
        self.music_feat_out = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        # AV value
        self.emo_model = VAModel(input_dim=32, embed_dim=d_emb_emo, va_dim=va_dim)


    def forward(self, datas):
        # Music Feature Embedding
        music_feat = datas['music_feat']
        rhy_ptn = self.rhy_ptn_embed(music_feat[..., :256])
        chord_color = self.chord_color_embed(music_feat[..., 256:512])
        struct_one = self.struct_embed_one(music_feat[..., 512:516])
        struct_two = self.struct_embed_two(music_feat[..., 516:518])
        contour_one = self.contour_embed_one(music_feat[..., 518:521])
        contour_two = self.contour_embed_two(music_feat[..., 521:524])
        music_feat = torch.cat([rhy_ptn, chord_color, struct_one, struct_two, contour_one, contour_two], dim=-1)
        embed_music_feat = self.music_feat_out(music_feat)
        # Predict VA B X 256 X 2
        predict_VA = self.emo_model(embed_music_feat)
        return predict_VA, embed_music_feat
