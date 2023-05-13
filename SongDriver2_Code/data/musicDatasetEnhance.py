import torch
import json
from torch.utils.data import Dataset
from data.data_preprocess import *
from models.transformer import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

notes2id = json.load(open('dataset/enhance_notes_vocab.json', 'r'))
melody2id = json.load(open('dataset/enhance_melody_vocab.json', 'r'))
chord2id = json.load(open('dataset/enhance_chord_vocab.json', 'r'))


class AiMusicDataset(Dataset):
    def __init__(self, data_pth, labeled=True, emotion_step=16):
        super(AiMusicDataset).__init__()
        self.emotion_step = emotion_step
        if labeled:
            melody_enhance, melody_origin, chord, contour, rhy_ptn, struct, chord_color, keys, emotion = read_npz(data_pth, labeled)
            self.melody_enhance = melody_enhance
            self.melody_origin = melody_origin
            self.chord = chord
            self.contour = contour
            self.rhy_ptn = rhy_ptn
            self.struct = struct
            self.chord_color = chord_color
            self.emotion = emotion
            self.keys = keys
        else:
            melody_enhance, melody_origin, chord, contour, rhy_ptn, struct, chord_color, keys = read_npz(data_pth, labeled)
            self.melody_enhance = melody_enhance
            self.melody_origin = melody_origin
            self.chord = chord
            self.contour = contour
            self.rhy_ptn = rhy_ptn
            self.struct = struct
            self.chord_color = chord_color
            self.keys = keys

        print(f'{len(self.melody_enhance)}')
        self.labeled = labeled

    def __len__(self):
        return len(self.melody_enhance)

    def __getitem__(self, ndx):
        melody = self.melody_enhance[ndx]
        note = self.melody_origin[ndx]
        chord = self.chord[ndx]
        # 4 musical features
        rhy_ptn = self.rhy_ptn[ndx]
        chord_color = self.chord_color[ndx]
        struct = self.struct[ndx]
        contour = self.contour[ndx]
        keys = self.keys[ndx]
        music_feat = np.concatenate((rhy_ptn, chord_color, struct, contour), axis=-1)
        out = {
            'melody': torch.IntTensor(melody),
            'note_in': torch.LongTensor(np.concatenate(([notes2id['S']], note), axis=-1)),
            'note_out': torch.LongTensor(np.concatenate((note, [notes2id['E']]), axis=-1)),       
            'chord_in': torch.LongTensor(np.concatenate(([chord2id['S']], chord), axis=-1)),
            'chord_out': torch.LongTensor(np.concatenate((chord, [chord2id['E']]), axis=-1)),
            'music_feat': torch.FloatTensor(music_feat),
            'tone': torch.FloatTensor(keys)
        }
        if out['melody'].shape[-1] != 64:
            pad = torch.zeros((64 - out['melody'].shape[-1]))
            out['melody'] = torch.cat((out['melody'], pad), dim=-1)
        if self.labeled:
            out['emotion'] = torch.FloatTensor(self.emotion[ndx])[::self.emotion_step, :]
        return out


class AiMusicDatasetEmo(Dataset):
    def __init__(self, data_pth, labeled=True, step=16):
        super(AiMusicDataset).__init__()
        self.step = step
        if labeled:
            melody_enhance, melody_origin, chord, contour, rhy_ptn, struct, chord_color, keys, emotion = read_npz(data_pth, labeled)
            self.melody_enhance = melody_enhance
            self.melody_origin = melody_origin
            self.chord = chord
            self.contour = contour
            self.rhy_ptn = rhy_ptn
            self.struct = struct
            self.chord_color = chord_color
            self.emotion = emotion
            self.keys = keys
        else:
            melody_enhance, melody_origin, chord, contour, rhy_ptn, struct, chord_color, keys = read_npz(data_pth, labeled)
            self.melody_enhance = melody_enhance
            self.melody_origin = melody_origin
            self.chord = chord
            self.contour = contour
            self.rhy_ptn = rhy_ptn
            self.struct = struct
            self.chord_color = chord_color
            self.keys = keys

        print(f'{len(self.melody_enhance)}')
        self.labeled = labeled

    def __len__(self):
        return len(self.melody_enhance)

    def __getitem__(self, ndx):
        melody = self.melody_enhance[ndx]
        note = self.melody_origin[ndx]
        chord = self.chord[ndx]
        # 4 musical features
        rhy_ptn = self.rhy_ptn[ndx]
        chord_color = self.chord_color[ndx]
        struct = self.struct[ndx]
        contour = self.contour[ndx]
        music_feat = np.concatenate((rhy_ptn, chord_color, struct, contour), axis=-1)
        out = {
            'melody': torch.IntTensor(melody),
            'chord_true': torch.LongTensor(chord),
            'note_true': torch.LongTensor(note),
            'music_feat': torch.FloatTensor(music_feat),
        }
        if out['melody'].shape[-1] != 64:
            print('shape')
            pad = torch.zeros((64 - out['melody'].shape[-1]))
            out['melody'] = torch.cat((out['melody'], pad), dim=-1)
        if self.labeled:
            out['emotion'] = torch.FloatTensor(self.emotion[ndx])[::self.step, :]
        return out

