import json
import os
import torch
from muthera_dual import DualMuThera
from utils import getDataFromMusicFile
from music_feature_extractor_ml import EmoFeatureExtractor

id2notes = json.load(open('dataset/id2note.json', 'r'))
id2chords = json.load(open('dataset/id2chord.json', 'r'))

notes2id = json.load(open('dataset/notes_vocab.json', 'r'))
melody2id = json.load(open('dataset/melody_vocab.json', 'r'))
chord2id = json.load(open('dataset/chord_vocab.json', 'r'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

NOTE_LENGTH = 256
CHORD_LENGTH = 16


'''
@param emotion: 长度 16的VA序列 
@param origin_music_file_pth: 音乐txt对应文件地址
@ret res: 字典包含了notes: [BatchSize * NOTE_LENGTH]和 chords: [BatchSize * CHORD_LENGTH]
@ret preouts: 用于下一次调用inferencea作为T-1时刻的信息
'''


@torch.no_grad()
def inference(model, emotion, music_file_pth, pre_outs=None):
    datas = getDataFromMusicFile(music_file_pth)
    music_extractor = EmoFeatureExtractor()
    batch_size = datas['note_in'].shape[0]

    now_emo = torch.Tensor(emotion).float().unsqueeze(
        0).tile((batch_size, 1, 1))
    now_datas = {
        'tone': datas['tone'].to(device),
        'melody': datas['melody'].to(device),
        'chord_in': datas['chord_in'].to(device),
        'note_in': datas['note_in'].to(device),
        'emotion': now_emo.to(device)
    }
    now_datas['chord_in'][:, 1:] = 0
    now_datas['note_in'][:, 1:] = 0
    # inference note
    for i in range(NOTE_LENGTH):
        outs = model(now_datas, pre_outs)
        note_logit = outs['note'].view(
            batch_size, NOTE_LENGTH + 1, -1)[:, i, :]
        _, note_ids = torch.topk(note_logit, k=1)
        for n, nid in enumerate(note_ids):
            now_datas['note_in'][n, i + 1] = nid
    # res
    note_logit = outs['note']
    notes = []
    _, note_ids = torch.topk(note_logit, k=1)
    for nid in note_ids:
        note = id2notes[str(nid.item())]
        if note in ['S', 'E', 'P']:
            note = '0'
        note = eval(note)
        notes.append(note)

    now_notes = []
    for i in range(0, len(notes), NOTE_LENGTH + 1):
        now_notes.extend(notes[i: i + NOTE_LENGTH])

    chord_logit = outs['chord']
    chords = []
    _, chord_ids = torch.topk(chord_logit, k=1)

    for cid in chord_ids:
        chord = id2chords[str(cid.item())]
        if chord in ['S', 'E', 'P']:
            chord = '[]'
        chord = eval(chord)
        chords.append(chord)

    now_chords = []
    for i in range(0, len(chords), CHORD_LENGTH + 1):
        now_chords.extend(chords[i: i + CHORD_LENGTH])

    # get music features
    pre_outs = outs
    pre_outs['music_feat'] = music_extractor(outs, contain_S_E=True).to(device)

    res = {'notes': now_notes, 'chords': now_chords}
    return res, pre_outs

def buildPlaySeq(res):
    n = res["notes"]
    c = res["chords"]
    res_str = ""
    for i in range(len(c)):
        noteArr = []
        for j in range(4):
            noteArr.append(n[(i*4+j)*4])
        chord = c[i]
        res_str+=f"{noteArr}|{chord}\n"
    return res_str
        

median = DualMuThera(emo_fusion_type='median').to(device)
# checkpoint file path
ckpt_pth = None
if ckpt_pth:
    median_ckpt = torch.load(ckpt_pth, map_location=device)
    median.load_state_dict(median_ckpt)

if __name__ == '__main__':
    # res, pre_out = inference(median, [(0.1, 0.1) for _ in range(
    #    16)], '/Users/male/Downloads/backend/test_data.txt')
    # res, pre_out = inference(median, [(0.1, 0.1) for _ in range(
    #    16)], '/Users/male/Downloads/backend/test_data.txt', pre_out)
    res, pre_out = inference(median, [(0.1, 0.1) for _ in range(16)],"test_data.txt")
    print(buildPlaySeq(res))
    print(len(res["notes"]),len(res["chords"]))
