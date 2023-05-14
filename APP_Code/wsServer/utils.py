
import json
import torch
import os
import numpy as np
from tqdm import tqdm
from strConverter import *
from music_feature_extractor_ml import EmoFeatureExtractor


melody_vocab = json.load(open('dataset/melody_vocab.json', 'r'))
chord_vocab = json.load(open('dataset/chord_vocab.json', 'r'))
note_vocab = json.load(open('dataset/notes_vocab.json', 'r'))

feat_extractor = EmoFeatureExtractor()

def keys2list(keystr):
    def key_2_ref_chord(key):
        if key == 'C.MAJOR':
            ans = [0, 4, 7]
        elif key == 'D.MAJOR':
            ans = [2, 6, 9]
        elif key == 'E.MAJOR':
            ans = [4, 8, 11]
        elif key == 'F.MAJOR':
            ans = [5, 9, 12]
        elif key == 'G.MAJOR':
            ans = [7, 11, 14]
        elif key == 'A.MAJOR':
            ans = [9, 13, 16]
        elif key == 'B.MAJOR':
            ans = [11, 15, 18]
        elif key == 'C.MINOR':
            ans = [0, 3, 7]
        elif key == 'D.MINOR':
            ans = [2, 5, 9]
        elif key == 'E.MINOR':
            ans = [4, 7, 11]
        elif key == 'F.MINOR':
            ans = [5, 8, 12]
        elif key == 'G.MINOR':
            ans = [7, 10, 14]
        elif key == 'A.MINOR':
            ans = [9, 12, 16]
        elif key == 'B.MINOR':
            ans = [11, 14, 18]
        else:
            ans = [0, 4, 7]
        return ans
    keystr = keystr[2:-2]
    keystr = keystr.split('),(')
    keys = []
    for t in keystr:
        pair = t.split(',')
        keys.append((pair[0], int(pair[1])))
    keys_high_sampled = []
    for t in keys:
        keys_high_sampled.extend([t[0]] * t[1])
    keys_1d = keys_high_sampled[::16]
    tones = []
    for i in range(len(keys_1d)):
        tones.append(key_2_ref_chord(keys_1d[i]))
    return tones

def getDataFromMusicFile(file_path):
    if not os.path.exists(file_path):
        print('[Error] File {} does not exist!'.format(file_path))
        exit(1)
    # Handle labeled data
    np_melody_high = []
    np_melody_low = []
    np_chord = []
    np_keys = []

    _, keys, melody_low, melody_high, chord, _, _ = read_txt(file_path)
    np_keys.extend(keys)

    # 2d-high melody to 1d
    melody_high_1d = []
    for melodys in melody_high:
        tmp = []
        for m in melodys:
            tmp.extend([m[0]] * m[1])
        melody_high_1d.append(tmp)  
    melody_high = melody_high_1d

    melody_low_ndx = []
    melody_high_ndx = []
    chord_ndx = []

    for m in melody_low:
        melody_low_ndx = []
        for word in m:
            word = str(word)
            if word not in melody_vocab:
                print('Error: {}'.format(word))
                exit(0)
            melody_low_ndx.append(melody_vocab[word])
        np_melody_low.append(melody_low_ndx)

    for m in melody_high:
        melody_high_ndx = np.zeros((256, ))
        for n, word in enumerate(m):
            word = str(word)
            if word not in note_vocab:
                print('Error: {}'.format(word))
                exit(0)
            melody_high_ndx[n] = note_vocab[word]
        np_melody_high.append(melody_high_ndx)

    for c in chord:
        chord_ndx = np.zeros((16, ))
        for n, word in enumerate(c):
            if word not in chord_vocab:
                print('Error: {}'.format(word))
                exit(0)
            chord_ndx[n] = chord_vocab[word]
        np_chord.append(chord_ndx)

    start_chord = np.concatenate([[[chord_vocab['S']]] * len(np_chord)], axis=0)
    start_note = np.concatenate([[[note_vocab['S']]] * len(np_melody_high)], axis=0)

    datas = {}
    datas['tone'] = torch.from_numpy(np.array(np_keys))
    datas['melody'] = torch.from_numpy(np.array(np_melody_low))
    datas['chord_in'] = torch.LongTensor(np.concatenate((start_chord, np_chord), axis=-1))
    datas['note_in'] = torch.LongTensor(np.concatenate((start_note, np_melody_high), axis=-1))
    datas['music_feat'] = 0
    return datas


def read_txt(pth):
    """
    :param pth: the path of the dataset which is saved as a txt file.
    :return:
    """
    emo_av = []
    keys = []
    melody_low = []
    melody_high = []
    chord = []
    terminate = []
    music_feat = []
    with open(pth, 'r') as f:
        strs = f.readlines()
        for n, i in enumerate(strs):
            datas = i.strip('\n').split('|')
            is_labeled = (len(datas) == 7)
            print(is_labeled)
            if is_labeled:
                for n, d in enumerate(datas):
                    # melody high
                    if n == 3:
                        datas[n] = d.replace('),', '), ')
                    elif n == 1:
                        continue
                    else:
                        datas[n] = d.replace(',', ', ')
                # AV
                if 'NaN' in datas[0]:
                    datas[0] = datas[0].replace('NaN', '0.00')
                    print(pth)
                if '(-' in datas[3]:
                    continue
                emo_av.append(eval(datas[0]))
                keys.append(keys2list(datas[1]))
                melody_low.append(eval(datas[2]))
                melody_high.append(eval(datas[3]))
                chord.append(chord_str_to_list(datas[4]))
                terminate.append(eval(datas[5]))
                # Music Features
                music_feat.append(eval(datas[6]))
            else:
                for n, d in enumerate(datas):
                # melody high
                    if n == 2:
                        datas[n] = d.replace('),', '), ')
                    elif n == 0:
                        continue
                    else:
                        datas[n] = d.replace(',', ', ')
                # AV
                if '(-' in datas[3]:
                    continue
                keys.append(keys2list(datas[0]))
                melody_low.append(eval(datas[1]))
                melody_high.append(eval(datas[2]))
                chord.append(chord_str_to_list(datas[3]))
                terminate.append(datas[4])
                # Music Features
                music_feat.append(eval(datas[5]))

    return emo_av, keys, melody_low, melody_high, chord, terminate, music_feat

if __name__ == '__main__':
    print(getDataFromMusicFile("dataset/labeled/ccmed-ypf-╟щ╕╨▒ъ╟й╥╤▒г┴Ї2╬╗╨б╩¤-╫ю╓╒-╬┤╔╛╝ї/xi-emo/Beethoven_Op030No2-01_109_20061215-SMD_2.mp3.emo.txt"))
    print(getDataFromMusicFile("dataset/unlabeled/unlabel-╚л▓┐-┤ж└э║═╔╕╤б═ъ▒╧/5.txt"))
