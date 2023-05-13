
import json
import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    from strConverter import *
else:
    from data.strConverter import *


'''
特征列表：
1: (VA, 持续时长)
2: (调性名, 连续出现次数 * 16), 每一行时长为16
3. (旋律), 采样旋律时长4/个
4. (和弦), 时长16/个; (二维旋律音高, 时长)
5. 连续出现的1结束位置index * 6, index -> [1-16]
'''
    
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

def handle_music_feat(music_feat):
    tmp_rhy_ptn = music_feat[0]
    tmp_chord_color = music_feat[1]
    tmp_contour = music_feat[2]
    tmp_struct = music_feat[3]

    rhy_ptn = np.zeros((256, ), dtype=np.bool8)
    onset = 0
    for n in tmp_rhy_ptn:
        onset += n
        rhy_ptn[onset - 1] = 1
    
    chord_color = []
    for t in tmp_chord_color:
        chord_color.extend([t[0]] * t[1])
    
    contour = []
    for c in tmp_contour:
        contour.extend(list(c))
    
    struct = []
    for c in tmp_struct:
        if type(c) == int:
            struct.append(c)
        else:
            struct.extend(list(c))
    return rhy_ptn, chord_color, contour, struct

def read_npz(pth, labeled=True):
    npz = np.load(pth, allow_pickle=True)
    melody_low = npz["melody_low"]
    melody_high = npz["melody_high"]
    chord = npz['chord']
    contour = npz["contour"]
    rhy_ptn = npz["rhy_ptn"]
    struct = npz["struct"]
    chord_color = npz["chord_color"]
    keys = npz["keys"]
    if labeled:
        emotion = npz['emotion']
        return melody_low, melody_high, chord, contour, rhy_ptn, struct, chord_color, keys, emotion
    return melody_low, melody_high, chord, contour, rhy_ptn, struct, chord_color, keys

def data2npz(root_path):
    labeled_files = []
    unlabeled_files = []
    # build-in recursive function to search txt
    def get_all_files(file_path):
        if os.path.isdir(file_path):
            # get son folders
            files = [os.path.join(file_path, f) for f in os.listdir(file_path)]
            for f in files:
                get_all_files(f)
        elif file_path.endswith('.txt'):
            if 'unlabeled' in file_path:
                unlabeled_files.append(file_path)
            elif 'labeled' in file_path:
                labeled_files.append(file_path)
            else:
                print('Dirty File: {}'.format(file_path))
                exit(0)
    get_all_files(root_path)
    print(f'Labeled Data: {len(labeled_files)}')
    print(f'Unlabeled Data: {len(unlabeled_files)}')

    # Update Dict
    melody_vocab = {}
    chord_vocab = {}
    note_vocab = {}

    # Handle unlabeled data
    np_melody_high = []
    np_melody_low = []
    np_chord = []
    np_rhy_ptn = []
    np_chord_color = []
    np_contour = []
    np_struct = []
    np_keys = []
    for uf in tqdm(unlabeled_files):
        _, keys, melody_low, melody_high, chord, _, music_feat = read_txt(uf, False)
        # 2d-high melody to 1d
        # melody_high_1d = []
        # for melodys in melody_high:
        #     tmp = []
        #     for m in melodys:
        #         tmp.extend([m[0]] * m[1])
        #     melody_high_1d.append(tmp)  
        # melody_high = melody_high_1d
        continue_flag = False
        for m in melody_low:
            melody_low_ndx = []
            for word in m:
                if word not in melody_vocab:
                    melody_vocab[word] = len(melody_vocab) + 1
                melody_low_ndx.append(melody_vocab[word])
            np_melody_low.append(melody_low_ndx)
        np_keys.extend(keys)

        for mf in music_feat:
            rhy_ptn, chord_color, contour, struct = handle_music_feat(mf)
            np_rhy_ptn.append(rhy_ptn)
            np_chord_color.append(chord_color)
            np_contour.append(contour)
            np_struct.append(struct)

        for m in melody_high:
            melody_high_ndx = np.zeros((64, ))
            for n, word in enumerate(m):
                if word not in note_vocab:
                    note_vocab[word] = len(note_vocab) + 1
                melody_high_ndx[n] = note_vocab[word]
            np_melody_high.append(melody_high_ndx)

        for c in chord:
            chord_ndx = np.zeros((16, ))
            for n, word in enumerate(c):
                if word not in chord_vocab:
                    chord_vocab[word] = len(chord_vocab) + 1
                chord_ndx[n] = chord_vocab[word]
            np_chord.append(chord_ndx)

    print(len(np_keys), len(np_melody_high))
    np.savez('dataset/enhance_unlabeled.npz', melody_low=np.array(np_melody_low),
        melody_high=np.array(np_melody_high), chord=np.array(np_chord), rhy_ptn=np.array(np_rhy_ptn),
        chord_color=np.array(np_chord_color), struct=np.array(np_struct), contour=np.array(np_contour),
        keys=np.array(np_keys))

    # Handle labeled data
    np_emotion = []
    np_melody_high = []
    np_melody_low = []
    np_chord = []
    np_rhy_ptn = []
    np_chord_color = []
    np_contour = []
    np_struct = []
    np_keys = []
    for lf in tqdm(labeled_files):
        emo_av, keys, melody_low, melody_high, chord, _, music_feat = read_txt(lf, True)
        np_keys.extend(keys)
        # Sequence Emotion
        for emo in emo_av:
            if type(emo) == tuple:
                emo = [emo]
            seq_emo = []
            for i in range(len(emo)):
                valence = emo[i][0][0]
                arousal = emo[i][0][1]
                duration = emo[i][1]
                seq_emo.extend([[valence, arousal] for _ in range(duration)])
            np_emotion.append(seq_emo)
        # music features
        for mf in music_feat:
            rhy_ptn, chord_color, contour, struct = handle_music_feat(mf)
            np_rhy_ptn.append(rhy_ptn)
            np_chord_color.append(chord_color)
            np_contour.append(contour)
            np_struct.append(struct)

        # # 2d-high melody to 1d
        # melody_high_1d = []
        # for melodys in melody_high:
        #     tmp = []
        #     for m in melodys:
        #         tmp.extend([m[0]] * m[1])
        #     melody_high_1d.append(tmp)  
        # melody_high = melody_high_1d
        # To ndx
        melody_low_ndx = []
        melody_high_ndx = []
        chord_ndx = []

        for m in melody_low:
            melody_low_ndx = []
            for word in m:
                if word not in melody_vocab:
                    melody_vocab[word] = len(melody_vocab) + 1
                melody_low_ndx.append(melody_vocab[word])
            np_melody_low.append(melody_low_ndx)

        for m in melody_high:
            melody_high_ndx = np.zeros((64, ))
            for n, word in enumerate(m):
                if word not in note_vocab:
                    note_vocab[word] = len(note_vocab) + 1
                melody_high_ndx[n] = note_vocab[word]
            np_melody_high.append(melody_high_ndx)

        for c in chord:
            chord_ndx = np.zeros((16, ))
            for n, word in enumerate(c):
                if word not in chord_vocab:
                    chord_vocab[word] = len(chord_vocab) + 1
                chord_ndx[n] = chord_vocab[word]
            np_chord.append(chord_ndx)
    
    print(len(np_melody_high), len(np_chord))
    np.savez('dataset/enhance_labeled.npz', melody_low=np.array(np_melody_low), emotion=np.array(np_emotion),
        melody_high=np.array(np_melody_high), chord=np.array(np_chord), rhy_ptn=np.array(np_rhy_ptn),
        chord_color=np.array(np_chord_color), struct=np.array(np_struct), contour=np.array(np_contour),
        keys=np.array(np_keys))

    with open('dataset/enhance_chord_vocab.json', 'w') as f:
        chord_vocab['S'] = len(chord_vocab) + 1
        chord_vocab['E'] = chord_vocab['S'] + 1
        chord_vocab['P'] = 0
        json.dump(chord_vocab, f)

    with open('dataset/enhance_melody_vocab.json', 'w') as f:
        melody_vocab['S'] = len(melody_vocab) + 1
        melody_vocab['E'] = melody_vocab['S'] + 1
        melody_vocab['P'] = 0
        json.dump(melody_vocab, f)   
    
    with open('dataset/enhance_notes_vocab.json', 'w') as f:
        note_vocab['S'] = len(note_vocab) + 1
        note_vocab['E'] = note_vocab['S'] + 1
        note_vocab['P'] = 0
        json.dump(note_vocab, f)

def read_txt(pth, is_labeled=True):
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
        for n, i in enumerate(f.readlines()):
            datas = i.strip('\n').split('|')
            if is_labeled:
                for n, d in enumerate(datas):
                    # melody high
                    if n == 3:
                        datas[n] = d.replace('),', '), ')
                    elif n == 1:
                        continue
                    else:
                        datas[n] = d.replace(',', ', ')
                dirty = False
                for m in eval(datas[2]):
                    if m < 0:
                        print('dirty: {}'.format(pth))
                        dirty = True
                        break
                if dirty:
                    continue
                # AV
                if 'NaN' in datas[0]:
                    datas[0] = datas[0].replace('NaN', '0.00')
                    print(pth)
                if '(-' in datas[3]:
                    print('shit')
                    continue
                emo_av.append(eval(datas[0]))
                keys.append(keys2list(datas[1]))
                melody_low.append(eval(datas[2]))

                melody_high.append(eval(datas[3]))
                chord.append(chord_str_to_list(datas[4]))
                # terminate.append(eval(datas[5]))
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
                dirty = False
                for m in eval(datas[1]):
                    if m < 0:
                        print('dirty: {}'.format(pth))
                        dirty = True
                        break
                if dirty:
                    continue
                keys.append(keys2list(datas[0]))
                melody_low.append(eval(datas[1]))
                melody_high.append(eval(datas[2]))
                chord.append(chord_str_to_list(datas[3]))
                # terminate.append(datas[4])
                # Music Features
                music_feat.append(eval(datas[5]))

    return emo_av, keys, melody_low, melody_high, chord, terminate, music_feat

if __name__ == '__main__':
    # args = Config('config_m2c.yaml')
    # args.input_start = 'S'
    # args.output_start = 'E'
    # get_npz_m2c('dataset/data_m2c.txt', 'dataset/melody_to_id.json', 'dataset/chord_to_id.json')
    #get_npz('dataset/data_m2c.txt')
    data2npz("dataset/enhance_cover20230318")
    # get_npz('dataset/labeled/hhr-pmemo_new-范围已修改-txt比情感标记长已解决-最终/piece0_1.txt')