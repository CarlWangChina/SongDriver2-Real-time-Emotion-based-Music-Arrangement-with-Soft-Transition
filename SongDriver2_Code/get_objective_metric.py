import torch
import json
import torch.nn.functional as F
import numpy as np
import os
from utils import keys2list
from tqdm import tqdm
from metric import get_emo_consistency, get_emo_fitness, get_similarity_melody
from models.music_feature_extractor import EmoFeatureExtractor
from models.emo_model import EmoModelTest

device = 'cpu'

# Audio Model
# vggish = torch.hub.load('harritaylor/torchvggish', 'vggish', device=device, postprocess=False)
# vggish = vggish.to(device).eval()

# for p in vggish.parameters():
#     p.requires_grad = False

# Emotion Model
emo_ckpt = torch.load('modelzoo/emoModel13.ckpt', map_location=device)
emo_model = EmoModelTest().to(device).eval()
emo_model.load_state_dict(emo_ckpt['model'], strict=True)
for p in emo_model.parameters():
    p.requires_grad = False

print('EmoModel loaded from Epoch {}'.format(emo_ckpt['epoch']))

# normal
id2notes = json.load(open('dataset/id2note.json', 'r'))
id2chords = json.load(open('dataset/id2chord.json', 'r'))

notes2id = json.load(open('dataset/notes_vocab.json', 'r'))
melody2id = json.load(open('dataset/melody_vocab.json', 'r'))
chord2id = json.load(open('dataset/chord_vocab.json', 'r'))


key_dict = {
    '丑八怪': '[(C.MAJOR, 256)]', '南山南': '[(C.MAJOR, 256)]',
    '起风了': '[(C.MAJOR, 256)]', '十年': '[(C.MAJOR, 256)]',
    '送别': '[(C.MAJOR, 256)]', '小幸运': '[(C.MAJOR, 256)]',
}

def get_emotion(name):
    def obj_hook(lst):
        result = []
        for _, val in lst:
            val = np.array(val)
            result.append(val)
        return np.array(result)
    json_root = 'testdatas/emo_seq'
    json_path = os.path.join(json_root, name + '.json')
    with open(json_path, 'r') as f:
        emo_seq = json.load(f, object_pairs_hook=obj_hook)

    emo_seq = torch.Tensor(emo_seq)
    return emo_seq

def read_txt(txt_pth):
    music_feat_extractor = EmoFeatureExtractor(id2notes, id2chords)
    with open(txt_pth, 'r') as f:
        lines = f.readlines()
    seq_melody = []
    seq_chord = []

    for n in key_dict.keys():
        if n in txt_pth:
            music_name = n
            break
        
    key = torch.Tensor(keys2list('[(C.MAJOR, 256)]')).unsqueeze(0).tile(15, 1, 1)

    for l in lines:
        tempo = l.split('|')
        melody = eval(tempo[0])
        chord = tempo[1].strip()
        if chord not in chord2id:
            chord = '[]'
        seq_melody.extend([notes2id[str(m)] for m in melody])
        seq_chord.append(chord2id[chord])

    seq_melody = torch.LongTensor(seq_melody).reshape(15, -1)
    seq_chord = torch.LongTensor(seq_chord).reshape(15, -1)
    music_feat = music_feat_extractor(
        inputs={'tone': key, 'note': seq_melody, 'chord': seq_chord},
        is_logit=False, contain_S_E=False
    )
    outs = {
        'note': seq_melody,
        'chord': seq_chord,
        'music_feat': music_feat
    }
    return outs

def recursive_get_txt_path(root_path):
    import os
    txt_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.txt') and '原曲' not in root:
                txt_paths.append(os.path.join(root, file))
    return txt_paths

def get_metrics(txt_path, ground_truth):
    datas = read_txt(txt_path)
    for key in datas:
        datas[key] = datas[key].to(device)
    music_name = None
    # Get music name
    for k in key_dict.keys():
        if k in txt_path:
            music_name = k
            break
    
    # audio_path = os.path.join('generate_wavs', txt_path.split('/')[-2], txt_path.split('/')[-1].split('.')[0] + '.wav')
    # g_audio = vggish(audio_path)
    # t_audio = ground_truth[music_name]['audio']
    g_melody = datas['note']
    pred_emotion, _ = emo_model(datas)
    pred_emotion = pred_emotion.cpu().detach()
    # true_emotion = ground_truth[music_name]['emotion']
    emo_name = txt_path.split('/')[-2].split('_')[-1]
    true_emotion = get_emotion(emo_name)

    emo_consistency, _ = get_emo_consistency(pred_emotion)
    emo_fitness, _ = get_emo_fitness(pred_emotion, true_emotion)
    if music_name:
        origin_emotion = ground_truth[music_name]['emotion']
        emo_origin, _ = get_emo_fitness(pred_emotion, origin_emotion)
    else:
        emo_origin = 0
    if music_name:
        t_melody = ground_truth[music_name]['note']
        similarity_symbol = get_similarity_melody(t_melody, g_melody)
    else:
        similarity_symbol = 0
    del datas
    return emo_consistency, emo_fitness, emo_origin, similarity_symbol

def get_audio_pth(txt_path):
    file_name = txt_path.split('/')[-1].split('.')[0]
    t_wav_path = 'hotsongs_wavs/' + file_name + '.wav'
    return t_wav_path

def get_ground_truth(root_pth):
    txt_paths = recursive_get_txt_path(root_pth)
    wav_paths = [t.replace('.txt', '.wav') for t in txt_paths]
    ground_truth = {}
    for i in range(len(txt_paths)):
        new_dic = {}
        datas = read_txt(txt_paths[i])
        for key in datas:
            datas[key] = datas[key].to(device)
        emotion, _ = emo_model(datas)
        new_dic['emotion'] = emotion.cpu().detach()
        new_dic['note'] = datas['note']
        # new_dic['audio'] = get_audio_feat(wav_paths[i], vggish)
        # find music name
        for k in key_dict.keys():
            name = None
            if k in txt_paths[i]:
                name = k
                break
        ground_truth[k] = new_dic
        del datas

    del wav_paths
    del txt_paths
    return ground_truth
        
# get metcis for all txt files
def get_all_metrics(txt_root_path):
    # Get ground Truth
    ground_truth = get_ground_truth('hotsongs_wavs')
    # calculate metrics
    file_tree = get_file_tree(txt_root_path)

    results = {}
    for key in tqdm(file_tree.keys()):
        musics = file_tree[key]
        for music_name in musics.keys():
            for txt_pth in musics[music_name]:
                method = txt_pth.split('/')[-1].split('.')[0]
                emo_consistency_, emo_fitness_, emo_origin, similarity_symbol_ = get_metrics(txt_pth, ground_truth)
                if method not in results:
                    results[method] = {}
                if key not in results[method]:
                    results[method][key] = {
                        '连贯距离': [],
                        '目标情感距离': [],
                        '原曲情感距离': [],
                        '原曲相似度': []
                    }

                results[method][key]['连贯距离'].append(emo_consistency_)
                results[method][key]['目标情感距离'].append(emo_fitness_)
                results[method][key]['原曲情感距离'].append(emo_origin)
                results[method][key]['原曲相似度'].append(similarity_symbol_)
    
    metrcis_mean_std = {}
    for method in results.keys():
        metrcis_mean_std[method] = {}
        # calculate mean for each emotion
        emo_consistency = []
        emo_fitness = []
        emo_origin = []
        similarity_symbol = []
        for emo in results[method].keys():
            emo_consistency.append(np.mean(results[method][emo]['连贯距离']))
            emo_fitness.append(np.mean(results[method][emo]['目标情感距离']))
            emo_origin.append(np.mean(results[method][emo]['原曲情感距离']))
            similarity_symbol.append(np.mean(results[method][emo]['原曲相似度']))
    
        metrcis_mean_std[method]['连贯距离'] = {'mean': np.mean(emo_consistency), 'std': np.std(emo_consistency)}
        metrcis_mean_std[method]['目标情感距离'] = {'mean': np.mean(emo_fitness), 'std': np.std(emo_fitness)}
        metrcis_mean_std[method]['原曲情感距离'] = {'mean': np.mean(emo_origin), 'std': np.std(emo_origin)}
        metrcis_mean_std[method]['原曲相似度'] = {'mean': np.mean(similarity_symbol), 'std': np.std(similarity_symbol)}
            
    return metrcis_mean_std
    
def get_file_tree(root_path):
    file_tree = {}
    # loop to build file tree, {emotion: {music_name: [txt_path]}}
    for root, _, files in os.walk(root_path):
        # filter out non-emotion folders``
        if '_' not in root:
            continue
        for file in files:
            # filter out non-txt files and original songs
            if not file.endswith('.txt') or '原曲' in root:
                continue
            name_emo = root.split('/')[-1].split('_')
            name = name_emo[0].strip()
            emo = name_emo[1].strip()
            if emo not in file_tree.keys():
                file_tree[emo] = {}
            if name not in file_tree[emo].keys():
                file_tree[emo][name] = []
            file_tree[emo][name].append(os.path.join(root, file))
    return file_tree

if __name__ == '__main__':
    metrcis_mean_std = get_all_metrics('chcpy/outputs/chordplay/main0407')

    res = metrcis_mean_std
    for key in res.keys():
        for key2 in res[key].keys():
                print('{} {} {} + {}'.format(key, key2, res[key][key2]['mean'], res[key][key2]['std']))