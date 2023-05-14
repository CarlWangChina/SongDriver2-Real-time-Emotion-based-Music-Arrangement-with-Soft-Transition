import math
import numpy as np
import torch
import json
from collections import deque


with open('dataset/id2note.json', 'r') as f:
    id2note = json.load(f)
    
with open('dataset/id2chord.json', 'r') as f:
    id2chord = json.load(f)
    

class EmoFeatureExtractor:
    def __init__(self) -> None:
        self.high_sampled_cache = deque(maxlen=21)
        self.chord_cache = deque(maxlen=21)
        self.rhythm_cache = deque(maxlen=21)
        self.tonality = []
        self.two_dimention_melody = []
        self.high_sampled_melody = []
        self.chord = []
        self.rhythm_pattern = []
    
    def __call__(self, inputs, contain_S_E=False):
        batch_size = inputs['tone'].shape[0]
        tone = inputs['tone'].cpu().numpy().tolist()
        note_logit = inputs['note']
        notes = []
        _, note_ids = torch.topk(note_logit, k=1)
        for nid in note_ids:
            note = id2note[str(nid.item())]
            if note == 'S' or note == 'E' or note =='P':
                note = '0'
            notes.append(eval(note))

        chord_logit = inputs['chord']
        chords = []
        _, chord_ids = torch.topk(chord_logit, k=1)
        for cid in chord_ids:
            chord = id2chord[str(cid.item())]
            if chord == 'S' or chord == 'E' or chord =='P':
                chord = '[]'
            chords.append(eval(chord))
        batch_music_feats = []
        for i in range(batch_size):
            if contain_S_E:
                offset = 1 if i != 0 else 0
            else:
                offset = 0
            datas = {
                'tone': tone[i],
                'note': notes[i * 256 + offset: (i + 1) * 256 + offset],
                'chord': chords[i * 16 + offset: (i + 1) * 16 + offset]
            }
            self.input_data(datas)
            music_feat = self.get_all_features()
            batch_music_feats.append(music_feat)
        return torch.stack(batch_music_feats, dim=0)

    def get_all_features(self):
        rhy_ptn = torch.Tensor(self.get_rhythm_pattern())
        chord_color = self.get_chord_color()
        flat_chord_color = self.flatten2d(chord_color)
        chord_color = torch.Tensor(flat_chord_color)
        contour = torch.Tensor(self.get_music_contour())
        struct = torch.Tensor(self.get_form_structure())
        return torch.cat([rhy_ptn, chord_color, contour, struct], dim=-1)

    def input_data(self, inputs):
        # 直接读取input
        self.tonality = inputs['tone']
        self.chord = inputs['chord']
        #if use 2-d high-melody 
        two_dimention_melody = []
        i = 0
        while i < len(inputs['note']):
            now = inputs['note'][i]
            count = 1
            while(i + 1 < len(inputs['note']) and inputs['note'][i + 1] == inputs['note'][i]):
                i += 1
                count += 1
            two_dimention_melody.append((now, count))
            i += 1
        self.two_dimention_melody = two_dimention_melody
        self.high_sampled_melody = self.flatten2d(self.two_dimention_melody)
        # self.high_sampled_melody = self.two_dimention_melody
        
        # 1-d to rhythm pattern
        count = {}
        self.rhythm_pattern = list(map(lambda x: x[1], self.two_dimention_melody))
        # cached
        self.high_sampled_cache.append(self.high_sampled_melody)
        self.chord_cache.append(self.chord)
        self.rhythm_cache.append(self.rhythm_pattern)

    def flatten2d(self, arr):
        """将第二类和第三类数据中的二维列表展开为一维
        """
        ret = []
        for t in arr:
            ret.extend([t[0]] * t[1])
        return ret

    def get_rhythm_pattern(self):
        ret_rhy_ptn = np.zeros((256, ))
        onset = 0
        for r in self.rhythm_pattern:
            onset += r
            ret_rhy_ptn[onset - 1] = 1
        return ret_rhy_ptn

    def get_music_contour(self):
        '''返回的是一个列表，包含两个元组，第一个元组代表（低音线最低音，低音线首末元素之差，低音线凹凸程度）
        第二个元组代表（高音线最高音，高音线首末元素之差，高音线凹凸程度）'''
        # 二维旋律转换为一维旋律
        melodies_1d = self.high_sampled_melody
        # 最高/低点
        melody_max = np.max(melodies_1d)
        melody_trend = melodies_1d[-1] - melodies_1d[0]
        melody_convex = (sum(melodies_1d) - (melodies_1d[0] + melodies_1d[-1]) * len(melodies_1d) // 2) // len(melodies_1d)

        chord = self.min_note_in_chord_sequence()
        if len(chord) > 0:
            chord_min = min(chord)
            chord_trend = chord[-1] - chord[0]
            chord_convex = (
                sum(chord) - (chord[0] + chord[-1]) * len(chord) // 2) // len(chord)
        else:
            chord_min = 0
            chord_trend = 0
            chord_convex = 0
        return [chord_min, chord_trend, chord_convex, melody_max, melody_trend, melody_convex]

    def get_chord_color(self):        
        cur_list = []
        for i in range(len(self.tonality)):
            if len(self.chord[i]) == 0:
                cur_color = 'undef'
            else:
                cur_ref = self.tonality[i]
                cur_input = self.chord[i]
                cur_color = self.cal_color(cur_ref, cur_input)
            if (len(cur_list) == 0 or cur_color != cur_list[-1][0]):
                cur_list.append([cur_color, 16])
            else:
                cur_list[-1] = [cur_list[-1][0], cur_list[-1][1]+16]

        undef_list = []
        if len(cur_list) == 1 and cur_list[0][0] == 'undef':
            cur_list[0][0] = 0
            return cur_list
        for i in range(len(cur_list)):
            if cur_list[i][0] == 'undef':
                undef_list.append(i)
        if (0 in undef_list):
            cur_list[0][0] = cur_list[1][0]/2
            undef_list.remove(0)
        if (len(cur_list)-1 in undef_list):
            cur_list[-1][0] = cur_list[-2][0]/2
            undef_list.remove(len(cur_list)-1)
        for i in range(len(undef_list)):
            cur_list[undef_list[i]][0] = float(
                cur_list[undef_list[i]-1][0] + cur_list[undef_list[i] + 1][0]) / 2

        ret_list = []
        item = cur_list[0]
        for i in range(1, len(cur_list)):
            if cur_list[i][0] != item[0]:
                ret_list.append(tuple(item))
                item = cur_list[i]
            else:
                item[1] += cur_list[i][1]
        ret_list.append(tuple(item))
        return ret_list

    def get_form_structure(self):
        form_rep_sign, form_rep_pos = self.get_form_rep()
        form_chord_sign = self.get_form_chord()
        form_molody_sign = self.get_form_melody()
        form_zone_sign = self.get_form_zone()
        form_rhythem_sign = self.get_form_rhythm()
        return [form_rhythem_sign, form_chord_sign, form_molody_sign, form_zone_sign, form_rep_sign, form_rep_pos]

    def get_form_rep(self):
        ret_sign = 0
        ret_pos = 0
        for i in range(4):
            cur_melody = self.high_sampled_melody[64 * i: 64 * (i + 1)]
            melody_cache_len = len(self.high_sampled_cache) - 1
            for j in range(melody_cache_len):
                pre_melody = self.high_sampled_cache[melody_cache_len - j - 1]
                for k in range(4):
                    ref_melody = pre_melody[64*k:64*(k+1)]
                    cnt = 0
                    for m in range(64):
                        if (ref_melody[m] == cur_melody[m]):
                            cnt += 1
                    ratio = cnt/64
                    if ratio > 0.9:
                        ret_sign = 1
                        ret_pos = j * 4 - k + i + 4
                        return (ret_sign, ret_pos)
        return (ret_sign, ret_pos)

    def get_form_rhythm(self):
        ret_sign = 0
        cur_rhythm = self.rhythm_pattern
        rhythm_cache_len = len(self.rhythm_cache)-1
        for j in range(rhythm_cache_len):
            pre_rhythm = self.rhythm_cache[rhythm_cache_len-j-1]
            union_len = len(set(self.high_sampled_cache[rhythm_cache_len - j - 1]) & set(
                self.high_sampled_cache[rhythm_cache_len-j-1]))
            join_len = len(set(self.high_sampled_cache[rhythm_cache_len - j - 1]) | set(
                self.high_sampled_cache[rhythm_cache_len-j-1]))
            same_ratio = union_len / join_len
            if same_ratio > 0.9:
                if (self.edit_distance(cur_rhythm, pre_rhythm) > 0.75):
                    ret_sign = 1
                    return ret_sign
        return ret_sign

    def get_form_melody(self):
        ret_sign = 0
        for i in range(4):
            cur_melody = self.high_sampled_melody[64*i:64*(i+1)]
            melody_cache_len = len(self.high_sampled_cache)-1
            for j in range(melody_cache_len):
                pre_melody = self.high_sampled_cache[melody_cache_len-j-1]
                for k in range(4):
                    ref_melody = pre_melody[64*k:64*(k+1)]
                    cnt = 0
                    diff_melody = [0]*64
                    for m in range(64):
                        diff_melody[m] = cur_melody[m] - ref_melody[m]
                    common_diff = max(set(diff_melody), key=diff_melody.count)
                    cnt = diff_melody.count(common_diff)
                    ratio = cnt/64
                    # print(ratio, common_diff)
                    if ratio > 0.9:
                        ret_sign = 1
                        return ret_sign
        return ret_sign

    def get_form_zone(self):
        ret_sign = 0
        for i in range(4):
            cur_melody = self.high_sampled_melody[64*i:64*(i+1)]
            melody_cache_len = len(self.high_sampled_cache)-1
            for j in range(melody_cache_len):
                pre_melody = self.high_sampled_cache[melody_cache_len-j-1]
                for k in range(4):
                    ref_melody = pre_melody[64*k:64*(k+1)]
                    cnt = 0
                    diff_melody = [0]*64
                    for m in range(64):
                        diff_melody[m] = cur_melody[m] - ref_melody[m]
                    common_diff = max(set(diff_melody), key=diff_melody.count)
                    cnt = diff_melody.count(common_diff)
                    ratio = cnt/64
                    if ratio > 0.9 and common_diff % 12 == 0:
                        ret_sign = 1
                        return ret_sign
        return ret_sign

    def get_form_chord(self):
        ret_sign = 0
        cur_chord = self.chord
        chord_cache_len = len(self.chord_cache)-1
        for j in range(chord_cache_len):
            pre_chord = self.chord_cache[chord_cache_len-j-1]

            ref_chord = pre_chord
            cnt = 0
            for i in range(16):
                if (len(ref_chord[i]) == len(cur_chord[i])):
                    temp1, temp2 = sorted(ref_chord[i]), sorted(cur_chord[i])
                    z = set()
                    for k in range(len(ref_chord[i])):
                        z.add(temp1[k]-temp2[k])
                    if len(z) == 1:
                        cnt += 1
            ratio = cnt/16
            if ratio > 0.9:
                ret_sign = 1
                return ret_sign
        return ret_sign

    def min_note_in_chord_sequence(self) -> list:
        """和弦序列中最低的音
        """
        ret = []
        for i in range(len(self.chord)):
            if len(self.chord[i]) != 0:
                ret.append(min(self.chord[i]))
            else:
                # temp = list(filter(lambda x : x!=0, self.high_sampled_melody[i*16:(i+1)*16]))
                # if temp:
                #     ret.append(min(temp))
                # else:
                pass
        return ret

    def key_2_ref_chord(self, key):
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

    def cal_note_in_circle(self, note_list):
        stage = (min(note_list) // 12) * 12
        # note_list = note_list - stage
        note_list = list(map(lambda x: x - stage, note_list))
        ans = []
        for i in note_list:
            if i % 12 == 0:
                cur_note = 0
            elif i % 12 == 1:
                cur_note = 7
            elif i % 12 == 2:
                cur_note = 2
            elif i % 12 == 3:
                cur_note = 9
            elif i % 12 == 4:
                cur_note = 4
            elif i % 12 == 5:
                cur_note = -1
            elif i % 12 == 6:
                cur_note = 6
            elif i % 12 == 7:
                cur_note = 1
            elif i % 12 == 8:
                cur_note = 8
            elif i % 12 == 9:
                cur_note = 3
            elif i % 12 == 10:
                cur_note = 10
            elif i % 12 == 11:
                cur_note = 5
            ans.append(cur_note)
        return ans

    def cal_KD(self, note_list):
        n = len(note_list)
        note_list = self.cal_note_in_circle(note_list)
        s = sum(note_list)
        ans = s / n
        if -6 <= ans <= 6:
            return ans
        elif ans < -6:
            while (ans < -6):
                ans += 12
            return ans
        else:
            while (ans > 6):
                ans -= 12
            return ans

    def cal_color(self, ref_chord, input_chord):
        if len(input_chord) == 0:
            return 'No chord'
        kd1 = self.cal_KD(ref_chord)
        kd2 = self.cal_KD(input_chord)
        if (kd2 - kd1 > 0):
            sgn = 1
        elif (kd2 - kd1 == 0):
            sgn = 0
        else:
            sgn = -1

        s = 0
        ref_chord = self.cal_note_in_circle(ref_chord)
        input_chord = self.cal_note_in_circle(input_chord)
        for i in ref_chord:
            for j in input_chord:
                s += abs(i-j)

        ans = sgn * 2 / math.pi * np.arctan(s/54)
        return ans*100

    def edit_distance(self, list1, list2):
        m, n = len(list1), len(list2)
        dp = np.zeros((m+1, n+1))
        # if condition == 'rhythem':
        #     weight = 32
        # else:
        #     weight = 6
        weight = 16
        for i in range(m+1):
            for j in range(n+1):
                if i == 0 and j == 0:
                    dp[i][j] = 0
                elif i == 0:
                    dp[i][j] = dp[i][j-1] + min(weight, list2[j-1])
                elif j == 0:
                    dp[i][j] = dp[i-1][j] + min(weight, list1[i-1])
                elif list1[i-1] == list2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i][j-1]+weight, dp[i-1]
                                   [j]+weight, abs(i-j)+dp[i-1][j-1])
        return int(dp[m][n])/(weight*(m+n))


if __name__ == '__main__':

    # str1 = '[((-0.01,-0.07),256)]|[(c#.minor,236),(d.minor,20)]|[0,0,0,0,76,76,76,76,78,78,78,78,81,81,81,81,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69,69]|[(0,16),(76,16),(78,16),(81,16),(80,128),(69,64)]|[[],[49],[51],[52],[53],[53],[53],[53],[53],[53],[53],[53],[45],[45],[45],[45]]|[64]'
    # str2 = '[((-0.01,-0.07),256)]|[(d.minor,4),(f.minor,72),(e.minor,100),(f.minor,80)]|[65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,65,67,67,0,0,71,71,71,71,71,71,71,71,0,0,0,0,76,76,76,76,78,78,78,78,81,81,81,81,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80,80]|[(65,64),(67,8),(0,8),(71,32),(0,16),(76,16),(78,16),(81,16),(80,80)]|[[41],[41],[41],[41],[43],[47],[47],[],[49],[51],[52],[53],[53],[53],[53],[59]]|[128,176]'
    # test.input_data(str1)
    # test.input_data(str2)
    fileHandler = open("dataset/VGMIDI/raw1/delta-1/piece3_30.txt",  "r")
#    Get list of all lines in file
    listOfLines = fileHandler.readlines()
    # Close file
    test_local_var = True
    test_rhythm_pattern = False
    test_music_contour = False
    test_chord_color = False
    test_form_structure = False
    test_all_features = True
    test = EmoFeatureExtractor()

    for n, line in enumerate(listOfLines):
        print(n, line)
        test.input_data(line)
        if test_local_var:
            print(test.va_time,
                  test.tonality,
                  test.low_sampled_melody,
                  test.two_dimention_melody,
                  test.chord,
                  test.termination)

        if test_rhythm_pattern:
            print(test.get_rhythm_pattern())

        if test_music_contour:
            print(test.get_music_contour())

        if test_chord_color:
            print(test.get_chord_color())

        if test_form_structure:
            print(test.get_form_structure())

        if test_all_features:
            print(test.get_all_features())
