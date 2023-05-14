def flag_str_to_list(flag_str):
    flag_lst = []
    for i in flag_str:
        flag_lst.append(eval(i))
    return flag_lst

def melody_str_to_int_list(melody_str):
    melody_lst = []
    for i in melody_str:
        full_str = [int(x) for x in i.lstrip('[').rstrip(']').split(', ')]
        melody_lst.append(full_str)
    return melody_lst

def melody_str_to_list(melody_str):
    melody_lst = []
    for i in melody_str:
        full_str = i.lstrip('[').rstrip(']').split(', ')
        full_str = ['0' + i if len(i) == 1 else i for i in full_str]
        sentence = []
        # print(f'full_str:{full_str}')
        for ele in [''.join(full_str[i:i + 1]) for i in range(0, len(full_str), 1)]:
            # print(f"ele:{ele}")
            sentence.append(ele)
        melody_lst.append(sentence)
    return melody_lst

def tunple_str_to_list(raw_strs):
    tunples = []
    type_f = lambda x: float(x) if float(x) - int(float(x)) != 0 else int(float(x))
    for raw_s in raw_strs:
        raw_tunples = raw_s.lstrip('[').rstrip(']').lstrip('(').rstrip(')').split('),')
        elements = []
        for s in raw_tunples:
            split_s = s.lstrip('(').rstrip(')').split(',')
            for e in split_s:
                e = e.lstrip(' ((')
                elements.append(type_f(e))
        tunples.append(elements)
    return tunples

def emotion_to_list(melody_str):
    melody_lst = []
    for i in melody_str:
        full_str = i.lstrip('[').rstrip(']').split(', ')
        full_str = ['0' + i if len(i) == 1 else i for i in full_str]
        sentence = []
        # print(f'full_str:{full_str}')
        for ele in [''.join(full_str[i:i + 1]) for i in range(0, len(full_str), 1)]:
            # print(f"ele:{ele}")
            sentence.append(ele)
        melody_lst.append(sentence)
    return melody_lst

def chord_str_to_list(chord_str):
    chord_lst = []
    full_str = chord_str.lstrip('[[').rstrip(']]').split('], [')
    sentence = []
    for ele in full_str:
        ele = '[' + ele + ']'
        sentence.append(ele)
    chord_lst.extend(sentence)
    return chord_lst

def make_dict(seq_lst):
    unique_lst = []
    for i in seq_lst:
        unique_lst.extend(i)
    unique_lst = np.unique(np.sort(unique_lst))
    seq_to_id = {ele: i + 1 for i, ele in enumerate(unique_lst)}
    # seq_to_id = sorted(seq_to_id.keys())
    return seq_to_id

def notes_str_to_list(notes_str):
    notes_lst = []
    for i in notes_str:
        full_str = i.lstrip('[').rstrip(']').split(', ')
        full_str = ['0' + i if len(i) == 1 else i for i in full_str]
        sentence = []
        # print(f'full_str:{full_str}')
        for ele in [''.join(full_str[i:i + 1]) for i in range(0, len(full_str), 1)]:
            # print(f"ele:{ele}")
            sentence.append(ele)
        notes_lst.append(sentence)
    # print(f'melody_list: {melody_lst}')
    return notes_lst