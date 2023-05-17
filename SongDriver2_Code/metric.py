import torch

# 
def get_emo_consistency(seq_emotion: torch.Tensor):
    # seq_emotion: [num_click, 16, 2]
    res_seq_emotion = torch.zeros_like(seq_emotion)
    res_seq_emotion[1:] = seq_emotion[:-1]
    l2_dist = torch.sqrt(
            torch.sum(
                (seq_emotion - res_seq_emotion) ** 2,
                dim=-1
            )
        )
    l2_dist = l2_dist[1:].mean(dim=-1)
    return l2_dist.mean(), l2_dist.std()

def get_emo_fitness(seq_emotion: torch.Tensor, pred_emotion: torch.Tensor):
    # seq_emotion: [num_click, 16, 2]
    if pred_emotion.shape[0] != seq_emotion.shape[0]:
        print('Error: pred_emotion and seq_emotion mismatched!')
        exit(0)

    l2_dist = torch.sqrt(
        torch.sum(
            (seq_emotion - pred_emotion) ** 2,
            dim=-1
        )
    )

    l2_dist = l2_dist.mean(dim=-1)
    return l2_dist.mean(), l2_dist.std()

def get_similarity_melody(melody_A, melody_B):
    score = 0
    assert melody_A.shape == melody_B.shape
    
    interval = abs(melody_A - melody_B)
    interval = interval % 12
    
    score = (interval == 0).sum()
    
    total = melody_A.shape[0] * melody_A.shape[1]
    return score / total

    