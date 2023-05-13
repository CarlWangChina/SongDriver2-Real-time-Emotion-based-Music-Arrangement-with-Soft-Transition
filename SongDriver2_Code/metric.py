import torch
import urllib
import torch.nn.functional as F


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


'''
@param fine_grained: True 1拍为单位; False 4小节为单位
@param t_melody / g_melody: [XX(batch_size), 16(src_len), 256embed_dim]
'''
def get_similarity_symbol(t_melody: torch.Tensor, g_melody: torch.Tensor, fine_grained=True):
    # [batch_size, 16 * embeddim]
    if fine_grained:
        t_melody = t_melody.flatten(start_dim=0, end_dim=1)
        g_melody = g_melody.flatten(start_dim=0, end_dim=1)
    else:
        t_melody = t_melody.flatten(start_dim=1)
        g_melody = g_melody.flatten(start_dim=1)
    # compute cosine similarity
    t_melody = F.normalize(t_melody, p=2, dim=1)
    g_melody = F.normalize(g_melody, p=2, dim=1)
    
    cosine = torch.matmul(t_melody, g_melody.T)
    cosine = torch.diag(cosine)
    return cosine.mean().item(), cosine.std().item()


def get_audio_feat(audio_pth, vggish):
    if not vggish:
        vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        vggish = vggish
    # [batch_size, embed_dim]
    audio_feat = vggish(audio_pth).cpu()
    return audio_feat

def edit_distance(X, Y):
    # 动态规划求编辑距离
    m = len(X)
    n = len(Y)
    dp = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def get_melody_similarity(A, B):
    A = A.flatten().cpu().numpy().tolist()
    B = B.flatten().cpu().numpy().tolist()
    # 计算两个旋律序列的相似度
    edit_dist = edit_distance(A, B)
    similarity = 1 - edit_dist / max(len(A), len(B))
    return similarity


def get_similarity_audio(t_audio: str, g_audio: str, vggish=None):
    if type(t_audio) == str:
        t_audio = get_audio_feat(t_audio, vggish)
    if type(g_audio) == str:
        g_audio = get_audio_feat(g_audio, vggish)
    # compute cosine similarity
    t_audio = F.normalize(t_audio, p=2, dim=1)
    g_audio = F.normalize(g_audio, p=2, dim=1)
    cosine = torch.matmul(t_audio, g_audio.T)
    cosine = torch.diag(cosine)
    return cosine.mean().item(), cosine.std().item()

def get_similarity_melody(melody_A, melody_B):
    score = 0
    assert melody_A.shape == melody_B.shape
    
    interval = abs(melody_A - melody_B)
    interval = interval % 12
    
    score = (interval == 0).sum()
    
    total = melody_A.shape[0] * melody_A.shape[1]
    return score / total

if __name__ == '__main__':
    torch.manual_seed(114514)
    print(get_similarity_audio('pop_songs/起风了.txt.mid.wav', 'pop_songs/丑八怪.txt.mid.wav'))
    dummy_seq_emo = torch.randn((15, 16, 2))
    pred_seq_emo = torch.randn((15, 16, 2))
    print(get_emo_consistency(dummy_seq_emo))
    print(get_emo_fitness(dummy_seq_emo, pred_seq_emo))
    
    t_melody = torch.randn((15, 16, 256))
    g_melody = torch.randn((15, 16, 256))
    print(get_similarity_symbol(t_melody, g_melody))
    print(get_similarity_symbol(t_melody, g_melody, False))
    