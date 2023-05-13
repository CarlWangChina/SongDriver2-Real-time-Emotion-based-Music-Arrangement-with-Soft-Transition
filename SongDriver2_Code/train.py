import json
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy
from torch.utils.data.dataset import random_split
from config import Config
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.musicDatasetFast import AiMusicDatasetLow
from models.muthera import DualMuThera


SEED = 3547
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_type = 'median'
# only use label
NAME = 'muTheraDual_{}'.format(fusion_type)
LOG_PATH = 'log/{}'.format(NAME)
CKPT_PATH = 'modelzoo/{}'.format(NAME)

if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)
    
if not os.path.exists(CKPT_PATH):
    os.mkdir(CKPT_PATH)

def pipeline(model, train_loader, eval_loader, test_loader, args, ckpt_pth=None):
    model = model.to(device)
    start_epoch = 0
    if ckpt_pth:
        ckpt = torch.load(ckpt_pth, map_location=device)
        start_epoch = int(ckpt_pth.split('/')[-1].split('.')[0]) + 1
        model.load_state_dict(ckpt)
        print('[Resume] load ckpt {}'.format(ckpt_pth))

    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=1e-5)
    postfix = {'running_loss': 0.0}
    for epoch in range(start_epoch, args.epoch):
        tqdm_train = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch + 1))
        train_loss = 0
        train_chord_loss = 0
        train_note_loss = 0
        n = 0
        # train
        pre_output = None
        for batch in tqdm_train:
            # To device
            for key in batch:
                batch[key] = batch[key].to(device)
            # Load data
            batch_size = batch['melody'].shape[0]
            outputs = model(batch, pre_output, alpha=epoch / args.epoch)
            # generating loss
            g_chord_loss = criterion(outputs['chord'], batch['chord_out'].view(-1))
            g_note_loss = criterion(outputs['note'], batch['note_out'].view(-1))
            loss = g_note_loss + g_chord_loss
            # Record Loss
            train_chord_loss += g_chord_loss.item() * batch_size
            train_note_loss += g_note_loss.item() * batch_size
            train_loss += loss.item() * batch_size
            n += batch_size
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (epoch + 1)
            tqdm_train.set_postfix(postfix)
            # record out for T+1
            pre_output = outputs
        scheduler.step()
        # evaluate
        eval_loss, eval_note_loss, eval_chord_loss = evaluate(model, eval_loader)
        # test
        chord_acc, note_acc = test(model, test_loader)
        res_json = {
            'epoch': epoch,
            'alpha': epoch / args.epoch,
            'chord_loss': train_chord_loss / n,
            'note_loss': train_note_loss / n,
            'train_loss': train_loss / n,
            'eval_chord_loss': eval_chord_loss,
            'eval_note_loss': eval_note_loss,
            'eval_loss': eval_loss,
            'test_accuracy_chord': chord_acc.compute().item(),
            'test_accuracy_note': note_acc.compute().item(),
        }
        with open('{}/epochs.json'.format(LOG_PATH), 'a') as f:
            res = json.dumps(res_json)
            f.write(res + '\n')
        # scheduler.step()
        torch.save(model.state_dict(), f'{CKPT_PATH}/{epoch}.ckpt')

@torch.no_grad()
def evaluate(model, loader):
    global device
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epoch, eta_min=0.0001)
    eval_loss = 0
    eval_chord_loss = 0
    eval_note_loss = 0
    n = 0
    pre_output = None
    for batch in tqdm(loader):
         # To device
        for key in batch:
            batch[key] = batch[key].to(device)
        # Load data
        batch_size = batch['melody'].shape[0]
        outputs = model(batch, pre_output)
        # generating loss
        g_chord_loss = criterion(outputs['chord'], batch['chord_out'].view(-1))
        g_note_loss = criterion(outputs['note'], batch['note_out'].view(-1))
        loss = g_note_loss + g_chord_loss
        # Record Loss
        eval_chord_loss += g_chord_loss.item() * batch_size
        eval_note_loss += g_note_loss.item() * batch_size
        eval_loss += loss.item() * batch_size
        n += batch_size
        # record out for T+1
        pre_output = outputs

    return eval_loss / n, eval_note_loss / n, eval_chord_loss / n

@torch.no_grad()
def test(model, loader):
    global device
    chord_acc = Accuracy(task='multiclass', num_classes=17327).to(device)
    note_acc = Accuracy(task='multiclass', num_classes=113).to(device)

    pre_output = None
    for batch in tqdm(loader):
         # To device
        for key in batch:
            batch[key] = batch[key].to(device)
        # Load data
        batch_size = batch['melody'].shape[0]
        outputs = model(batch, pre_output)
        # generating loss
        g_chord = outputs['chord']
        g_notes = outputs['note']

        _ = chord_acc(g_chord, batch['chord_out'].view(-1))
        _ = note_acc(g_notes, batch['note_out'].view(-1))

    return chord_acc, note_acc

if __name__ == '__main__':
    args = Config('config_muthera.yaml')
    args.input_start = 'S'
    args.output_start = 'E'
    args.batch_size = 16
    print(f"batch_size:{args.batch_size}")
    full_dataset = AiMusicDatasetLow('dataset/labeled.npz', labeled=True)
    data_len = len(full_dataset)
    train_set, test_set, eval_set = random_split(full_dataset, [data_len - 2000, 1000, 1000], generator=torch.Generator().manual_seed(SEED))

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_set, args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, drop_last=True)
    
    print('Dataloader done.')

    model = DualMuThera(emo_ckpt_pth=None,emo_fusion_type=fusion_type).to(device)
    pipeline(model, train_loader, eval_loader, test_loader, args)
