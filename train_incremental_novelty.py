import json
import os
import sys
import argparse
import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np
from datetime import datetime
import random
from itertools import cycle

# Novelty modules
from dataloader_novelty import NoveltyLoader, exemplarLoaderNovelty, collate_fn_novelty
from audio_visual_model_novelty import NoveltyAudioVisualNet, info_nce_loss
from memory_novelty import update_exemplar_set_with_prototypes

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def CE_loss(step_out_ans_num, logits, label):
    targets = F.one_hot(label, num_classes=step_out_ans_num)
    loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))
    return loss

def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def train(args, task, step, train_set, val_set, exemplar_set):
    train_loader = DataLoader(train_set, batch_size=args.train_batch_size, num_workers=args.num_workers,
                               pin_memory=True, drop_last=True, shuffle=True, collate_fn=collate_fn_novelty)
    val_loader = DataLoader(val_set, batch_size=args.infer_batch_size, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False, shuffle=False, collate_fn=collate_fn_novelty)

    step_out_ans_num = train_set.num_current_step_ans
    
    if task == 0 and step == 0:
        model = NoveltyAudioVisualNet(args, step_out_ans_num).to(device)
    else:
        # Load previous best model's state_dict
        prev_task = task if step > 0 else task - 1
        prev_step = step - 1 if step > 0 else 5
        
        # Get previous answer count to instantiate model correctly
        prev_train_set = NoveltyLoader(args, mode='train')
        prev_train_set.set_incremental_step(prev_task, prev_step)
        prev_ans_num = prev_train_set.num_current_step_ans
        
        model = NoveltyAudioVisualNet(args, prev_ans_num).to(device)
        model_path = './save_novelty/task_{}_step_{}_best_model.pth'.format(prev_task, prev_step)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        # Expand for current step
        model.incremental_classifier(step_out_ans_num)
        model = model.to(device)
        
        exemplar_loader = DataLoader(exemplar_set, batch_size=min(args.exemplar_batch_size, len(exemplar_set)), 
                                     num_workers=args.num_workers, pin_memory=True, drop_last=True, shuffle=True, collate_fn=collate_fn_novelty)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_res = 0.0
    for epoch in range(args.max_epoches):
        model.train()
        train_loss = 0.0
        num_steps = 0
        
        if task == 0 and step == 0:
            iterator = tqdm(train_loader, desc=f"Task {task} Step {step} Epoch {epoch}")
        else:
            iterator = tzip(train_loader, cycle(exemplar_loader), desc=f"Task {task} Step {step} Epoch {epoch}")

        for samples in iterator:
            if task == 0 and step == 0:
                audio, visual, ques, attn_mask, answers, que_ids, labels = samples
                audio, visual, ques, attn_mask, answers = audio.to(device), visual.to(device), ques.to(device), attn_mask.to(device), answers.to(device)
                
                out, proj_a, proj_v = model(audio=audio, visual=visual, question=ques, attention_mask=attn_mask, out_contrastive=True)
                
                l_cls = CE_loss(step_out_ans_num, out, answers)
                l_cont = info_nce_loss(proj_a, proj_v) if args.use_contrastive_loss else 0.0
                
                loss = l_cls + args.lambda_contrast * l_cont
            else:
                curr, prev = samples
                audio, visual, ques, attn_mask, answers, que_ids, labels = curr
                e_audio, e_visual, e_ques, e_attn, e_answers, e_ids, e_labels = prev
                
                t_audio = torch.cat([audio, e_audio]).to(device)
                t_visual = torch.cat([visual, e_visual]).to(device)
                t_ques = torch.cat([ques, e_ques]).to(device)
                t_attn = torch.cat([attn_mask, e_attn]).to(device)
                t_answers = torch.cat([answers, e_answers]).to(device)
                
                out, proj_a, proj_v = model(audio=t_audio, visual=t_visual, question=t_ques, attention_mask=t_attn, out_contrastive=True)
                
                # Base classification loss
                l_cls = CE_loss(step_out_ans_num, out, t_answers)
                
                l_cont = info_nce_loss(proj_a, proj_v) if args.use_contrastive_loss else 0.0
                loss = l_cls + args.lambda_contrast * l_cont 
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            num_steps += 1
            
        print(f"Epoch {epoch} Loss: {train_loss / num_steps:.4f}")
        
        # Validation
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for v_samples in tqdm(val_loader, desc="Validating"):
                va, vv, vq, vm, vans, vids, vlbs = v_samples
                va, vv, vq, vm = va.to(device), vv.to(device), vq.to(device), vm.to(device)
                vo = model(audio=va, visual=vv, question=vq, attention_mask=vm)
                val_acc += top_1_acc(vo.cpu(), vans)
        val_acc /= len(val_loader)
        print(f"Epoch {epoch} Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_res:
            best_val_res = val_acc
            os.makedirs('./save_novelty', exist_ok=True)
            torch.save(model.state_dict(), './save_novelty/task_{}_step_{}_best_model.pth'.format(task, step))

    # Task/Step completion: Update memory buffer with prototypes
    if args.use_prototype_memory:
        update_exemplar_set_with_prototypes(args, model, train_set, exemplar_set, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--infer_batch_size', type=int, default=16)
    parser.add_argument('--max_epoches', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4) # Low LR for transformers
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--memory_size', type=int, default=700)
    parser.add_argument('--use_transformer_encoders', type=bool, default=True)
    parser.add_argument('--use_contrastive_loss', type=bool, default=True)
    parser.add_argument('--use_adapters', type=bool, default=True)
    parser.add_argument('--use_prototype_memory', type=bool, default=True)
    parser.add_argument('--lambda_contrast', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--video_dir', type=str, default='/mnt/raid/obed/Speech/MUSIC_AVQA_videos/MUSIC-AVQA-videos-Real')
    
    args = parser.parse_args()
    setup_seed(42)
    
    train_set = NoveltyLoader(args, mode='train')
    val_set = NoveltyLoader(args, mode='val')
    exemplar_set = exemplarLoaderNovelty(args)
    
    for task in range(5):
        for step in range(6):
            print(f"Starting Task {task} Step {step}")
            train_set.set_incremental_step(task, step)
            val_set.set_incremental_step(task, step)
            train(args, task, step, train_set, val_set, exemplar_set)
