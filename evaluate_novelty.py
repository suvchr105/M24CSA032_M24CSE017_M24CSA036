import json
import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader_novelty import NoveltyLoader, collate_fn_novelty
from audio_visual_model_novelty import NoveltyAudioVisualNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def top_1_acc(logits, target):
    top1_res = logits.argmax(dim=1)
    top1_acc = torch.eq(target, top1_res).sum().float() / len(target)
    return top1_acc.item()

def evaluate_model(model, test_loader):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for audio, visual, ques, mask, labels, *_ in tqdm(test_loader):
            audio, visual, ques, mask = audio.to(device), visual.to(device), ques.to(device), mask.to(device)
            logits = model(audio=audio, visual=visual, question=ques, attention_mask=mask)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return top_1_acc(all_logits, all_labels), all_logits, all_labels

def main(args):
    # This script assumes models are saved for each task-step
    total_tasks = 5
    total_steps = 6
    
    # Initialize variables to track accuracy
    acc_matrix = np.zeros((total_tasks * total_steps, total_tasks * total_steps))
    
    test_set = NoveltyLoader(args, mode='test')
    
    # In incremental training, we usually evaluate the best model at current step
    # on ALL previously seen data.
    
    for t in range(total_tasks):
        for s in range(total_steps):
            current_idx = t * total_steps + s
            model_path = f'./save_novelty/task_{t}_step_{s}_best_model.pth'
            if not os.path.exists(model_path): continue
            
            print(f"Evaluating Model: Task {t} Step {s}")
            
            # Instantiate model with correct answer count for this step
            test_set.set_incremental_step(t, s)
            ans_num = test_set.num_current_step_ans
            model = NoveltyAudioVisualNet(args, ans_num).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            
            # For each seen step i, evaluate
            for i in range(current_idx + 1):
                task_i = i // total_steps
                step_i = i % total_steps
                
                # Setup test data for step i
                test_set.set_incremental_step(task_i, step_i)
                test_loader = DataLoader(test_set, batch_size=args.infer_batch_size, collate_fn=collate_fn_novelty)
                
                acc, _, _ = evaluate_model(model, test_loader)
                acc_matrix[current_idx, i] = acc
                
    # Compute Metrics
    ma = np.mean([acc_matrix[i, i] for i in range(total_tasks * total_steps)]) # Simplified MA
    # AF = Mean of (Acc_at_step_i - Acc_at_final_step) for each step i
    # Actually, AF is typically computed at the end
    last_idx = total_tasks * total_steps - 1
    af = np.mean([acc_matrix[i, i] - acc_matrix[last_idx, i] for i in range(last_idx)])
    
    print(f"\nFinal Results:")
    print(f"Mean Accuracy (MA): {ma:.4f}")
    print(f"Average Forgetting (AF): {af:.4f}")
    
    with open('novelty_performance_results.json', 'w') as f:
        json.dump({'acc_matrix': acc_matrix.tolist(), 'MA': ma, 'AF': af}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_batch_size', type=int, default=16)
    parser.add_argument('--video_dir', type=str, default='/mnt/raid/obed/Speech/MUSIC_AVQA_videos/MUSIC-AVQA-videos-Real')
    args = parser.parse_args()
    main(args)
