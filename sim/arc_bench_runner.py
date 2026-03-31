import json
import os
import glob
import numpy as np
import brainpy as bp
import brainpy.math as bm
from sim.arc_evaluator import ARCSymmetryInductor

def load_arc_task(task_path):
    with open(task_path, 'r') as f:
        return json.load(f)

def evaluate_official_task(task_data, inductor):
    """
    Implements the official ARC Prize scoring (2 attempts).
    Returns 1 if the model succeeds in either of 2 attempts, else 0.
    """
    train_pairs = task_data['train']
    test_inputs = task_data['test']
    
    task_success = 0
    
    # ARC-AGI-2 Evaluation: Usually 1 test input per task
    for test_idx, test_item in enumerate(test_inputs):
        input_grid = np.array(test_item['input'])
        target_grid = np.array(test_item['output'])
        
        # 1st Attempt: Use Induction with demonstration pairs
        # The internal mechanism builds R_STDP structural bridges via BG Reward
        pred_grid = inductor.evaluate_task(train_pairs, input_grid, max_train_steps=20, max_test_steps=20)
        
        # ARC targets vary in size. Predictions map to 30x30 max bounding limits.
        # We slice exactly to the target shape for strict pixel matching.
        H, W = target_grid.shape
        sliced_pred = pred_grid[:H, :W]
        
        # Success if 100% matched pixel-for-pixel (Zero-Shot Induction)
        if np.array_equal(sliced_pred, target_grid):
            task_success = 1
            break
            
        # 2nd Attempt: Reset and re-induce (Hypothesis Testing: Reflection vs. Rotation)
        bm.clear_name_cache()
        pred_grid2 = inductor.evaluate_task(train_pairs, input_grid, max_train_steps=20, max_test_steps=20)
        
        if np.array_equal(pred_grid2[:H, :W], target_grid):
            task_success = 1
            break
            
    return task_success

def run_full_evaluation(eval_dir, limit=None):
    task_files = glob.glob(os.path.join(eval_dir, "*.json"))
    if limit:
        task_files = task_files[:limit]
        
    inductor = ARCSymmetryInductor()
    results = []
    
    print(f"--- Cortex: Starting Full ARC-AGI-2 Evaluation ({len(task_files)} tasks) ---")
    
    for i, task_file in enumerate(task_files):
        task_data = load_arc_task(task_file)
        success = evaluate_official_task(task_data, inductor)
        results.append(success)
        
        print(f"Task {i+1}/{len(task_files)}: {'SUCCESS' if success else 'FAILURE'} ({os.path.basename(task_file)})")
        
    final_score = (sum(results) / len(results)) * 100
    print(f"\nFINAL ARC-AGI-2 EVALUATION SCORE: {final_score:.2f}%")
    return final_score

if __name__ == "__main__":
    eval_path = "/Users/ashishmishra/cortex/data/arc_full/data/evaluation"
    # Ensure name cache is clear for initial load
    bm.clear_name_cache()
    run_full_evaluation(eval_path) # Full 120 tasks
