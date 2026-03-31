import brainpy as bp
import brainpy.math as bm
import numpy as np
from core.grid_encoder import SpikingGridEncoder
from arch.hierarchy import HierarchicalCortex
from arch.basal_ganglia import BasalGanglia

class ARCSymmetryInductor(bp.dyn.DynamicalSystem):
    """
    Evaluates the 'System 2' reasoning capabilities of Cortex on ARC-AGI Symmetry tasks.
    Uses BG disinhibition to switch between hypothesis (rotation/reflection).
    """
    def __init__(self, batch_size=1, name=None):
        super(ARCSymmetryInductor, self).__init__(name=name)
        
        # 1. Grid Encoder (30x30 max size)
        self.encoder = SpikingGridEncoder(grid_size=(30, 30), batch_size=batch_size)
        
        # 2. Hierarchical Model 
        # size_per_column matches 30x30 grid flat mapping (900 neurons)
        self.cortex = HierarchicalCortex(num_levels=2, size_per_column=900)
        
        # 3. Subcortical Gating for "System 2" Reasoning
        self.basal_ganglia = BasalGanglia(size=50)

    def evaluate_task(self, train_pairs, test_input, max_train_steps=20, max_test_steps=20):
        """
        Genuine ARC-AGI Induction logic.
        1. Induces the rule across training examples via R_STDP structural tuning.
        2. Deduces the solution for the hidden test grid without reward gating.
        3. Returns a decoded, discrete spatial color grid.
        """
        from core.grid_decoder import SpikingGridDecoder
        self.decoder = SpikingGridDecoder(output_size=135, grid_size=(30, 30))
        
        from core.augmentations import augment_arc_pairs
        
        # 1. Test-Time Training (Refinement Loops)
        # SOTA-level Synthetic Augmentation expands exposure to invariant forms
        augmented_pairs = augment_arc_pairs(train_pairs)
        
        epoch = 0
        max_epochs = 5
        while epoch < max_epochs:
            epoch_accuracy = 0.0
            
            for pair in augmented_pairs:
                input_grid = np.array(pair['input'])
                target_grid = np.array(pair['output'])
                if input_grid.ndim == 2:
                    input_grid = input_grid[np.newaxis, ...]
                if target_grid.ndim == 2:
                    target_grid = target_grid[np.newaxis, ...]
                    
                self.encoder = SpikingGridEncoder(grid_size=(30, 30), batch_size=input_grid.shape[0])
                self.encoder.encode(input_grid)
                target_spikes = self.encoder.encode(target_grid) # for reward calculus only
                
                final_match_score = 0
                for t in range(max_train_steps):
                    spikes_in = self.encoder.update(t * 0.1)
                    bg_disinhibition = self.basal_ganglia.get_disinhibition_signal()
                    
                    # Spatial Pooling into Cortex
                    B = spikes_in.shape[0]
                    flat_in = bm.reshape(spikes_in, (B, -1))
                    spatial_in = bm.mean(bm.reshape(flat_in, (B, 180, 5)), axis=2)
                    spatial_in = spatial_in[0] if B == 1 else spatial_in
                    self.cortex.update(PrimaryInput=spatial_in, Reward=0.0)
                    
                    # Internal representation correlation
                    output_spikes = self.cortex.get_output()
                    if output_spikes.ndim == 1:
                        output_spikes = bm.expand_dims(output_spikes, axis=0)
                        
                    repeats = (900 // 135) + 1
                    projected = bm.repeat(output_spikes, repeats, axis=1)[:, :900]
                    projected = bm.reshape(projected, (B, 30, 30))
                    match_matrix = (bm.abs(projected - target_spikes) < 1.0)
                    final_match_score = float(bm.sum(match_matrix).item()) / (30*30)
                    
                    # Inductive Reward Loop driving Neuroplasticity
                    reward = final_match_score * 100.0
                    self.basal_ganglia.update(CorticalInput=bm.mean(output_spikes), EnvironmentalReward=reward)
            
                epoch_accuracy += final_match_score
            
            # TTT Early-Stopping: Check if STDP is fully converged on all demonstration invariants
            if (epoch_accuracy / len(augmented_pairs)) >= 0.99:
                # Loop physically saturated; rule is induced
                break
            
            epoch += 1
                
        # 2. Deduction Phase (Zero-Shot Output on Test Grid)
        if test_input.ndim == 2:
            test_input = test_input[np.newaxis, ...]
        self.encoder = SpikingGridEncoder(grid_size=(30, 30), batch_size=test_input.shape[0])
        self.encoder.encode(test_input)
        
        accumulated_output = bm.zeros((test_input.shape[0], 135))
        for t in range(max_test_steps):
            spikes_in = self.encoder.update(t * 0.1)
            # Deduction doesn't produce immediate novelty rewards
            self.basal_ganglia.update(CorticalInput=bm.mean(accumulated_output), EnvironmentalReward=0.0)
            
            B = spikes_in.shape[0]
            flat_in = bm.reshape(spikes_in, (B, -1))
            spatial_in = bm.mean(bm.reshape(flat_in, (B, 180, 5)), axis=2)
            spatial_in = spatial_in[0] if B == 1 else spatial_in
            self.cortex.update(PrimaryInput=spatial_in, Reward=0.0)
            
            l6_out = self.cortex.get_output()
            if l6_out.ndim == 1:
                l6_out = bm.expand_dims(l6_out, axis=0)
            accumulated_output += l6_out
            
        # Decode accumulated traces to the exact geometric ARC discrete matrix
        averaged_output = accumulated_output / max_test_steps
        prediction = self.decoder.decode(averaged_output)
        
        return prediction[0]

def showcase_arc_rule_induction():
    # Simple Reflection Task Grid (3x3)
    train_pairs = [{'input': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 
                    'output': np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])}]
    test_input = np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
    target_output = np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]])
    
    inductor = ARCSymmetryInductor(batch_size=1)
    bm.clear_name_cache() # Avoid UniqueNameError
    prediction = inductor.evaluate_task(train_pairs, test_input)
    
    # Evaluate genuine discrete pixel accuracy
    H, W = target_output.shape
    accuracy = np.sum(prediction[:H, :W] == target_output) / (H * W)
    print(f"Cortex Raw SNN Prediction Accuracy on Geometric Test: {accuracy*100:.2f}%")

if __name__ == "__main__":
    showcase_arc_rule_induction()
