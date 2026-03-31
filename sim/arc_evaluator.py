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

    def evaluate_task(self, input_grid, target_grid, max_steps=1000):
        """
        Induces the rule by comparing encoded input to target spikes.
        Handles both single grid (H, W) and batched grid (B, H, W) inputs.
        """
        # Ensure input is batched (B, H, W)
        if input_grid.ndim == 2:
            input_grid = input_grid[np.newaxis, ...]
        if target_grid.ndim == 2:
            target_grid = target_grid[np.newaxis, ...]
            
        # Encode Target Grid for comparison
        target_spikes = self.encoder.encode(target_grid)
        
        # Batch simulation loop
        # Accuracy accumulated across time and batch
        correct_pixels = 0.0
        
        for t in range(max_steps):
            # Encode Input Grid
            spikes_in = self.encoder.update(t * 0.1)
            
            # Subcortical Gating (Hypothesis Testing)
            bg_disinhibition = self.basal_ganglia.get_disinhibition_signal()
            
            # Cortex update (Gated hierarchy)
            # Pool spatial grid from 900 -> 180 to maintain topology into L4
            B = spikes_in.shape[0]
            flat_in = bm.reshape(spikes_in, (B, -1))
            spatial_in = bm.mean(bm.reshape(flat_in, (B, 180, 5)), axis=2)
            self.cortex.update(PrimaryInput=spatial_in, Reward=0.0)
            
            # Extract batched L6 values (B, 135)
            # Output check
            output_spikes = self.cortex.get_output()
            if output_spikes.ndim == 1:
                output_spikes = bm.expand_dims(output_spikes, axis=0) # (1, 135)
                
            # Basic Latent-to-Grid Projection for benchmark scoring (135 -> 900)
            B = target_spikes.shape[0]
            # Tile elements to fill 900 mapping
            repeats = (900 // 135) + 1
            projected = bm.repeat(output_spikes, repeats, axis=1)[:, :900]
            projected = bm.reshape(projected, (B, 30, 30))
            
            # Since output is continuous V and target is latency, we compare topology correlation
            # For exact match required by ARC:
            match_matrix = (bm.abs(projected - target_spikes) < 1.0)
            correct_pixels += bm.sum(match_matrix)
            
            # Reward Signaling (Induction)
            reward = (bm.sum(match_matrix) / (30*30)) * 10.0
            self.basal_ganglia.update(CorticalInput=bm.mean(output_spikes), EnvironmentalReward=reward)
            
        final_acc = correct_pixels.item() / (max_steps * 30 * 30 * input_grid.shape[0])
        return final_acc

def showcase_arc_rule_induction():
    # Simple Reflection Task Grid (3x3)
    input_g = np.array([[1, 0, 0], 
                        [0, 1, 0], 
                        [0, 0, 1]])
    target_g = np.array([[0, 0, 1], 
                         [0, 1, 0], 
                         [1, 0, 0]])
    
    inductor = ARCSymmetryInductor(batch_size=1)
    bm.clear_name_cache() # Avoid UniqueNameError
    accuracy = inductor.evaluate_task(input_g, target_g)
    
    print(f"Cortex Performance on ARC-AGI-2 Reflection: {accuracy*100:.2f}%")

if __name__ == "__main__":
    showcase_arc_rule_induction()
