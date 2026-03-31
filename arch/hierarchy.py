import brainpy as bp
import brainpy.math as bm
from arch.column import CorticalColumn

class HierarchicalCortex(bp.dyn.DynamicalSystem):
    """
    Assembles multiple CorticalColumns into a hierarchical processing stream.
    Mimics the Feedforward (L4) and Feedback (L6/L1) connectivity of the brain.
    """
    def __init__(self, num_levels=3, size_per_column=200, name=None):
        super(HierarchicalCortex, self).__init__(name=name)
        self.num_levels = num_levels
        
        # Instantiate columns for each hierarchy level
        # Level 0: Primary (Sensory/Token input)
        # Level 1: Associative (Word/Phrase abstraction)
        # Level 2: Higher-Order (Sentence/Context)
        self.levels = [CorticalColumn(size=size_per_column, name=f'Level_{i}') for i in range(num_levels)]
        # Register nodes for BrainPy state tracking
        for level in self.levels:
            self.register_implicit_nodes(level)
        
    def update(self, PrimaryInput=None, Reward=None):
        # 1. Level 0: Primary input processing
        self.levels[0].update(ThalamicInput=PrimaryInput, Reward=Reward)
        
        # 2. Sequential Hierarchy Updates
        for i in range(1, self.num_levels):
            # Feedforward: Lower level L2/3 projects to higher level L4
            # Keep spatial vectors via pooling rather than flat mean
            lower_v = self.levels[i-1].L23.V_soma
            l4_size = self.levels[i].L4.size[0] if isinstance(self.levels[i].L4.size, (tuple, list)) else self.levels[i].L4.size
            pool_factor = max(1, lower_v.size // l4_size)
            p_size = pool_factor * l4_size
            spatial_in = bm.mean(bm.reshape(lower_v[:p_size], (-1, l4_size, pool_factor)), axis=2)
            self.levels[i].update(ThalamicInput=spatial_in, Reward=Reward)
            
            # Feedback: Higher level L6 projects back to lower level L1 (Apical)
            higher_feedback = self.levels[i].L6.V
            l5_size = self.levels[i-1].L5.size[0] if isinstance(self.levels[i-1].L5.size, (tuple, list)) else self.levels[i-1].L5.size
            expand_factor = (l5_size // higher_feedback.size) + 1
            feedback_in = bm.repeat(higher_feedback, expand_factor, axis=-1)[..., :l5_size]
            self.levels[i-1].L5.input_apical.value += feedback_in
            
    def get_output(self):
        """Returns the activity of the highest level in the hierarchy."""
        return self.levels[-1].L6.V
