import brainpy as bp
import brainpy.math as bm
from core.neurons import Izhikevich

class MediodorsalThalamus(bp.dyn.DynamicalSystem):
    """
    Subcortical Associative Router (v2.0).
    Replaces Self-Attention with a spiking KV-separation logic.
    Complexity: O(T) due to rhythmic disinhibition.
    """
    def __init__(self, size=100, name=None):
        super(MediodorsalThalamus, self).__init__(name=name)
        self.size = size
        
        # 1. Key Population: Represents the current Input (from L6)
        self.keys = Izhikevich(size, a=0.02, b=0.2, c=-65.0, d=8.0, name='Thalamus_Keys')
        
        # 2. Value Population: Represents the Associative context (Working Memory)
        self.values = Izhikevich(size, a=0.03, b=0.25, c=-60.0, d=4.0, name='Thalamus_Values')
        
        # Latent representation state (Gated output)
        self.latent_state = bm.Variable(bm.zeros(size))
        
    def update(self, CorticalOutput=None, BG_Disinhibition=0.0):
        # 1. Update Keys from Cortical L6 Feedback
        self.keys.update(x=CorticalOutput if CorticalOutput is not None else 0.0)
        
        # 2. Update Values (Internal dynamics / associative traces)
        # Driven by Key spikes (Excitatory coupling)
        self.values.update(x=bm.sum(self.keys.spike) * 10.0) 
        
        # Action Selection (KV Gating): Disinhibition from Basal Ganglia
        # BG_Disinhibition is the mean spike rate (0 to 1).
        # gate = 1.0 (fully disinhibited) to 0.2 (low-level tonic)
        gate = bm.maximum(0.2, 1.0 - BG_Disinhibition) 
        
        # Output: Gated context representation
        # Complexity is O(1) per step, O(T) total vs O(T^2) for Transformers
        self.latent_state.value = self.values.spike.astype(float) * gate
        
    def get_modulatory_signal(self):
        return self.latent_state
