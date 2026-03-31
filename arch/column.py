import brainpy as bp
import brainpy.math as bm
from core.neurons import LIF, MultiCompartmentNeuron, Izhikevich
from core.synapses import STDP, R_STDP, HomeostaticScaling

class CorticalColumn(bp.dyn.DynamicalSystem):
    """
    A 6-layer laminated cortical column.
    
    L1: Apical dendrites of deep layers.
    L2/3: Local feature integration.
    L4: Primary input layer from thalamus.
    L5: Output layer (RS/IB neurons).
    L6: Feedback layer to thalamus.
    """
    def __init__(self, size=100, **kwargs):
        super(CorticalColumn, self).__init__(**kwargs)
        
        # Layer Sizes (Approximate ratios)
        N_l23 = int(size * 0.4)
        N_l4  = int(size * 0.2)
        N_l5  = int(size * 0.25)
        N_l6  = int(size * 0.15)
        
        # Layer Populations (Names are now automatically managed by BrainPy for uniqueness)
        self.L23 = MultiCompartmentNeuron(N_l23)
        self.L4 = LIF(N_l4)
        self.L5 = MultiCompartmentNeuron(N_l5)
        self.L6 = LIF(N_l6)
        
        # Connectivity (Simplified canonical microcircuit)
        # L4 -> L2/3 (Feedforward)
        self.conn_l4_l23 = R_STDP(pre=self.L4, post=self.L23, 
                                conn=bp.conn.FixedProb(0.1))
        
        # L2/3 -> L5 (Deep integration)
        self.conn_l23_l5 = R_STDP(pre=self.L23, post=self.L5,
                                conn=bp.conn.FixedProb(0.15))
        
        # L5 -> L6 (Feedback relay)
        self.conn_l5_l6 = R_STDP(pre=self.L5, post=self.L6,
                                conn=bp.conn.FixedProb(0.1))
        
        self.conn_l6_l4 = HomeostaticScaling(pre=self.L6, post=self.L4,
                                        target_rate=0.01)

    def update(self, ThalamicInput=None, Reward=None):
        # Input to L4
        ThalamicInput = 0. if ThalamicInput is None else ThalamicInput
        self.L4.update(x=ThalamicInput)
        
        # Sequential updates for layers (simplified)
        self.conn_l4_l23.update(reward=Reward)
        self.L23.update()
        
        self.conn_l23_l5.update(reward=Reward)
        self.L5.update()
        
        self.conn_l5_l6.update(reward=Reward)
        self.L6.update()
