import brainpy as bp
import brainpy.math as bm
from core.neurons import LIF, Izhikevich

class Cerebellum(bp.dyn.DynamicalSystem):
    """
    Subcortical Error-Correction system.
    Implements the granular and Purkinje cell loop for motor prediction and jitter reduction.
    """
    def __init__(self, size=100, name=None):
        super(Cerebellum, self).__init__(name=name)
        self.size = size

        # 1. Granule Cells (GR): High-dimensional sensory expansion
        # These are small, fast, and very numerous in biological brains
        self.GR = LIF(size * 5, V_th=-50.0, tau=10.0, name='Granule_Cells')
        
        # 2. Purkinje Cells (PC): Inhibitory integrators
        # Large dendritic trees, receive input from thousands of Granule Cells
        self.PC = Izhikevich(int(size / 10), a=0.03, b=0.25, c=-60.0, d=4.0, name='Purkinje_Cells')

        # 3. Deep Cerebellar Nuclei (DCN): Output stage
        # Fast excitatory response that synchronizes with movement
        self.DCN = Izhikevich(int(size / 10), a=0.02, b=0.2, c=-65.0, d=8.0, name='DCN')

    def update(self, SensoryInput=None, CorticalCopy=None, Error=None):
        """
        SensoryInput: Mossy Fiber input (external sensation)
        CorticalCopy: Efference Copy from Motor Cortex
        Error: Climbing Fiber input (inferior olive reward/punishment)
        """
        # Expansion of Sensory and Cortical data in Granule Cells
        # Repeat the vector drive across the expanded population
        gr_drive = (SensoryInput + CorticalCopy) / 2. if SensoryInput is not None else 0.
        gr_sz = self.GR.size[0] if isinstance(self.GR.size, (tuple, list)) else self.GR.size
        # Correctly broadcast scaler or vector to the 250 Granule nodes
        if isinstance(gr_drive, (int, float)) or (hasattr(gr_drive, 'ndim') and gr_drive.ndim == 0):
            expanded_drive = bm.full((gr_sz,), float(gr_drive))
        else:
            expanded_drive = bm.repeat(gr_drive, 5, axis=-1)
        
        self.GR.update(x=expanded_drive)
        
        # Integration of Purkinje and DCN...

        # Purkinje Cells integrate GR spikes and Error signals
        # Error signal (Climbing Fiber) induces massive complex spikes
        pc_drive = bm.sum(self.GR.spike) * 0.1 
        error_val = 0. if Error is None else Error
        pc_drive += error_val * 20.0 # Error-driven plasticity/spiking
        
        self.PC.update(x=pc_drive)

        # Output: DCN is inhibited by Purkinje Cells
        # Disinhibition results in fast motor output sync
        # Logic: DCN receives tonic excitation, inhibited by PC
        dcn_drive = 15.0 - bm.sum(self.PC.spike) * 2.0
        self.DCN.update(x=dcn_drive)

    def get_motor_prediction(self):
        """
        Fast excitatory signal to synchronize with cortical L5 outputs.
        """
        return bm.mean(self.DCN.V) # Simplified scalar prediction for motor alignment
