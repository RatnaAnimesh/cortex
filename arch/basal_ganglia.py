import brainpy as bp
import brainpy.math as bm
from core.neurons import LIF, Izhikevich

class BasalGanglia(bp.dyn.DynamicalSystem):
    """
    Subcortical Action Selection system.
    Implements the Direct and Indirect pathways for disinhibitory control of the Thalamus.
    """
    def __init__(self, size=100, name=None):
        super(BasalGanglia, self).__init__(name=name)
        self.size = size

        # 1. Striatum (D1 and D2 MSNs)
        self.D1_MSN = Izhikevich(size, a=0.02, b=0.2, c=-65.0, d=8.0)
        self.D2_MSN = Izhikevich(size, a=0.02, b=0.2, c=-65.0, d=8.0)

        # 2. Output Nuclei: GPi/SNr
        self.GPi_SNr = Izhikevich(size, a=0.02, b=0.2, c=-65.0, d=8.0)
        # Initialize with bias current to simulate tonic activity
        self.GPi_SNr.V.value = bm.full(size, -50.0) 

        # 3. Neuromodulation: SNc
        self.SNc = LIF(1)
        
        # 4. Moving average trace for smooth gating
        self.gpi_rate = bm.Variable(bm.zeros(1))

    def update(self, CorticalInput=None, EnvironmentalReward=None):
        dt = getattr(bp.share, 'dt', 0.1)
        
        # Dopamine Signal (Reward)
        Reward = 0. if EnvironmentalReward is None else EnvironmentalReward
        self.SNc.update(x=Reward)
        Dopamine = self.SNc.spike[0] # Simplification: Binary dopamine pulse

        # Cortical Input to Striatum
        # Dopamine potentiates D1 and depresses D2
        # Use bm.full for broadcasting scalar/vector inputs
        d_size = self.D1_MSN.size[0] if isinstance(self.D1_MSN.size, (tuple, list)) else self.D1_MSN.size
        d1_drive = bm.full((d_size,), CorticalInput * (1 + Dopamine) if CorticalInput is not None else 0.0)
        d2_drive = bm.full((d_size,), CorticalInput * (1 - Dopamine) if CorticalInput is not None else 0.0)
        
        self.D1_MSN.update(x=d1_drive)
        self.D2_MSN.update(x=d2_drive)

        # Output stage: GPi/SNr receives inhibition from D1
        # Disinhibition logic: more D1 activity -> less GPi activity -> more Thalamic activity
        # Scale to ensure stable physical integration
        gpi_inhibition = bm.sum(self.D1_MSN.spike.astype(float)) * -5.0
        # Gentle tonic baseline current for GPi (dominates by default)
        g_size = self.GPi_SNr.size[0] if isinstance(self.GPi_SNr.size, (tuple, list)) else self.GPi_SNr.size
        gpi_drive = bm.full((g_size,), 30.0 + gpi_inhibition)
        
        self.GPi_SNr.update(x=gpi_drive)
        
        # Smooth the firing rate to create a continuous inhibitory gate
        self.gpi_rate.value += (-self.gpi_rate + bm.mean(self.GPi_SNr.spike.astype(float))) / 5.0 * dt

    def get_disinhibition_signal(self):
        """
        Returns the normalized spiking activity of the GPi/SNr (0.0 to 1.0).
        High value (~1.0) = Thalamus is inhibited.
        Low value (~0.0) = Thalamus is disinhibited.
        """
        # Multiply by scaling factor because average biological firing fraction per dt is << 1.0
        return bm.clip(self.gpi_rate[0] * 50.0, 0.0, 1.0)

# In arch/thalamus.py (Logic only, I will use replace_file_content for the file next)
# gate = 1.0 - BG_Disinhibition
