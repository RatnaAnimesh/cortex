import brainpy as bp
import brainpy.math as bm
from arch.column import CorticalColumn
from arch.thalamus import MediodorsalThalamus
from arch.basal_ganglia import BasalGanglia
from arch.cerebellum import Cerebellum

class SIM_Engine(bp.dyn.DynamicalSystem):
    """
    Simulation Engine for the Cortex architecture (v2.0).
    Orchestrates the interplay between the 6-layer column and its Subcortical controllers.
    """
    def __init__(self, size=200, name=None):
        super(SIM_Engine, self).__init__(name=name)
        
        # 1. Cortical Column (Macro-Architecture)
        self.column = CorticalColumn(size=size, name='Cortex')
        
        # 2. Subcortical Routing & Action Selection
        # Thalamus: KV-Separation associative Memory
        self.md_thalamus = MediodorsalThalamus(size=int(size*0.15), name='Thalamus')
        # Basal Ganglia: Disinhibitory Gating
        self.basal_ganglia = BasalGanglia(size=50, name='BasalGanglia')
        # Cerebellum: Fast Error Correction
        self.cerebellum = Cerebellum(size=50, name='Cerebellum')
        
    def update(self, ExternalInput=None, Reward=None):
        # 1. Basal Ganglia Action Selection
        # Receives input from L5 (Soma) and Dopaminergic Reward
        self.basal_ganglia.update(CorticalInput=self.column.L5.V_soma, EnvironmentalReward=Reward)
        bg_disinhibition = self.basal_ganglia.get_disinhibition_signal()
        
        # 2. Thalamic Associative Gating
        # Receives L6 feedback and BG action selection signal
        self.md_thalamus.update(CorticalOutput=self.column.L6.V, BG_Disinhibition=bg_disinhibition)
        mod_signal = self.md_thalamus.get_modulatory_signal()
        
        # 3. Cerebellar Motor Prediction
        # Receives sensory copy and motor copy from L5
        self.cerebellum.update(SensoryInput=ExternalInput, CorticalCopy=self.column.L5.V_soma, Error=-Reward if Reward is not None else 0.)
        motor_prediction = self.cerebellum.get_motor_prediction()
        
        # 4. Cortical Column Processing
        # L5 integrates Thalamic context (apical) and Cerebellar prediction (sync)
        self.column.L5.V_apical += bm.mean(mod_signal)
        # Synchronization principle: Phase-locking L5 with subcortical motor prediction
        self.column.L5.V_soma += motor_prediction * 0.1 
        
        # Main Column update
        self.column.update(ThalamicInput=ExternalInput, Reward=Reward)

def run_cortex_trial(duration=300, input_val=15.0):
    """
    Experimental environment for the Spiking World Model.
    """
    engine = SIM_Engine()
    runner = bp.DSRunner(engine, 
                         monitors={'L5_soma': engine.column.L5.V_soma,
                                   'L5_spikes': engine.column.L5.spike,
                                   'Thalamus_latent': engine.md_thalamus.latent_state,
                                   'BG_signal': engine.basal_ganglia.GPi_SNr.V},
                         dt=0.1)
    
    # Run with continuous inputs
    num_steps = int(duration / 0.1)
    # Simple sine wave input to test temporal dynamics
    inputs = bm.sin(bm.linspace(0, 10, num_steps)) * 5.0 + input_val
    inputs = inputs.reshape((num_steps, 1))
    
    runner.run(inputs=inputs)
    return runner

if __name__ == "__main__":
    run_cortex_trial()
