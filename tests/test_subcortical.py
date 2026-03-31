import pytest
import brainpy as bp
import brainpy.math as bm
from arch.basal_ganglia import BasalGanglia
from arch.thalamus import MediodorsalThalamus

def test_bg_thalamus_disinhibition():
    """
    Test Case: Verify that Basal Ganglia reward input disinhibits the Thalamus.
    """
    bp.share.dt = 0.1
    bg = BasalGanglia(size=50)
    thalamus = MediodorsalThalamus(size=50)
    
    # 1. State: Baseline (No Reward)
    # GPi should be tonically active, inhibiting the Thalamus
    spikes_no_reward = 0.0
    for _ in range(1000):
        bg.update(CorticalInput=10.0, EnvironmentalReward=0.0)
        bg_signal = bg.get_disinhibition_signal()
        thalamus.update(CorticalOutput=30.0, BG_Disinhibition=bg_signal)
        # thalamus.latent_state is gated by BG disinhibition
        spikes_no_reward += bm.sum(thalamus.latent_state > 0.5) 
    
    # 2. State: Reward (Dopamine Pulse)
    # Clear name cache to avoid UniqueNameError and ensure clear state transition
    bm.clear_name_cache()
    bg_r = BasalGanglia(size=50)
    thalamus_r = MediodorsalThalamus(size=50)
    spikes_with_reward = 0.0
    for _ in range(1000):
        bg_r.update(CorticalInput=10.0, EnvironmentalReward=100.0) # Saturated reward
        bg_signal = bg_r.get_disinhibition_signal()
        thalamus_r.update(CorticalOutput=30.0, BG_Disinhibition=bg_signal)
        spikes_with_reward += bm.sum(thalamus_r.latent_state > 0.5)
    
    # Disinhibition Principle: Thalamic output must increase when rewarded
    assert spikes_with_reward > spikes_no_reward

def test_cerebellum_sync():
    """
    Test Case: Verify that Cerebellum motor prediction responds to Error.
    """
    from arch.cerebellum import Cerebellum
    cerebellum = Cerebellum(size=50)
    
    # No Error
    cerebellum.update(SensoryInput=5.0, CorticalCopy=5.0, Error=0.0)
    pred_low_error = cerebellum.get_motor_prediction()
    
    # High Error (Climbing Fiber activation)
    # This induces massive PC activity which inhibits DCN output
    for _ in range(100):
        cerebellum.update(SensoryInput=5.0, CorticalCopy=5.0, Error=10.0)
    
    pred_high_error = cerebellum.get_motor_prediction()
    
    # Cerebellar Principle: Prediction changes in response to error feedback
    assert pred_high_error != pred_low_error
