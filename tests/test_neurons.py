import pytest
import brainpy as bp
import brainpy.math as bm
from core.neurons import LIF, Izhikevich, MultiCompartmentNeuron

def test_lif_spiking():
    # Test if LIF spikes when input is above threshold
    lif = LIF(1, V_th=10., tau=10.)
    
    # Run for several steps with high input
    for _ in range(200):
        lif.update(x=20.)
        if lif.spike[0]:
            break
    
    assert lif.spike[0] == True
    assert lif.V[0] <= lif.V_reset + 1e-5

def test_izhikevich_patterns():
    # Test RS (Regular Spiking) pattern
    iz = Izhikevich(1, a=0.02, b=0.2, c=-65., d=8.)
    
    spikes = 0
    for _ in range(1000):
        iz.update(x=10.)
        if iz.spike[0]:
            spikes += 1
            
    assert spikes > 0

def test_mcn_gating():
    # Test if Apical input gates Basal input in MCN
    mcn = MultiCompartmentNeuron(1, V_th=10.)
    
    # Case 1: Basal input only
    mcn.V_soma.value = bm.zeros(1)
    for _ in range(100):
        mcn.input_basal.value = bm.full(1, 5.)
        mcn.input_apical.value = bm.zeros(1)
        mcn.update()
    v_no_apical = mcn.V_soma[0]
    
    # Reset
    mcn.V_soma.value = bm.zeros(1)
    mcn.V_basal.value = bm.zeros(1)
    mcn.V_apical.value = bm.zeros(1)
    
    # Case 2: Basal + Apical input
    for _ in range(100):
        mcn.input_basal.value = bm.full(1, 5.)
        mcn.input_apical.value = bm.full(1, 10.)
        mcn.update()
    v_with_apical = mcn.V_soma[0]
    
    # Coincidence gating should result in higher somatic potential
    assert v_with_apical > v_no_apical
