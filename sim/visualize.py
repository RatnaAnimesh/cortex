import matplotlib.pyplot as plt
import numpy as np
import brainpy as bp
import brainpy.math as bm
from sim.engine import SIM_Engine

def visualize_activity(runner):
    """
    Plots the activity of the simulated Cortex.
    """
    # 2. Extract Data
    times = runner.mon.ts
    v_l5 = runner.mon['L5_soma'][:, 0]
    spikes_l5 = runner.mon['L5_spikes'][:, 0]
    thalamus_state = bm.mean(runner.mon['Thalamus_latent'], axis=1)
    bg_signal = bm.mean(runner.mon['BG_signal'], axis=1)
    
    # 3. Plotting
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Subplot 1: L5 Somatic Potential
    axes[0].plot(times, v_l5, label='L5 Soma (mV)', color='blue')
    axes[0].set_title('Layer 5 Somatic Potential')
    axes[0].legend()
    
    # Subplot 2: L5 Spikes
    axes[1].eventplot(times[spikes_l5 > 0], color='black')
    axes[1].set_title('L5 Spike Train')
    
    # Subplot 3: Thalamic Latent State (Associative Gating)
    axes[2].plot(times, thalamus_state, label='Thalamic Latent', color='green')
    axes[2].set_title('Thalamocortical Associative Gating')
    axes[2].legend()
    
    # Subplot 4: Basal Ganglia Disinhibition Signal (GPi Output)
    axes[3].plot(times, bg_signal, label='GPi Potential', color='red')
    axes[3].set_title('Basal Ganglia Output (Action Gating)')
    axes[3].set_xlabel('Time (ms)')
    axes[3].legend()
    
    plt.tight_layout()
    plt.savefig('subcortical_activity_v2.png')
    print("Activity plot saved to subcortical_activity_v2.png")

if __name__ == "__main__":
    from sim.engine import run_cortex_trial
    # Run a short trial
    runner = run_cortex_trial(duration=300, input_val=15.0)
    # Visualize
    visualize_activity(runner)
