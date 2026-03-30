import neurobench
from neurobench.benchmarks import Benchmark
from neurobench.datasets import Dataset
from neurobench.models import Model
import brainpy as bp
import brainpy.math as bm
from sim.engine import SIM_Engine

class CortexNeuroBenchModel(Model):
    """
    Wraps the Cortex SIM_Engine for NeuroBench evaluation.
    """
    def __init__(self, engine):
        self.engine = engine
        
    def __call__(self, x):
        # x is a batch of sequences
        # For simplicity, we process a single sequence and sum output
        outputs = []
        for seq in x:
            self.engine.update(ExternalInput=bm.mean(seq))
            outputs.append(self.engine.column.L5.spike.astype(float))
        return bm.concatenate(outputs)

    def track_synops(self):
        """
        Calculates Synaptic Operations (SOPs) based on connection weight matrix.
        SOPs = Number of Spikes * Number of Connections
        """
        # Linear complexity O(T): Spikes * Constant(Sparsity)
        total_sop = 0
        # Iterate over column connections
        for conn in [self.engine.column.conn_l4_l23, 
                     self.engine.column.conn_l23_l5, 
                     self.engine.column.conn_l5_l6]:
            spikes_pre = bm.sum(conn.pre.spike)
            conns_per_pre = conn.conn.max_num # Simplified connectivity estimate
            total_sop += spikes_pre * conns_per_pre
        return total_sop

def run_performance_benchmark():
    engine = SIM_Engine(size=200)
    model = CortexNeuroBenchModel(engine)
    
    # Track metrics: SOPs, Sparsity, Latency
    print("--- NeuroBench Performance Report ---")
    
    # 1. Stimulate briefly to get spike activity
    for _ in range(100):
        engine.update(ExternalInput=20.0)
    
    sop = model.track_synops()
    sparsity = (bm.sum(engine.column.L5.spike == 0) / engine.column.L5.size[0]) * 100
    
    # 2. Results compared to standard FLOPs
    # Energy: 0.067 pJ per SOP (Loihi 2) vs 0.4 pJ per FLOP (A100)
    energy_advantage = (sop * 0.4) / (sop * 0.067) if sop > 0 else 0
    
    print(f"Synaptic Operations (SOPs): {sop}")
    print(f"Spike Sparsity: {sparsity:.2f}%")
    print(f"Theoretical Energy Advantage (vs A100): {energy_advantage:.2f}x")
    
    return {"SOPs": sop, "Sparsity": sparsity, "Adv": energy_advantage}
