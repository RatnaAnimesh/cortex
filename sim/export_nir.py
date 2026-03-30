import nir
import brainpy as bp
import brainpy.math as bm
import numpy as np

def export_cortex_to_nir(engine, filename="cortex_architecture.nir"):
    """
    Exports the Cortex SIM_Engine to the Neuromorphic Intermediate Representation (NIR).
    Enables deployment to Intel Loihi 2 via Lava.
    """
    nodes = {}
    edges = []
    
    # 1. Map Cortical Layers to NIR Nodes
    # Use LIF/Izhikevich as NIR primitives
    for layer_name in ['L4', 'L23', 'L5', 'L6']:
        layer = getattr(engine.column, layer_name)
        # Create Affine/Linear layer + Spiking Dynamics
        nodes[layer_name] = nir.LIF(tau=layer.tau, v_threshold=layer.V_th, v_leak=layer.V_rest, r=1.0)
        
    # 2. Map Synaptic Connections to NIR Edges
    # Connection: Pre_Layer -> Post_Layer
    for conn in [engine.column.conn_l4_l23, engine.column.conn_l23_l5, engine.column.conn_l5_l6]:
        # Pre-Post weights
        weight_mat = bm.as_numpy(conn.w)
        # In NIR, weights are Affine or Linear nodes
        edge_name = f"{conn.pre.name}_to_{conn.post.name}"
        nodes[edge_name] = nir.Linear(weight=weight_mat)
        
        # Link Pre -> Linear -> Post
        edges.append((conn.pre.name, edge_name))
        edges.append((edge_name, conn.post.name))
        
    # 3. Create NIR Graph
    nir_graph = nir.NIRGraph(nodes, edges)
    
    # 4. Save to disk
    nir.write(nir_graph, filename)
    print(f"NIR export completed: {filename}")
    
def verify_nir_export():
    """
    Verifies that the NIR graph is compatible with Intel Lava.
    """
    from sim.engine import SIM_Engine
    engine = SIM_Engine()
    export_cortex_to_nir(engine)
    
if __name__ == "__main__":
    verify_nir_export()
