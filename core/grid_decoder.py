import brainpy.math as bm
import numpy as np

class SpikingGridDecoder:
    """
    Decodes the continuous L6 output potentials back into a discrete 30x30 ARC grid.
    Since the physical network output (135 neurons) must map to 900 pixels,
    we tile and quantize the voltages to integer colors (0-9).
    """
    def __init__(self, output_size=135, grid_size=(30, 30)):
        self.output_size = output_size
        self.grid_size = grid_size
        self.target_pixels = grid_size[0] * grid_size[1]
        self.repeats = (self.target_pixels // self.output_size) + 1

    def decode(self, l6_potentials):
        """
        Translates continuous voltage traces into a discrete color matrix.
        l6_potentials: (Batch, output_size)
        """
        B = l6_potentials.shape[0] if len(l6_potentials.shape) > 1 else 1
        if l6_potentials.ndim == 1:
            l6_potentials = bm.expand_dims(l6_potentials, axis=0)
            
        # Tile elements to fill target topological grid
        projected = bm.repeat(l6_potentials, self.repeats, axis=1)[:, :self.target_pixels]
        
        # Normalize continuous outputs to discrete ARC colors (0-9)
        # Using simple binning based on activity magnitude
        v_min = bm.min(projected, axis=1, keepdims=True)
        v_max = bm.max(projected, axis=1, keepdims=True)
        normalized = (projected - v_min) / (v_max - v_min + 1e-9)
        
        discrete_colors = bm.floor(normalized * 9.99).astype(bm.int32)
        discrete_colors = bm.reshape(discrete_colors, (B, *self.grid_size))
        
        return np.array(discrete_colors)
