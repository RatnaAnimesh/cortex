import brainpy as bp
import brainpy.math as bm
import numpy as np

class SpikingGridEncoder(bp.dyn.DynamicalSystem):
    """
    Translates 2D ARC color grids into spatiotemporal spike patterns.
    Uses Latency Coding: Color ID (0-9) -> Spike Offset (ms).
    """
    def __init__(self, grid_size=(30, 30), time_window=10.0, batch_size=1, name=None):
        super(SpikingGridEncoder, self).__init__(name=name)
        self.grid_size = grid_size
        self.time_window = time_window # ms
        self.batch_size = batch_size

        # Latent state of the grid (Batch, Height, Width)
        self.grid = bm.Variable(bm.zeros((batch_size,) + grid_size))
        self.spike_times = bm.Variable(bm.zeros((batch_size,) + grid_size))
        self.spikes = bm.Variable(bm.zeros((batch_size,) + grid_size, dtype=bool))

    def encode(self, grid_data):
        """
        Maps grid color IDs (0-9) to spike latencies.
        Handles batched inputs (B, H, W).
        """
        # Ensure grid_data is batched (B, H, W)
        if grid_data.ndim == 2:
            grid_data = bm.expand_dims(grid_data, axis=0)
            
        B, H, W = grid_data.shape
        padded = bm.zeros((self.batch_size, self.grid_size[0], self.grid_size[1]))
        padded = bm.where(bm.arange(self.batch_size)[:, None, None] < B, 
                         padded.at[:, :H, :W].set(bm.as_ndarray(grid_data)), 
                         padded)
        
        self.grid.value = padded
        self.spike_times.value = (padded / 10.0) * self.time_window
        return self.spike_times

    def update(self, current_time):
        dt = getattr(bp.share, 'dt', 0.1)
        local_time = current_time % self.time_window
        
        # A spike occurs if local_time matches the latency offset
        self.spikes.value = bm.abs(local_time - self.spike_times) < dt
        return self.spikes
