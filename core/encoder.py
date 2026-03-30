import brainpy as bp
import brainpy.math as bm
import numpy as np

class SpikeTimeLatencyEncoder(bp.dyn.DynamicalSystem):
    """
    Encodes linguistic tokens into spatiotemporal spike patterns.
    Uses Latency Coding: Earlier spikes = Higher Activation.
    """
    def __init__(self, vocab_size=1000, embedding_dim=128, time_window=10.0, name=None):
        super(SpikeTimeLatencyEncoder, self).__init__(name=name)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.time_window = time_window # ms

        # Static embedding matrix (randomly initialized or from pre-trained)
        # In a real LLM replacement, this would be learned
        self.embeddings = bm.Variable(bm.random.randn(vocab_size, embedding_dim))
        
        # State: last encoded token and spike times
        self.spike_times = bm.Variable(bm.zeros(embedding_dim))
        self.spikes = bm.Variable(bm.zeros(embedding_dim, dtype=bool))

    def encode(self, token_id):
        """
        Computes spike latencies for a given token.
        Latency = 1.0 / (normalized_activation + epsilon)
        """
        # Get embedding vector
        vec = self.embeddings[token_id]
        
        # Normalize to [0, 1] range for latency mapping
        vec_min, vec_max = bm.min(vec), bm.max(vec)
        normalized = (vec - vec_min) / (vec_max - vec_min + 1e-8)
        
        # Map to latency: High activation -> 0ms offset, Low activation -> time_window offset
        self.spike_times.value = (1.0 - normalized) * self.time_window
        return self.spike_times

    def update(self, current_time):
        """
        Checks which dimensions should spike at the current simulation time.
        """
        # A spike occurs if the current time within the token window matches the latency
        # In practice, we use a small delta check or a threshold
        local_time = current_time % self.time_window
        self.spikes.value = bm.abs(local_time - self.spike_times) < getattr(bp.share, 'dt', 0.1)
        return self.spikes

class TokenBuffer(bp.dyn.DynamicalSystem):
    """
    Maintains temporal context using fading eligibility traces.
    Replaces the explicit KV Cache.
    """
    def __init__(self, embedding_dim=128, tau_trace=50.0, name=None):
        super(TokenBuffer, self).__init__(name=name)
        self.trace = bm.Variable(bm.zeros(embedding_dim))
        self.tau_trace = tau_trace

    def update(self, spikes):
        dt = getattr(bp.share, 'dt', 0.1)
        # Exponential decay of context
        self.trace.value += (-self.trace + spikes.astype(float)) / self.tau_trace * dt
        return self.trace
