import brainpy as bp
import brainpy.math as bm

class R_STDP(bp.dyn.SynConn):
    """
    Reward-Modulated Spike-Timing-Dependent Plasticity (R-STDP).
    Requires a neuromodulatory signal (dopamine/reward).
    """
    def __init__(self, pre, post, conn, tau_stdp=20., tau_elig=50., w_init=1.0, w_max=2.0, **kwargs):
        super(R_STDP, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
        
        # Connection indices
        self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
        
        # Weights and Traces
        self.w = bm.Variable(bm.full(self.pre_ids.shape, w_init))
        self.w_max = w_max
        self.tau_stdp = tau_stdp
        self.tau_elig = tau_elig
        
        # Pre/Post traces for STDP
        self.trace_pre = bm.Variable(bm.zeros(self.pre.size))
        self.trace_post = bm.Variable(bm.zeros(self.post.size))
        
        # Eligibility trace for each synapse
        self.E = bm.Variable(bm.zeros(self.pre_ids.shape))

    def update(self, reward=None):
        reward = 0. if reward is None else reward
        dt = getattr(bp.share, 'dt', 0.1)
        
        # Update Pre/Post traces
        self.trace_pre.value += (-self.trace_pre + self.pre.spike.astype(float)) / self.tau_stdp * dt
        self.trace_post.value += (-self.trace_post + self.post.spike.astype(float)) / self.tau_stdp * dt
        
        # Local STDP contribution to Eligibility trace
        # If pre fires, E += -trace_post
        # If post fires, E += trace_pre
        pre_spikes = self.pre.spike[self.pre_ids]
        post_spikes = self.post.spike[self.post_ids]
        
        # Calculate instantaneous STDP update
        dw_inst = (self.trace_pre[self.pre_ids] * post_spikes.astype(float) - 
                   self.trace_post[self.post_ids] * pre_spikes.astype(float))
        
        # Decay and increment eligibility trace
        self.E.value += (-self.E + dw_inst) / self.tau_elig * dt
        
        # Apply Reward modulation to weight update
        # dw = reward * E
        self.w.value = bm.clip(self.w + reward * self.E * dt, 0., self.w_max)
        
        # Post-synaptic current sum (simple delta synapse)
        # unsorted_segment_sum is the JAX-compatible way to handle this
        weighted_spikes = bm.where(self.pre.spike[self.pre_ids], self.w, 0.0)
        num_post = int(self.post.size[0]) if isinstance(self.post.size, (tuple, list)) else int(self.post.size)
        post_input = bm.unsorted_segment_sum(weighted_spikes, self.post_ids, num_post)
        
        # Use input_basal if available (MCN), otherwise use V (Point Neurons)
        if hasattr(self.post, 'input_basal'):
            self.post.input_basal.value += post_input 
        else:
            self.post.V.value += post_input
class STDP(bp.dyn.SynConn):
    """
    Standard Hebbian Spike-Timing-Dependent Plasticity.
    """
    def __init__(self, pre, post, conn, tau_stdp=20., w_init=1.0, w_max=2.0, **kwargs):
        super(STDP, self).__init__(pre=pre, post=post, conn=conn, **kwargs)
        self.pre_ids, self.post_ids = self.conn.require('pre_ids', 'post_ids')
        self.w = bm.Variable(bm.full(self.pre_ids.shape, w_init))
        self.w_max = w_max
        self.tau_stdp = tau_stdp
        self.trace_pre = bm.Variable(bm.zeros(self.pre.size))
        self.trace_post = bm.Variable(bm.zeros(self.post.size))

    def update(self, **kwargs):
        dt = getattr(bp.share, 'dt', 0.1)
        # Pre/Post traces
        self.trace_pre.value += (-self.trace_pre + self.pre.spike.astype(float)) / self.tau_stdp * dt
        self.trace_post.value += (-self.trace_post + self.post.spike.astype(float)) / self.tau_stdp * dt
        # Instantaneous update
        pre_spikes = self.pre.spike[self.pre_ids]
        post_spikes = self.post.spike[self.post_ids]
        dw = (self.trace_pre[self.pre_ids] * post_spikes.astype(float) - 
              self.trace_post[self.post_ids] * pre_spikes.astype(float))
        self.w.value = bm.clip(self.w + dw * dt, 0., self.w_max)
        # Delta Synapse Current
        weighted_spikes = bm.where(self.pre.spike[self.pre_ids], self.w, 0.0)
        num_post = int(self.post.size[0]) if isinstance(self.post.size, (tuple, list)) else int(self.post.size)
        post_input = bm.unsorted_segment_sum(weighted_spikes, self.post_ids, num_post)
        
        if hasattr(self.post, 'input_basal'):
            self.post.input_basal.value += post_input
        else:
            self.post.V.value += post_input

class HomeostaticScaling(bp.dyn.SynConn):
    """
    Maintains a target firing rate by scaling input current.
    """
    def __init__(self, pre, post, target_rate=0.01, tau_avg=1000., **kwargs):
        super(HomeostaticScaling, self).__init__(pre=pre, post=post, **kwargs)
        self.target_rate = target_rate
        self.tau_avg = tau_avg
        self.avg_firing_rate = bm.Variable(bm.full(self.post.size, target_rate))

    def update(self):
        dt = getattr(bp.share, 'dt', 0.1)
        # Moving average of post-synaptic spikes
        self.avg_firing_rate.value += (-self.avg_firing_rate + self.post.spike.astype(float)/dt) / self.tau_avg * dt
        
        # Calculate error and provide feedback current
        error = self.target_rate - self.avg_firing_rate
        if hasattr(self.post, 'input_basal'):
            self.post.input_basal.value += error * 0.1
        else:
            self.post.V.value += error * 0.1
