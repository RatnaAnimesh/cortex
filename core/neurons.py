import brainpy as bp
import brainpy.math as bm

class LIF(bp.dyn.NeuGroup):
    def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., tau=20., **kwargs):
        super(LIF, self).__init__(size=size, **kwargs)

        # parameters
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_th = V_th
        self.tau = tau

        # variables
        self.V = bm.Variable(bm.full(self.size, V_rest))
        self.spike = bm.Variable(bm.zeros(self.size, dtype=bool))
        self.t_last_spike = bm.Variable(bm.full(self.size, -1e10))

    def update(self, x=None):
        # input current
        x = 0. if x is None else x
        dt = getattr(bp.share, 'dt', 0.1)
        
        # update potential
        dv = (self.V_rest - self.V + x) / self.tau * dt
        self.V += dv
        
        # check spikes
        self.spike.value = self.V >= self.V_th
        self.t_last_spike.value = bm.where(self.spike, getattr(bp.share, 't', 0.0), self.t_last_spike)
        self.V.value = bm.where(self.spike, self.reset_v(), bm.maximum(self.V, -90.0))

    def reset_v(self):
        return self.V_reset

class Izhikevich(bp.dyn.NeuGroup):
    """
    Izhikevich model for diverse firing patterns.
    dv/dt = 0.04v^2 + 5v + 140 - u + I
    du/dt = a(bv - u)
    """
    def __init__(self, size, a=0.02, b=0.2, c=-65., d=8., **kwargs):
        super(Izhikevich, self).__init__(size=size, **kwargs)
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        
        self.V = bm.Variable(bm.full(self.size, -65.))
        self.U = bm.Variable(self.b * self.V)
        self.spike = bm.Variable(bm.zeros(self.size, dtype=bool))

    def update(self, x=None):
        x = 0. if x is None else x
        dt = getattr(bp.share, 'dt', 0.1)
        
        # Run 2 sub-steps for stability (common in Izhikevich)
        for _ in range(2):
            dv = 0.5 * (0.04 * self.V**2 + 5 * self.V + 140 - self.U + x)
            self.V += dv * dt
        
        du = self.a * (self.b * self.V - self.U)
        self.U += du * dt
        
        self.spike.value = self.V >= 30.
        self.V.value = bm.where(self.spike, self.c, self.V)
        self.U.value = bm.where(self.spike, self.U + self.d, self.U)

class MultiCompartmentNeuron(bp.dyn.NeuGroup):
    """
    Multi-Compartment Neuron (MCN) with Somatic, Apical, and Basal compartments.
    Enables contextual gating as described in the baseline.
    """
    def __init__(self, size, tau=20., V_th=20., **kwargs):
        super(MultiCompartmentNeuron, self).__init__(size=size, **kwargs)
        
        self.tau = tau
        self.V_th = V_th
        
        self.V_soma = bm.Variable(bm.zeros(self.size))
        self.V_apical = bm.Variable(bm.zeros(self.size))
        self.V_basal = bm.Variable(bm.zeros(self.size))
        self.input_basal = bm.Variable(bm.zeros(self.size))
        self.input_apical = bm.Variable(bm.zeros(self.size))
        self.spike = bm.Variable(bm.zeros(self.size, dtype=bool))

    def update(self):
        dt = getattr(bp.share, 'dt', 0.1)
        
        # Dendritic dynamics using input buffers
        self.V_basal.value += ( -self.V_basal + self.input_basal ) / self.tau * dt
        self.V_apical.value += ( -self.V_apical + self.input_apical ) / self.tau * dt
        
        # Somatic integration (Non-linear coincidence detection)
        # Apical input gates Basal input
        soma_input = self.V_basal * (1.0 + bm.sigmoid(self.V_apical))
        self.V_soma.value += ( -self.V_soma + soma_input ) / self.tau * dt
        
        # Reset input buffers for next step
        self.input_basal.value = bm.zeros(self.size)
        self.input_apical.value = bm.zeros(self.size)
        
        self.spike.value = self.V_soma >= self.V_th
        self.V_soma.value = bm.where(self.spike, 0., bm.maximum(self.V_soma, -90.0))
        self.V_basal.value = bm.where(self.spike, 0., bm.maximum(self.V_basal, -90.0))
        self.V_apical.value = bm.where(self.spike, 0., bm.maximum(self.V_apical, -90.0))
