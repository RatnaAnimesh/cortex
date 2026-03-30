import numpy as np

def victor_purpura_distance(t_pre, t_post, q=1.0):
    """
    Victor-Purpura spike train distance.
    t_pre, t_post: List or array of spike times.
    q: Cost per unit of time shift.
    """
    n_pre = len(t_pre)
    n_post = len(t_post)
    
    if n_pre == 0: return n_post
    if n_post == 0: return n_pre
    
    # DP matrix
    d = np.zeros((n_pre + 1, n_post + 1))
    
    # Boundary conditions
    for i in range(n_pre + 1): d[i, 0] = i
    for j in range(n_post + 1): d[0, j] = j
    
    # Fill matrix
    for i in range(1, n_pre + 1):
        for j in range(1, n_post + 1):
            d[i, j] = min(
                d[i-1, j] + 1, # Deletion
                d[i, j-1] + 1, # Insertion
                d[i-1, j-1] + q * abs(t_pre[i-1] - t_post[j-1]) # Shift
            )
            
    return d[n_pre, n_post]

def van_rossum_distance(t_pre, t_post, tau=20.0):
    """
    Van Rossum spike train distance using an exponential kernel.
    t_pre, t_post: Array of spike times.
    tau: Decay time of the kernel.
    """
    # Convolve with exponential kernel
    # This is often computed numerically or analytically
    # Simple approximation using sum of exponential difference
    
    dist = 0
    # Analytic form for exp kernel distance
    # D^2 = sum_i sum_j (exp(-|ti-tj|/tau))
    def self_dist(t):
        s = 0
        for i in range(len(t)):
            for j in range(len(t)):
                s += np.exp(-abs(t[i] - t[j]) / tau)
        return s
    
    def cross_dist(t1, t2):
        s = 0
        for i in range(len(t1)):
            for j in range(len(t2)):
                s += np.exp(-abs(t1[i] - t2[j]) / tau)
        return s
        
    d2 = self_dist(t_pre) + self_dist(t_post) - 2 * cross_dist(t_pre, t_post)
    return np.sqrt(max(0, d2))

if __name__ == "__main__":
    s1 = [10, 20, 30]
    s2 = [11, 21, 31]
    print(f"VP Distance (q=1.0): {victor_purpura_distance(s1, s2, q=1.0)}")
    print(f"Van Rossum Distance (tau=20.0): {van_rossum_distance(np.array(s1), np.array(s2), tau=20.0)}")
