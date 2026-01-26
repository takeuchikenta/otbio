import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

def findN(pointset, sNum):
    """
    Find K Closest Neighbor Points.
    
    Args:
        pointset: (n, dims) array
        sNum: Number of neighbors (K)
        
    Returns:
        m: (n, sNum) array of indices of the K nearest neighbors
    """
    n = pointset.shape[0]
    # Calculate distance matrix
    # M[i, j] is the distance between point i and point j
    M = distance_matrix(pointset, pointset)
    
    m = np.zeros((n, sNum), dtype=int)
    for i in range(n):
        # Sort distances for point i
        # argsort returns indices that would sort the array
        sorted_indices = np.argsort(M[i, :])
        # Take the K nearest neighbors (excluding the point itself at index 0)
        m[i, :] = sorted_indices[1:sNum+1]
        
    return m

def lap(M):
    """
    Linear Assignment Problem solver.
    Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm).
    
    Args:
        M: Cost matrix
        
    Returns:
        Assignment matrix (binary)
    """
    row_ind, col_ind = linear_sum_assignment(M)
    n = M.shape[0]
    m = M.shape[1]
    assignment = np.zeros((n, m))
    assignment[row_ind, col_ind] = 1
    return assignment

def lap_wrapper(M, threshold):
    """
    Wrapper for LAP to handle outlier cases (not fully implemented in provided MATLAB snippet logic, 
    but mimicking the structure).
    In the provided MATLAB code, lap_wrapper calls a C++ mex function.
    For Python, we can use the same linear_sum_assignment if the matrix is square.
    If not square, linear_sum_assignment still works for rectangular matrices.
    
    Args:
        M: Cost matrix
        threshold: Threshold value (unused in standard LAP but kept for signature compatibility)
        
    Returns:
        Assignment matrix
    """
    # Scipy's linear_sum_assignment handles rectangular matrices automatically.
    return lap(M)

def cal_m(source, Nm, target, Nf, T):
    """
    Calculate Corresponding Matrix.
    
    Args:
        source: (n, dims) array
        Nm: (n, K) array of source neighbor indices
        target: (m, dims) array
        Nf: (m, K) array of target neighbor indices
        T: Temperature parameter
        
    Returns:
        M: Correspondence matrix (binary)
    """
    n, dims = source.shape
    m, _ = target.shape
    
    M0 = np.ones((n, m))
    
    # Pre-compute ones for broadcasting
    ones_m = np.ones((1, m))
    ones_n = np.ones((n, 1))
    
    for i in range(dims):
        # Term 1: Squared distance between all pairs in dimension i
        # source[:, i] is (n,), target[:, i] is (m,)
        # (n, 1) * (1, m) -> (n, m)
        diff = (source[:, i:i+1] @ ones_m - ones_n @ target[:, i:i+1].T)
        a0 = diff ** 2
        a000 = diff
        
        a00 = np.zeros((n, m))
        K = Nm.shape[1]
        
        for j in range(K):
            # Neighbor term
            # source[Nm[:, j], i] gets the i-th coord of the j-th neighbor for each source point
            s_neighbor = source[Nm[:, j], i:i+1] # (n, 1)
            t_neighbor = target[Nf[:, j], i:i+1] # (m, 1)
            
            # (source_neighbor - source_current) - (target_neighbor - target_current)
            # Note: a000 is (source_current - target_current)
            # The formula in MATLAB: 
            # a00=a00+((source(Nm(:,j),i) * ones(1,m)-a000 - ones(n,1) * target(Nf(:,j),i)').^2);
            # Let's break it down:
            # Term A: source(Nm(:,j),i) * ones(1,m) -> (n, m) matrix where each row is the coord of j-th neighbor of source i
            # Term B: a000 -> (n, m) matrix of (source_i - target_k)
            # Term C: ones(n,1) * target(Nf(:,j),i)' -> (n, m) matrix where each col is the coord of j-th neighbor of target k
            
            term_a = s_neighbor @ ones_m
            term_c = ones_n @ t_neighbor.T
            
            a00 += (term_a - a000 - term_c) ** 2
            
        M0 = M0 + a0 + (K**2) * T * a00
        
    # Solve assignment problem
    # In MATLAB: M=round(M0*1e6); M=lap(M);
    # We keep it as float for scipy, but maybe scaling helps numerical stability?
    # The MATLAB code scales by 1e6 and rounds, likely for integer-based LAP solver or precision issues.
    # Scipy's solver works with floats.
    
    if n == m:
        M_out = lap(M0)
    else:
        # Outlier case
        M_out = lap_wrapper(M0, 1e9)
        
    return M_out

def cal_K(x, z):
    """
    Calculate Kernel Matrix K.
    
    Args:
        x: (n, M) array (augmented with ones)
        z: (m, M) array (augmented with ones)
        
    Returns:
        K: (n, m) Kernel matrix
    """
    n, M_dim = x.shape
    m, _ = z.shape
    dim = M_dim - 1 # Original dimension (2 or 3)
    
    K = np.zeros((n, m))
    
    # x and z have 1s in the first column. The coordinates start from index 1.
    for it_dim in range(dim):
        # x[:, it_dim+1] is the coordinate
        tmp = x[:, it_dim+1:it_dim+2] @ np.ones((1, m)) - np.ones((n, 1)) @ z[:, it_dim+1:it_dim+2].T
        tmp = tmp * tmp
        K = K + tmp
        
    mask = K < 1e-10
    
    if dim == 2:
        # K = r^2 * log(r)
        # Here K is r^2. So we want K * log(sqrt(K)) = K * 0.5 * log(K)
        # MATLAB: K = 0.5 * K .* log(K + mask) .* (K>1e-10);
        # Note: log(K + mask) handles the 0 case by adding a small value (mask is boolean 0/1, but here used as value?)
        # In MATLAB mask is logical. K+mask adds 1 to 0 entries, making log(1)=0.
        
        K_log = np.log(K + mask.astype(float))
        K = 0.5 * K * K_log * (~mask).astype(float)
    else:
        # For 3D, typically K = -r = -sqrt(K_squared)
        # But the MATLAB code says:
        # %K = - sqrt(K).* (K>1e-10);                % For Face3D
        # K = 0.5 * K .* log(K + mask) .* (K>1e-10); % For 2D Demo cases
        # It seems the provided code uses the 2D kernel even for 3D demo?
        # Let's stick to the provided code logic.
        K_log = np.log(K + mask.astype(float))
        K = 0.5 * K * K_log * (~mask).astype(float)
        
    return K

def cal_QR(x):
    """
    Calculate QR decomposition parts.
    
    Args:
        x: (n, M) matrix
        
    Returns:
        q1: (n, M)
        q2: (n, n-M)
        R: (M, M)
    """
    n, M = x.shape
    # numpy qr returns q (n, n) and r (n, M) by default with mode='complete'
    q, r = np.linalg.qr(x, mode='complete')
    
    q1 = q[:, :M]
    q2 = q[:, M:]
    R = r[:M, :M]
    
    return q1, q2, R

def cal_wd(lamda, q1, q2, R, K, y):
    """
    Calculate TPS parameters w and d.
    
    Args:
        lamda: Regularization parameter
        q1, q2, R: QR decomposition parts
        K: Kernel matrix
        y: Target points (augmented)
        
    Returns:
        w: Weights
        d: Affine coefficients
    """
    n, M = y.shape
    
    # gamma = (q2'*K*q2 + lamda*eye(n-M)) \ q2' * y
    # A \ B is solve(A, B)
    
    A = q2.T @ K @ q2 + lamda * np.eye(n - M)
    B = q2.T @ y
    
    gamma = np.linalg.solve(A, B)
    
    w = q2 @ gamma
    
    # d = pinv(R) * q1' * (y - K*q2*gamma)
    # y - K*w
    term = y - K @ w
    d = np.linalg.pinv(R) @ q1.T @ term
    
    return w, d

def update_tps(x, y, lamda):
    """
    Update TPS transformation.
    
    Args:
        x: Source points (n, dim)
        y: Target/Correspondence points (n, dim)
        lamda: Regularization parameter
        
    Returns:
        w: Weights
        d: Affine coefficients
        K: Kernel matrix
    """
    n = x.shape[0]
    # Augment x with ones
    x_aug = np.hstack((np.ones((n, 1)), x))
    
    m = y.shape[0]
    # Augment y with ones
    y_aug = np.hstack((np.ones((m, 1)), y))
    
    K = cal_K(x_aug, x_aug)
    q1, q2, R = cal_QR(x_aug)
    w, d = cal_wd(lamda, q1, q2, R, K, y_aug)
    
    return w, d, K

def update_xw(x, w, d, K):
    """
    Update warped source points.
    
    Args:
        x: Source points (n, dim)
        w: Weights
        d: Affine coefficients
        K: Kernel matrix (calculated on x)
        
    Returns:
        xx: Warped points (n, dim)
    """
    n, dim = x.shape
    x_aug = np.hstack((np.ones((n, 1)), x))
    
    # xx = x*d + K*w
    # Note: x_aug is (n, M), d is (M, M) -> (n, M)
    # K is (n, n), w is (n, M) -> (n, M)
    
    xx = x_aug @ d + K @ w
    
    # Remove the first column (ones)
    xx = xx[:, 1:dim+1]
    
    return xx

def TPS3D(points, ctrlpoints, object_pts):
    """
    3D Thin Plate Spline Warping Function (for generating demo data).
    
    Args:
        points: Control points (n, 3)
        ctrlpoints: Deformed control points (n, 3)
        object_pts: Points to be warped (m, 3)
        
    Returns:
        wobject: Warped object points
    """
    npnts = points.shape[0]
    
    # Calculate K for control points
    # K[i, j] = ||points[i] - points[j]||^2
    K = distance_matrix(points, points) ** 2
    K = np.maximum(K, 1e-320)
    K = np.sqrt(K) # R
    
    # P matrix: [1, x, y, z]
    P = np.hstack((np.ones((npnts, 1)), points))
    
    # L matrix
    # [ K  P ]
    # [ P' 0 ]
    zeros_4x4 = np.zeros((4, 4))
    L_top = np.hstack((K, P))
    L_bot = np.hstack((P.T, zeros_4x4))
    L = np.vstack((L_top, L_bot))
    
    # param = pinv(L) * [ctrlpoints; zeros(4,3)]
    rhs = np.vstack((ctrlpoints, np.zeros((4, 3))))
    param = np.linalg.pinv(L) @ rhs
    
    # Calculate new coordinates
    pntsNum = object_pts.shape[0]
    
    # K for object points
    # K[i, j] = ||object[i] - points[j]||
    K_obj = distance_matrix(object_pts, points)
    K_obj = np.maximum(K_obj, 1e-320)
    # K_obj is already sqrt(dist^2) from distance_matrix
    
    P_obj = np.hstack((np.ones((pntsNum, 1)), object_pts))
    
    L_obj = np.hstack((K_obj, P_obj))
    
    wobject = L_obj @ param
    
    return wobject

def glmdtps_registration(Mpoints, Fpoints, max_iter=100, visualize=False):
    """
    Main GLMDTPS registration loop.
    
    Args:
        Mpoints: Source points (n, dim)
        Fpoints: Target points (m, dim)
        max_iter: Maximum iterations
        visualize: Boolean to show plot
        
    Returns:
        Mpoints_transformed: Transformed source points
    """
    # Initial Parameters
    Source = Mpoints.copy()
    Target = Fpoints.copy()
    
    x = Mpoints.copy() # source
    y = Fpoints.copy() # target
    xw = x.copy()      # Initial x^w
    
    # Annealing Parameter
    lambda_init = len(x)
    anneal_rate = 0.93 # Slightly higher than 0.7 for stability in demo? MATLAB uses 0.7
    anneal_rate = 0.7
    
    # Set 5 closest neighbor points
    K_neighbors = 5
    Nm = findN(Mpoints, K_neighbors)
    Nf = findN(Fpoints, K_neighbors)
    
    # Calculate initial T
    # T = sqrt(max(max(dis)))/10
    # dis matrix of squared distances
    dis = distance_matrix(Mpoints, Fpoints) ** 2
    T = np.sqrt(np.max(dis)) / 10.0
    
    # Calculate T_final
    # Tmax = distance_matrix(Mpoints, Mpoints) ** 2
    # For each point, find min distance to others (excluding self)
    # In MATLAB: Tmax(i,ind)=1000; then min.
    # Here we can just fill diagonal with infinity
    Tmax = distance_matrix(Mpoints, Mpoints) ** 2
    np.fill_diagonal(Tmax, np.inf)
    min_dists = np.min(Tmax, axis=1)
    T_final = np.sum(min_dists) / (len(Mpoints) * 8)
    
    lambda_val = lambda_init * len(x) * T
    
    step = 0
    flag_stop = False
    
    if visualize:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    while not flag_stop and step < max_iter:
        if T < T_final:
            flag_stop = True
            
        if visualize:
            ax.clear()
            ax.scatter(Fpoints[:,0], Fpoints[:,1], Fpoints[:,2], c='y', marker='o', label='Target')
            ax.scatter(xw[:,0], xw[:,1], xw[:,2], c='g', marker='+', label='Source (Warped)')
            ax.set_title(f'Iteration: {step}, T: {T:.4f}')
            ax.legend()
            plt.pause(0.1)
            
        # Calculate two-way corresponding matrix M
        m_corr = cal_m(xw, Nm, y, Nf, T)
        
        # Update the correspondence x^c
        xc = m_corr @ y
        
        # Update TPS transformation
        lambda_val = lambda_init * T
        w, d, K_tps = update_tps(x, xc, lambda_val)
        
        # Update the warping template x^w
        xw = update_xw(x, w, d, K_tps)
        
        # Reduce T
        T = T * anneal_rate
        step += 1
        
    return xw

if __name__ == "__main__":
    # Demo
    print("Running GLMDTPS Demo...")
    
    # Generate fish-like data (using random points for simplicity as we don't have fish.mat)
    # Create a 3D surface or curve
    t = np.linspace(0, 2*np.pi, 100)
    x = np.sin(t)
    y = np.cos(t)
    z = t / (2*np.pi)
    Mpoints = np.column_stack((x, y, z))
    
    # Rescale to [0, 1]
    for i in range(3):
        Mpoints[:, i] = (Mpoints[:, i] - np.min(Mpoints[:, i])) / (np.max(Mpoints[:, i]) - np.min(Mpoints[:, i]))
        
    # Generate deformation
    # Control points
    points = np.array([
        [0, 0, 0], [0.5, 0, 0], [1, 0, 0],
        [1, 0.5, 0], [1, 1, 0], [0.5, 1, 0],
        [0, 1, 0], [0, 0.5, 0]
    ])
    
    # Deform control points randomly
    wpoints = points.copy()
    wpoints[0] += [0.1, 0.1, 0]
    wpoints[4] += [-0.1, -0.1, 0]
    
    # Apply TPS3D to get Fpoints (Target)
    Fpoints = TPS3D(points, wpoints, Mpoints)
    
    # Shuffle Fpoints to make it harder (loss of correspondence)
    np.random.shuffle(Fpoints)
    
    # Run Registration
    result = glmdtps_registration(Mpoints, Fpoints, visualize=False)
    
    # Calculate Error
    # Since we shuffled, we can't compare point-to-point directly without known correspondence.
    # But in this demo, we can just check if the shapes align visually or by nearest neighbor distance.
    
    # Nearest neighbor distance from Result to Target
    dists = distance_matrix(result, Fpoints)
    min_dists = np.min(dists, axis=1)
    mean_error = np.mean(min_dists)
    
    print(f"Registration Complete. Mean Nearest Neighbor Error: {mean_error:.6f}")
    
    # Visualization
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(Mpoints[:,0], Mpoints[:,1], Mpoints[:,2], c='b', marker='.', label='Source')
    ax1.scatter(Fpoints[:,0], Fpoints[:,1], Fpoints[:,2], c='r', marker='.', label='Target')
    ax1.set_title("Before Registration")
    ax1.legend()
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(result[:,0], result[:,1], result[:,2], c='g', marker='.', label='Registered')
    ax2.scatter(Fpoints[:,0], Fpoints[:,1], Fpoints[:,2], c='r', marker='.', label='Target')
    ax2.set_title("After Registration")
    ax2.legend()
    
    plt.show()
