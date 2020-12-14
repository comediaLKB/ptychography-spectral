import numpy as np

def spectral_init(imgs, probe, k_ill_x, k_ill_y, propagator_fw, propagator_bw, x0, preprocessing, beta, th_factor, n_pow_iter):
    """
    Power iterations to determine the leading eigenvector of the ptychographic matrix.
    Because the ptychographic matrix is often too large to store in memory, we propose a sequential (i.e, non-vectorized) version of the power iteration.
    Inputs:
     - imgs: stack of experimental images
     - probe: illumination function
     - k_ill_x, k_ill_y: shifts of the probe relative to the object
     - propagator_fw, propagator_bw: kernel for forward and backward propagation in free space
     - x0: initial solution
     - preprocessing: preprocessing function for experimental images
         'lu': Eq. (29) with no noise in W. Luo et al., arXiv:1811.04420v1, 2018
         'marchesini': preprocessing function T_a in S. Marchesini et al, Appl. Comput. Harmon. Anal 41, 815-851 (2016)
     - beta: constant for the lower bound of the preprocessed intensity (to prevent large negative eigenvalues)
     - th_factor: threshold factor of the corresponding preprocessing function
     - n_power_iter: number of power iterations
    Subfunctions (not included here):
     - propagate: function applying the kernel for forward and backward propagation in free space 
    """
    L = imgs.shape[2]
    m = imgs.shape[0] * imgs.shape[1]
    sol = x0
    size_x = x0.shape[0]
    size_y = x0.shape[1]
    
    for i_pow in range(n_pow_iter):
        curr_imgs = np.zeros((np.sqrt(m).astype(int), np.sqrt(m).astype(int), L))
        for i_ill in range(L):
            roi_x1 = (size_x / 2 - k_ill_x[i_ill] - np.sqrt(m) / 2).astype(int)
            roi_x2 = (size_x / 2 - k_ill_x[i_ill] + np.sqrt(m) / 2).astype(int)
            roi_y1 = (size_y / 2 - k_ill_y[i_ill] - np.sqrt(m) / 2).astype(int)
            roi_y2 = (size_y / 2 - k_ill_y[i_ill] + np.sqrt(m) / 2).astype(int)
            sol_crop = sol[roi_x1 : roi_x2, roi_y1 : roi_y2]
            img = (np.abs(propagate(probe * sol_crop, propagator_fw)))**2
            curr_imgs[:, :, i_ill] = img
        
        v = np.zeros((size_x, size_y), dtype = complex)
        for i_ill in range(L):
            roi_x1 = (size_x / 2 - k_ill_x[i_ill] - np.sqrt(m) / 2).astype(int)
            roi_x2 = (size_x / 2 - k_ill_x[i_ill] + np.sqrt(m) / 2).astype(int)
            roi_y1 = (size_y / 2 - k_ill_y[i_ill] - np.sqrt(m) / 2).astype(int)
            roi_y2 = (size_y / 2 - k_ill_y[i_ill] + np.sqrt(m) / 2).astype(int)
            sol_roi = sol[roi_x1 : roi_x2, roi_y1 : roi_y2]
            
            if preprocessing == 'lu':
                img = imgs[:, :, i_ill] / np.mean(imgs[:, :, i_ill])
                img = 1 - 1 / (img + 1e-5 * np.mean(img))
                img[img < -beta] = -beta
            elif preprocessing == 'marchesini':
                sorted_desc_unique = (np.unique(imgs[:, :, i_ill], axis = None))[::-1]
                th = sorted_desc_unique[int(np.ceil(len(sorted_desc_unique) * th_factor)-1)]
                img = imgs[:, :, i_ill] > th
            else:
                img = imgs[:, :, i_ill]
            prop = propagate(probe * sol_roi, propagator_fw)
            v[roi_x1 : roi_x2, roi_y1 : roi_y2] += np.conj(probe) * propagate(img * prop, propagator_bw)

        sol /= np.linalg.norm(sol)
     
    return sol


def gradient_descent(imgs, probe, k_ill_x, k_ill_y, propagator_fw, propagator_bw, x0, n_iter=100, step_size=1e-2):
    """
    Gradient descent to solve the ptychographic problem with an amplitude loss function.
    Because the ptychographic matrix is often too large to store in memory, we propose a sequential (i.e, non-vectorized) version of the gradient descent implementation.
    Inputs:
     - imgs: stack of experimental images
     - probe: probe
     - k_ill_x, k_ill_y: shifts of the probe relative to the object
     - propagator_fw, propagator_bw: kernel for forward and backward propagation in free space
     - x0: initial solution
     - n_iter: number of iterations. If None, the algorithm will continue until the error is lower than err_final
     - step_size: step size of the gradient descent update
    Subfunctions (not included here):
     - err_exp: function comparing the measurements imgs with the prediction from the current solution
     - propagate: function applying the kernel for forward and backward propagation in free space 
    """
        
    curr_x = x0
    L = imgs.shape[2]
    m = imgs.shape[0] * imgs.shape[1]
    size_x = x0.shape[0]
    size_y = x0.shape[1]

    for i_iter in range(n_iter):
        grad = np.zeros((size_x, size_y), dtype = complex)
        for i_ill in range(L):
            roi_x1 = size_x / 2 - k_ill_x[i_ill] - np.sqrt(m) / 2
            roi_x2 = size_x / 2 - k_ill_x[i_ill] + np.sqrt(m) / 2
            roi_y1 = size_y / 2 - k_ill_y[i_ill] - np.sqrt(m) / 2
            roi_y2 = size_y / 2 - k_ill_y[i_ill] + np.sqrt(m) / 2
            curr_x_crop = curr_x[roi_x1 : roi_x2, roi_y1 : roi_y2]
            
            img = imgs[:, :, i_ill]
            amp_est = propagate(probe * curr_x_crop, propagator_fw)
            diff_amp = np.abs(amp_est) - np.sqrt(img)
            # Gradient of amplitude loss function
            grad[roi_x1: roi_x2, roi_y1: roi_y2] += np.conj(probe) * \
                                                    propagate(diff_amp * np.exp(1j * np.angle(amp_est)),
                                                              propagator_bw) / L / np.amax(np.abs(probe) ** 2)

        curr_x = curr_x - step_size * grad.view(np.complex128)

    return curr_x