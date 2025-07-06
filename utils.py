"""
Utilities for sparse Wasserstein distance computation and optimization.
This module provides implementations of sparse matrix operations, optimization routines
for efficient distance computation using mirror descent.
"""

import numpy as np
from scipy.sparse import dia_matrix
from scipy.optimize import minimize, Bounds
from typing import List
from wasserstein import NMRSpectrum

# ------------------------------------------------------------------------------
# Sparse Matrix Operations
# ------------------------------------------------------------------------------


def multidiagonal_cost(v1, v2, C):
    """
    Construct a multidiagonal sparse matrix where M[i,j] = abs(v1[i] - v2[j]) if abs(i-j) < C.
    Parameters:
        v1, v2 (array-like): Input vectors (must be same length)
        C (int): Bandwidth parameter controlling how many diagonals are non-zero

    Returns:
        dia_matrix: The constructed sparse matrix in DIAgonal format
    """
    n = len(v1)
    assert len(v2) == n, "v1 and v2 must have the same length"

    offsets = np.arange(-C + 1, C)  # Diagonals from -C+1 to C-1

    # Preallocate the data array
    data = np.zeros((len(offsets), n))

    # For each diagonal offset
    for i, k in enumerate(offsets):
        # Calculate valid indices for this diagonal
        i_min = max(0, -k)
        i_max = min(n, n - k)
        diag_length = i_max - i_min

        # Calculate values for this diagonal
        diag_values = np.abs(v1[i_min:i_max] - v2[i_min + k : i_max + k])

        # Place values in the correct position with proper padding
        if k < 0:
            # Lower diagonal: place values at the beginning, pad at the end
            data[i, :diag_length] = diag_values
        else:
            # Upper diagonal: place values at the end, pad at the beginning
            data[i, n - diag_length :] = diag_values

    return dia_matrix((data, offsets), shape=(n, n))


def reg_distribiution(N, C):
    """
    Construct a multidiagonal sparse matrix of regularization coefficients.

    Parameters:
        N (int): Size of the matrix
        C (int): Bandwidth parameter controlling how many diagonals are non-zero

    Returns:
        dia_matrix: The constructed sparse matrix in DIAgonal format
    """
    offsets = np.arange(-C + 1, C)  # Diagonals from -C+1 to C-1

    # Preallocate the data array
    data = np.zeros((len(offsets), N))

    # Calculate the value once outside the loop
    value = 1 / (N * (2 * C - 1) - C * (C - 1))

    # For each diagonal offset
    for i, k in enumerate(offsets):
        diag_length = N - np.abs(k)

        # Place values in the correct position with proper padding
        if k < 0:
            # Lower diagonal: place values at the beginning, pad at the end
            data[i, :diag_length] = value
        else:
            # Upper diagonal: place values at the end, pad at the beginning
            data[i, N - diag_length :] = value

    return dia_matrix((data, offsets), shape=(N, N))


def warmstart_sparse(p1, p2, C):
    """
    Create a warm-start sparse multidiagonal transport plan where:
    - Diagonal (i,i) gets min(p1[i], p2[i])
    - Remaining mass is distributed to adjacent cells (i±k) within bandwidth C

    Args:
        p1 (np.array): Source distribution (size n)
        p2 (np.array): Target distribution (size n)
        C (int): Bandwidth (max allowed offset from diagonal)

    Returns:
        dia_matrix: Sparse multidiagonal matrix of shape (n,n)
    """
    n = len(p1)
    assert len(p2) == n, "p1 and p2 must have same length"

    # Initialize diagonals with proper padding
    offsets = list(range(-C + 1, C))

    # Create storage for diagonals with proper padding
    diagonals = {}
    for offset in offsets:
        diagonals[offset] = np.zeros(n)

    for i in range(n):
        remaining_mass = p1[i]

        # Step 1: Assign to diagonal (i,i)
        assign = min(remaining_mass, p2[i])
        diagonals[0][i] = assign
        remaining_mass -= assign

        # Step 2: Distribute remaining mass to adjacent cells
        radius = 1
        while remaining_mass > 1e-10 and radius < C:
            # Right neighbor (i, i+radius)
            if i + radius < n:
                idx = i + radius
                assign = min(remaining_mass, p2[idx])
                diagonals[radius][idx] += assign
                remaining_mass -= assign

            # Left neighbor (i, i-radius)
            if i - radius >= 0 and remaining_mass > 1e-10:
                idx = i - radius
                assign = min(remaining_mass, p2[idx])
                diagonals[-radius][idx] += assign
                remaining_mass -= assign

            radius += 1

    diagonals = [diagonals[offset] for offset in offsets]
    sum = np.sum(diagonals)
    diagonals /= sum

    return dia_matrix((np.array(diagonals), offsets), shape=(n, n), dtype=np.float64)


def flatten_multidiagonal(matrix_data, offsets):
    """
    Flatten a multidiagonal matrix data into a 1D array.

    Parameters:
        matrix_data: Array of diagonal data
        offsets: Array of diagonal offsets

    Returns:
        np.array: Flattened 1D array
    """
    flat_vector = []

    for diag, offset in zip(matrix_data, offsets):
        if offset < 0:
            flat_vector.extend(diag[:offset])
        else:
            flat_vector.extend(diag[offset:])

    return np.array(flat_vector)


def reconstruct_multidiagonal(flat_vector, offsets, N):
    """
    Reconstruct the diagonal data from the flat vector.

    Parameters:
        flat_vector: Flat array of meaningful diagonal values
        offsets: The offsets array
        N: Size of the matrix (N x N)

    Returns:
        List of arrays containing the diagonal data
    """
    M_data = []
    ptr = 0

    for offset in offsets:
        diag_length = N - abs(offset)
        diag_values = flat_vector[ptr : ptr + diag_length]

        # Reconstruct the diagonal with proper padding
        if offset < 0:
            padded_diag = np.pad(diag_values, (0, abs(offset)), mode="constant")
        else:
            padded_diag = np.pad(diag_values, (abs(offset), 0), mode="constant")

        M_data.append(padded_diag)
        ptr += diag_length

    return np.array(M_data)


# ------------------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------------------


def signif_features(spectrum, n_features):
    """
    Extract the most significant features from a spectrum.

    Parameters:
        spectrum (NMRSpectrum): Input spectrum
        n_features (int): Number of features to extract

    Returns:
        NMRSpectrum: Spectrum with only the significant features
    """
    spectrum_confs = sorted(spectrum.confs, key=lambda x: x[1], reverse=True)[
        :n_features
    ]
    spectrum_signif = NMRSpectrum(confs=spectrum_confs, protons=spectrum.protons)
    spectrum_signif.normalize()
    return spectrum_signif


def get_nus(s_list):
    tagged_s = []
    for i, s in enumerate(s_list):
        tagged_s.extend([(v, p, f"s{i + 1}") for v, p in s])

    unique_v = sorted({v for v, _, _ in tagged_s})

    nus = [np.zeros(len(unique_v)) for _ in s_list]

    v_to_idx = {v: idx for idx, v in enumerate(unique_v)}

    for v, p, source in tagged_s:
        idx = v_to_idx[v]
        src_idx = int(source[1:]) - 1
        nus[src_idx][idx] = p

    return nus


# ------------------------------------------------------------------------------
# Optimization Classes
# ------------------------------------------------------------------------------


class UtilsSparse:
    """
    Utilities for sparse matrix optimization of optimal transport.

    This class implements optimization routines for unbalanced optimal transport
    using sparse multidiagonal matrices.
    """

    def __init__(self, spectra, mix, N, C, reg, reg_m1, reg_m2):
        """
        Initialize the sparse optimization utilities.

        Parameters:
            a (np.array): Source distribution
            b (np.array): Target distribution
            c (dia_matrix): Reference distribution for KL regularization
            G0_sparse (dia_matrix): Initial transport plan
            M_sparse (dia_matrix): Cost matrix
            reg (float): KL regularization strength
            reg_m1 (float): Regularization for source marginal
            reg_m2 (float): Regularization for target marginal
        """
        spectra = [signif_features(spectrum, N) for spectrum in spectra]

        ratio = np.ones(len(spectra)) / len(spectra)

        unique_v = sorted({v for spectrum in spectra for v, _ in spectrum.confs})
        total_unique_v = len(unique_v)

        mix_og = signif_features(mix, total_unique_v)

        nus = get_nus([si.confs for si in spectra])

        a = np.array([p for _, p in mix_og.confs])
        b = sum(nu_i * p_i for nu_i, p_i in zip(nus, ratio))

        v1 = np.array([v for v, _ in mix_og.confs])
        v2 = unique_v

        assert len(v1) == len(v2), "Only square multidiagonal matrices supported"

        M = multidiagonal_cost(v1, v2, C)
        G0_sparse = warmstart_sparse(a, b, C)
        c = reg_distribiution(total_unique_v, C)

        self.a = a
        self.b = b
        self.c_data = c.data
        self.m, self.n = M.shape

        assert len(a) == len(b) == self.m, "Marginals must match matrix dimensions"

        self.G0_sparse = G0_sparse
        self.offsets = M.offsets
        self.data = M.data

        self.reg = reg
        self.reg_m1 = reg_m1
        self.reg_m2 = reg_m2

        self.nus = nus

    def sparse_dot(self, G_data, G_offsets):
        """
        Efficient dot product between two multidiagonal matrices.

        Parameters:
            G_data: Diagonal data of the second matrix
            G_offsets: Offsets of the second matrix

        Returns:
            float: Dot product value
        """
        total = 0.0
        for i, offset1 in enumerate(self.offsets):
            for j, offset2 in enumerate(G_offsets):
                if offset1 == offset2:
                    min_len = min(len(self.data[i]), len(G_data[j]))
                    total += np.sum(self.data[i][:min_len] * G_data[j][:min_len])
        return total

    def sparse_row_sum(self, G_data, G_offsets):
        """Row sums for multidiagonal matrix (square matrix version)"""
        row_sums = np.zeros(self.m)
        for offset, diag in zip(G_offsets, G_data):
            if offset == 0:  # Main diagonal
                row_sums += diag[: self.m]
            elif offset < 0:  # Lower diagonal
                k = -offset
                row_sums[k:] += diag[: self.m - k]
            else:  # Upper diagonal
                row_sums[:-offset] += diag[offset : offset + self.m - offset]
        return row_sums

    def sparse_col_sum(self, G_data, _G_offsets):
        """Column sums for multidiagonal matrix (square matrix version)"""
        col_sums = np.zeros(self.m)
        for diag in G_data:
            col_sums += diag
        return col_sums

    def reg_kl_sparse(self, G_data, G_offsets):
        G_flat = flatten_multidiagonal(G_data, G_offsets)
        C_flat = flatten_multidiagonal(self.c_data, G_offsets)

        return (
            np.sum(G_flat * np.log(G_flat / C_flat + 1e-16))
            + np.sum(C_flat - G_flat) * self.reg
        )

    def grad_kl_sparse(self, G_data, G_offsets):
        """Gradient of KL divergence for sparse matrix"""
        G_flat = flatten_multidiagonal(G_data, G_offsets)
        C_flat = flatten_multidiagonal(self.c_data, G_offsets)

        grad_flat = (np.log(G_flat / C_flat + 1e-16) + 1) * self.reg
        return reconstruct_multidiagonal(grad_flat, G_offsets, self.m) * self.reg

    def marg_tv_sparse(self, G_data, G_offsets):
        """TV marginal penalty for sparse matrix"""
        row_sums = self.sparse_row_sum(G_data, G_offsets)
        col_sums = self.sparse_col_sum(G_data, G_offsets)
        return self.reg_m1 * np.sum(np.abs(row_sums - self.a)) + self.reg_m2 * np.sum(
            np.abs(col_sums - self.b)
        )

    def marg_tv_sparse_rm1(self, G_data, G_offsets):
        """TV marginal penalty for sparse matrix"""
        row_sums = self.sparse_row_sum(G_data, G_offsets)
        return self.reg_m1 * np.sum(np.abs(row_sums - self.a))

    def marg_tv_sparse_rm2(self, G_data, G_offsets):
        """TV marginal penalty for sparse matrix"""
        col_sums = self.sparse_col_sum(G_data, G_offsets)
        return self.reg_m2 * np.sum(np.abs(col_sums - self.b))

    def grad_marg_tv_sparse(self, G_data, G_offsets):
        """Gradient of TV marginal penalty"""
        row_sums = self.sparse_row_sum(G_data, G_offsets)
        col_sums = self.sparse_col_sum(G_data, G_offsets)

        # Compute sign terms for rows and columns
        row_signs = self.reg_m1 * np.sign(row_sums - self.a)
        col_signs = self.reg_m2 * np.sign(col_sums - self.b)

        grad_data = []
        for offset in G_offsets:
            if offset == 0:  # Main diagonal
                grad_diag = row_signs + col_signs
            elif offset < 0:  # Lower diagonal
                k = -offset
                grad_diag = row_signs[k:] + col_signs[: self.m - k]
                grad_diag = np.pad(grad_diag, (0, k), "constant")
            else:  # Upper diagonal
                grad_diag = row_signs[: self.m - offset] + col_signs[offset:]
                grad_diag = np.pad(grad_diag, (offset, 0), "constant")
            grad_data.append(grad_diag)

        return np.array(grad_data)

    def func_sparse(self, G_flat):
        """Combined loss function and gradient for sparse optimization"""
        # Reconstruct sparse matrix from flattened representation
        G_data = reconstruct_multidiagonal(G_flat, self.offsets, self.m)

        # Compute loss
        transport_cost = self.sparse_dot(G_data, self.offsets)
        marginal_penalty = self.marg_tv_sparse(G_data, self.offsets)
        val = transport_cost + marginal_penalty
        if self.reg > 0:
            val += self.reg_kl_sparse(G_data, self.offsets)

        # Compute gradient
        grad = self.data + self.grad_marg_tv_sparse(G_data, self.offsets)

        if self.reg > 0:
            grad += self.grad_kl_sparse(G_data, self.offsets)

        grad_flat = flatten_multidiagonal(grad, self.offsets)

        return val, grad_flat

    def lbfgsb_unbalanced(self, numItermax=1000, stopThr=1e-15):
        _func = self.func_sparse

        # panic for now
        if self.c is None:
            raise ValueError("Reference distribution 'c' must be provided for unbalanced OT")
        if self.G0_sparse is None:
            raise ValueError("Initial transport plan 'G0' must be provided")

        res = minimize(
            _func,
            flatten_multidiagonal(self.G0_sparse.data, self.G0_sparse.offsets),
            method="L-BFGS-B",
            jac=True,
            bounds=Bounds(0, np.inf),
            tol=stopThr,
            options={
                # "ftol": 1e-12,
                # "gtol": 1e-8,
                "maxiter": numItermax
            },
        )

        G = reconstruct_multidiagonal(res.x, self.G0_sparse.offsets, self.m)

        log = {
            "total_cost": res.fun,
            "cost": self.sparse_dot(G, self.offsets),
            "res": res,
        }
        return G, log

    def mirror_descent_unbalanced(
        self, numItermax=1000, step_size=0.0001, stopThr=1e-6, gamma=1.0, patience=50
    ):
        """
        Solves the unbalanced OT problem using mirror descent with exponential updates.

        Parameters:
            numItermax (int): Maximum number of iterations
            step_size (float): Fixed step size for gradient updates
            stopThr (float): Stopping threshold for relative change in objective
            patience (int): Number of iterations with no improvement to wait before stopping.

        Returns:
            tuple: (G, log) where G is the optimal transport plan and log contains information about the optimization
        """
        if self.G0_sparse is None:
            raise ValueError("Initial transport plan 'G0' must be provided")

        G_offsets = self.G0_sparse.offsets.copy()
        G_flat = flatten_multidiagonal(self.G0_sparse.data, G_offsets)

        val_prev, grad_flat = self.func_sparse(G_flat)

        log = {
            "loss": [val_prev],
        }

        stalled_iterations = 0

        for i in range(numItermax):
            grad_flat_clipped = np.clip(grad_flat, -100, 100)
            G_w = G_flat * np.exp(-step_size * grad_flat_clipped)
            G_w = np.maximum(G_w, 1e-15)
            G_flat_new = G_w / np.sum(G_w)

            val_new, grad_flat_new = self.func_sparse(G_flat_new)

            rel_change = abs(val_new - val_prev) / max(abs(val_prev), 1e-10)
            log["loss"].append(val_new)

            if rel_change < stopThr:
                stalled_iterations += 1
            else:
                stalled_iterations = 0

            if stalled_iterations >= patience:
                log["convergence"] = True
                log["iterations"] = i + 1
                G_flat = G_flat_new
                break

            G_flat = G_flat_new
            grad_flat = grad_flat_new
            val_prev = val_new
            step_size *= gamma

        else:
            log["convergence"] = False
            log["iterations"] = numItermax

        G = reconstruct_multidiagonal(G_flat, G_offsets, self.m)
        log["total_cost"] = val_prev
        log["cost"] = self.sparse_dot(G, G_offsets)

        rm1 = self.marg_tv_sparse_rm1(G, G_offsets)
        rm2 = self.marg_tv_sparse_rm2(G, G_offsets)

        log["final_distance"] = log["cost"] + rm1 / self.reg_m1 + rm2 / self.reg_m2

        return G, log
    

    def compute_p_gradient(self, G_flat):
        """
        Compute the gradient of the objective function with respect to mixing proportions p.

        Parameters:
            G_data (np.array): Diagonal data of the transport plan matrix
            nus (List[np.array]): List of spectral distributions for each component

        Returns:
            np.array: Gradient vector for mixing proportions
        """
        G_data = reconstruct_multidiagonal(G_flat, self.offsets, self.m)
        col_sums = self.sparse_col_sum(G_data, self.offsets)
        sign_diff = np.sign(col_sums - self.b)
        return -self.reg_m2 * np.array([np.dot(nu_i, sign_diff) for nu_i in self.nus])
    

    def joint_md(
        self, eta_G, eta_p, max_iter, tol=1e-6, gamma=1.0, patience=50, verbose=True
    ):
        """
        Joint mirror descent optimization for transport plan and mixing proportions.

        Simultaneously optimizes the transport plan G and mixing proportions p using
        mirror descent with exponential updates. The algorithm alternates between
        updating G (transport plan) and p (mixing proportions) while maintaining
        the probability simplex constraints.

        Parameters:
            eta_G (float): Learning rate for transport plan updates
            eta_p (float): Learning rate for mixing proportion updates
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance for relative change in objective
            gamma (float): Learning rate decay factor (applied each iteration)
            patience (int): Number of iterations with no improvement before stopping

        Returns:
            tuple: (G, p) where G is the optimal transport plan as dia_matrix and
                   p is the optimal mixing proportions as np.array
        """
        n = len(self.nus)
        p = np.ones(n) / n

        G_flat = flatten_multidiagonal(self.G0_sparse.data, self.offsets)

        prev_val = None
        stalled_iterations = 0

        for i in range(max_iter):
            # Update transport plan G
            val, grad = self.func_sparse(G_flat)
            G_flat *= np.exp(-eta_G * grad)
            G_flat /= G_flat.sum()

            # Update mixing proportions p
            grad_p = self.compute_p_gradient(G_flat)
            p *= np.exp(-eta_p * grad_p)
            p /= p.sum()

            # Update target distribution b based on new mixing proportions
            self.b = sum(nu_i * p_i for nu_i, p_i in zip(self.nus, p))

            # Check convergence
            if prev_val is not None:
                rel_change = abs(val - prev_val) / max(abs(prev_val), 1e-10)
                if rel_change < tol:
                    stalled_iterations += 1
                else:
                    stalled_iterations = 0

                if stalled_iterations >= patience:
                    break

            prev_val = val

            # Update learning rates
            eta_G *= gamma
            eta_p *= gamma

            if verbose and i % 20 == 0:
                print(f"Iteration {i}: p = {np.round(p, 4)}")

        G_data = reconstruct_multidiagonal(G_flat, self.offsets, self.m)
        return dia_matrix((G_data, self.offsets), shape=(self.m, self.n)), p
