# -*- coding: utf-8 -*-
"""
Created on 30 July 2026

Author: Yunhui Xie

Affiliation: University of Southampton

Licensed under the Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International License.
https://creativecommons.org/licenses/by-nc-nd/4.0/

SPDX-License-Identifier: CC-BY-NC-ND-4.0
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence, Tuple, Union

import cupy as cp
import numpy as np
from scipy.special import wofz

Number = Union[int, float]
VectorInput = Union[Sequence[Number], np.ndarray]

def _pw_chunk_worker(args):

    (p_chunk,
     q_chunk,
     beamlet_waist,
     window_size_x,
     window_size_y,
     displacement_x,
     displacement_y,
     mp_dps,) = args

    try:
        
        import mpmath as mp
        
    except ImportError as exc:
        
        raise RuntimeError("mpmath is required when 'mp_dps' is not None.") from exc

    out = np.empty(p_chunk.shape, dtype=np.complex128)

    with mp.workdps(int(mp_dps)):
        
        w = mp.mpf(beamlet_waist)
        Lx = mp.mpf(window_size_x)
        Ly = mp.mpf(window_size_y)
        lx = mp.mpf(displacement_x)
        ly = mp.mpf(displacement_y)

        sqrt_two = mp.sqrt(2)
        scale = sqrt_two * mp.pi * w

        def axis_factor(delta, window_size, displacement):
            
            b = delta / (sqrt_two * w)
            upper = scale * (window_size + displacement)
            lower = scale * (-window_size + displacement)

            return mp.exp(-(b * b)) * (mp.erf(upper - mp.j * b) - mp.erf(lower - mp.j * b))

        for index, (px, qy) in enumerate(zip(p_chunk, q_chunk)):
            
            dx = mp.mpf(repr(float(px)))
            dy = mp.mpf(repr(float(qy)))
            out[index] = complex(axis_factor(dx, Lx, lx) * axis_factor(dy, Ly, ly))

    return out

class GradCal:

    def __init__(self,
                 x: VectorInput,
                 y: VectorInput,
                 beamlet_waist: Number,
                 beamlet_amplitude: Number,
                 PIB_windows_size_x: Union[Number, VectorInput],
                 PIB_windows_size_y: Union[Number, VectorInput],
                 PIB_windows_displace_x: Union[Number, VectorInput],
                 PIB_windows_displace_y: Union[Number, VectorInput],
                 n_workers: Union[int, None] = None,
                 chunk_size: int = 131_072,
                 mp_dps: Union[int, None] = None,
                 deduplicate_displacements: bool = True,
                 f: Union[Number, None] = None,
                 z: Union[Number, None] = None,
                 K: Union[Number, None] = None,):
        
        self.x, self.y = self._normalize_coordinates(x, y)
        self.num_beamlets = int(self.x.size)

        self.w = float(beamlet_waist)
        self.A = float(beamlet_amplitude)

        if not np.isfinite(self.w) or self.w <= 0:
            
            raise ValueError("'beamlet_waist' must be positive and finite.")
            
        if not np.isfinite(self.A):
            
            raise ValueError("'beamlet_amplitude' must be finite.")

        available_cpus = os.cpu_count() or 1
        requested_workers = ( available_cpus if n_workers is None else int(n_workers))
        
        if requested_workers < 1:
            
            raise ValueError("'n_workers' must be at least 1.")

        self.n_workers = min(requested_workers, available_cpus)
        self.chunk_size = int(chunk_size)
        
        if self.chunk_size < 1:
            
            raise ValueError("'chunk_size' must be at least 1.")

        self.mp_dps = None if mp_dps is None else int(mp_dps)
        
        if self.mp_dps is not None and self.mp_dps < 2:
            
            raise ValueError("'mp_dps' must be at least 2 or None.")

        self.deduplicate_displacements = bool(deduplicate_displacements)

        propagation_values = (f, z, K)
        any_propagation_value = any(value is not None for value in propagation_values)
        all_propagation_values = all(value is not None for value in propagation_values)

        if any_propagation_value and not all_propagation_values:
            raise ValueError("'f', 'z', and 'K' must either all be provided or all be None.")

        self.off_fourier_plane = all_propagation_values
        self.f = None if f is None else float(f)
        self.z = None if z is None else float(z)
        self.K = None if K is None else float(K)

        if self.off_fourier_plane:
            if not np.isfinite(self.f) or self.f == 0:
                raise ValueError("'f' must be finite and non-zero.")
            if not np.isfinite(self.z):
                raise ValueError("'z' must be finite.")
            if not np.isfinite(self.K) or self.K <= 0:
                raise ValueError("'K' must be positive and finite.")
            if self.mp_dps is not None:
                raise ValueError("'mp_dps' is only supported in the Fourier-plane model.")

            self.alpha = (1.0 / (self.w * self.w) + 1j * self.K / (2.0 * self.f))
            wavelength = 2.0 * np.pi / self.K
            self.gamma = np.pi * np.pi / self.alpha + 1j * np.pi * wavelength * self.z
            self.epsilon = float(2.0 * np.real(1.0 / self.gamma))

            if not np.isfinite(self.epsilon) or self.epsilon <= 0:
                raise ValueError("The supplied 'f', 'z', and 'K' produce an invalid aperture integral.")

            self.obj_prefix = (self.A * self.A * np.pi**3
                               / (4.0 * abs(self.alpha)**2 * abs(self.gamma)**2 * self.epsilon))
        else:
            self.alpha = None
            self.gamma = None
            self.epsilon = None
            self.obj_prefix = (self.A * self.A * self.w * self.w * np.pi) / 8.0

        self.grad_prefix = -1

        (self.Lx_vec, self.Ly_vec, self.lx_vec, self.ly_vec,) = self._normalize_pib_parameters(PIB_windows_size_x,
                                                                                               PIB_windows_size_y,
                                                                                               PIB_windows_displace_x,
                                                                                               PIB_windows_displace_y,)
                                                        
        self.n_pib = int(self.Lx_vec.size)
        self._prepare_displacement_pairs()

        self.pw0_list: List[complex] = []
        self.pw_table_list: List[np.ndarray] = []
        self.pw_table = np.zeros((self.num_beamlets, self.num_beamlets), dtype=np.complex128,)

        use_pool = (not self.off_fourier_plane
                    and self.mp_dps is not None
                    and self.n_workers > 1
                    and self._evaluation_deltas.shape[0] > self.chunk_size)

        if use_pool:
            
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                
                self._build_cpu_tables(pool)
                
        else:
            
            self._build_cpu_tables(pool=None)

        self.x_cp = cp.asarray(self.x, dtype=cp.float64)
        self.y_cp = cp.asarray(self.y, dtype=cp.float64)

        self.pw_table_cp_list = [cp.asarray(table, dtype=cp.complex128) for table in self.pw_table_list]
        self.pw_table_cp = cp.asarray(self.pw_table, dtype=cp.complex128,)

        self.obj_prefix_cp = cp.asarray(self.obj_prefix, dtype=cp.float64,)
        self.grad_prefix_cp = cp.asarray(self.grad_prefix, dtype=cp.float64,)
        self.pw0_cp_list = [cp.asarray(value, dtype=cp.complex128) for value in self.pw0_list]

    @staticmethod
    def _normalize_coordinates(x: VectorInput,
                               y: VectorInput,) -> Tuple[np.ndarray, np.ndarray]:
        
        x_vec = np.asarray(x, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64)

        if x_vec.ndim != 1 or y_vec.ndim != 1:
            raise ValueError("'x' and 'y' must be one-dimensional coordinate arrays.")
        if x_vec.size == 0:
            raise ValueError("'x' and 'y' must contain at least one coordinate.")
        if x_vec.size != y_vec.size:
            raise ValueError("'x' and 'y' must have the same length.")
        if not (np.all(np.isfinite(x_vec)) and np.all(np.isfinite(y_vec))):
            raise ValueError("'x' and 'y' must contain only finite values.")

        return x_vec, y_vec

    @staticmethod
    def _normalize_pib_parameters(Lx, Ly, lx, ly,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        
        def to_1d(value):
            
            if np.isscalar(value):
                
                return np.array([value], dtype=np.float64)

            return np.asarray(value, dtype=np.float64).ravel()

        Lx_vec = to_1d(Lx)
        Ly_vec = to_1d(Ly)
        lx_vec = to_1d(lx)
        ly_vec = to_1d(ly)

        if not (Lx_vec.size == Ly_vec.size == lx_vec.size == ly_vec.size):
            
            raise ValueError("'PIB_windows_size_x', 'PIB_windows_size_y', "
                             "'PIB_windows_displace_x', and "
                             "'PIB_windows_displace_y' must have the same length.")
            
        if Lx_vec.size == 0:
            
            raise ValueError("At least one PIB window is required.")
            
        if not (np.all(np.isfinite(Lx_vec))
                and np.all(np.isfinite(Ly_vec))
                and np.all(np.isfinite(lx_vec))
                and np.all(np.isfinite(ly_vec))):
            
            raise ValueError("All PIB window values must be finite.")

        return Lx_vec, Ly_vec, lx_vec, ly_vec

    def _as_phase_vector(self, phases):
        
        phi = cp.asarray(phases, dtype=cp.float64)

        if phi.ndim != 1 or phi.size != self.num_beamlets:
            raise ValueError("'phases' must be a one-dimensional array with "
                             f"{self.num_beamlets} entries.")

        return phi

    def _prepare_displacement_pairs(self):
        
        self._upper_row, self._upper_col = np.triu_indices(self.num_beamlets, k=1,)

        dx = self.x[self._upper_row] - self.x[self._upper_col]
        dy = self.y[self._upper_row] - self.y[self._upper_col]

        if dx.size:
            
            pair_deltas = np.column_stack((dx, dy))
            
        else:
            
            pair_deltas = np.empty((0, 2), dtype=np.float64)

        if self.deduplicate_displacements and pair_deltas.size:
            
            unique_deltas, inverse = np.unique(pair_deltas,
                                               axis=0,
                                               return_inverse=True,)
            
            self._pair_deltas = unique_deltas

            if unique_deltas.shape[0] <= np.iinfo(np.int32).max:
                
                inverse = inverse.astype(np.int32, copy=False)
                
            self._upper_to_pair = inverse
            
        else:
            
            self._pair_deltas = pair_deltas
            self._upper_to_pair = None

        self._evaluation_deltas = np.vstack((np.zeros((1, 2), dtype=np.float64), self._pair_deltas,))

    @staticmethod
    def _scaled_erf_fast(a: float, b: np.ndarray) -> np.ndarray:

        gaussian = np.exp(-(b * b))
        factor = np.exp(-(a * a) + 2j * a * b)

        if a >= 0:
            
            return gaussian - factor * wofz(b + 1j * a)

        return -gaussian + factor * wofz(-b - 1j * a)

    @staticmethod
    def _scaled_erf_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:

        a, b = np.broadcast_arrays(np.asarray(a, dtype=np.float64),
                                   np.asarray(b, dtype=np.float64),)
        out = np.empty(a.shape, dtype=np.complex128)
        positive = a >= 0

        if np.any(positive):
            ap, bp = a[positive], b[positive]
            out[positive] = (np.exp(-(bp * bp))
                             - np.exp(-(ap * ap) + 2j * ap * bp) * wofz(bp + 1j * ap))

        if np.any(~positive):
            an, bn = a[~positive], b[~positive]
            out[~positive] = (-np.exp(-(bn * bn))
                              + np.exp(-(an * an) + 2j * an * bn) * wofz(-bn - 1j * an))

        return out

    def _axis_factor_fast(self,
                          delta: np.ndarray,
                          L: float,
                          displacement: float,) -> np.ndarray:
        
        b = delta / (np.sqrt(2.0) * self.w)
        scale = np.sqrt(2.0) * np.pi * self.w

        upper = scale * (L + displacement)
        lower = scale * (-L + displacement)

        return (self._scaled_erf_fast(upper, b) - self._scaled_erf_fast(lower, b))

    def _evaluate_pairs_fast(self,
                             Lx: float,
                             Ly: float,
                             lx: float,
                             ly: float,) -> np.ndarray:
        
        pair_count = self._evaluation_deltas.shape[0]
        values = np.empty(pair_count, dtype=np.complex128)

        for start in range(0, pair_count, self.chunk_size):
            
            stop = min(start + self.chunk_size, pair_count)
            deltas = self._evaluation_deltas[start:stop]

            values[start:stop] = (self._axis_factor_fast(deltas[:, 0], Lx, lx) * self._axis_factor_fast(deltas[:, 1], Ly, ly))

        return values

    def _make_mpmath_task(self,
                          start: int,
                          stop: int,
                          Lx: float,
                          Ly: float,
                          lx: float,
                          ly: float,):
        
        deltas = self._evaluation_deltas[start:stop]

        return (deltas[:, 0],
                deltas[:, 1],
                repr(self.w),
                repr(float(Lx)),
                repr(float(Ly)),
                repr(float(lx)),
                repr(float(ly)),
                self.mp_dps,)

    def _evaluate_pairs_mpmath(self,
                               Lx: float,
                               Ly: float,
                               lx: float,
                               ly: float,
                               pool: Union[ProcessPoolExecutor, None],) -> np.ndarray:
        
        pair_count = self._evaluation_deltas.shape[0]

        if pool is None or pair_count <= self.chunk_size:
            
            task = self._make_mpmath_task(0, pair_count, Lx, Ly, lx, ly,)
            
            return _pw_chunk_worker(task)

        tasks = (self._make_mpmath_task(start, min(start + self.chunk_size, pair_count), Lx, Ly, lx, ly,)
                 for start in range(0, pair_count, self.chunk_size))
        
        parts = list(pool.map(_pw_chunk_worker, tasks))

        return np.concatenate(parts)

    def _build_apw_table(self,
                         Lx: float,
                         Ly: float,
                         lx: float,
                         ly: float,) -> Tuple[complex, np.ndarray]:

        x_left, x_right = self.x[:, None], self.x[None, :]
        y_left, y_right = self.y[:, None], self.y[None, :]
        shift_scale = 1.0 / (self.w * self.w * self.alpha * self.gamma)
        zeta_scale = 1.0 / (self.w**4 * self.alpha**2 * self.gamma)

        def axis_factor(left, right, L, displacement):
            
            epsilon_axis = (shift_scale * left + np.conj(shift_scale) * right)
            zeta_axis = (zeta_scale * left * left + np.conj(zeta_scale) * right * right)
            base = (-(left * left + right * right) / (self.w * self.w) + left * left / (self.w**4 * self.alpha) + right * right / (self.w**4 * np.conj(self.alpha)))

            erf_scale = np.pi * np.sqrt(self.epsilon)
            centre = epsilon_axis / self.epsilon
            b = erf_scale * np.imag(centre)
            lower = erf_scale * (displacement - L - np.real(centre))
            upper = erf_scale * (displacement + L - np.real(centre))
            erf_difference = (self._scaled_erf_array(upper, b) - self._scaled_erf_array(lower, b))
            exponent = (base - np.pi**2 * zeta_axis + np.pi**2 * epsilon_axis**2 / self.epsilon + b * b)

            return np.exp(exponent) * erf_difference

        apw = (axis_factor(x_left, x_right, Lx, lx)
               * axis_factor(y_left, y_right, Ly, ly))
        apw = 0.5 * (apw + apw.conj().T)
        pw0 = complex(np.trace(apw).real / self.num_beamlets)
        np.fill_diagonal(apw, 0.0)

        return pw0, apw

    def _build_pw_table(self,
                        Lx: float,
                        Ly: float,
                        lx: float,
                        ly: float,
                        pool: Union[ProcessPoolExecutor, None],) -> Tuple[complex, np.ndarray]:

        if self.off_fourier_plane:

            return self._build_apw_table(Lx, Ly, lx, ly,)
        
        if self.mp_dps is None:
            
            evaluated = self._evaluate_pairs_fast(Lx, Ly, lx, ly,)
            
        else:
            
            evaluated = self._evaluate_pairs_mpmath(Lx, Ly, lx, ly, pool,)

        pw0 = complex(evaluated[0])
        pair_values = evaluated[1:]

        if self._upper_to_pair is None:
            
            upper_values = pair_values
            
        else:
            
            upper_values = pair_values[self._upper_to_pair]

        pw = np.zeros((self.num_beamlets, self.num_beamlets), dtype=np.complex128,)
        
        pw[self._upper_row, self._upper_col] = upper_values
        pw[self._upper_col, self._upper_row] = np.conj(upper_values)

        return pw0, pw

    def _build_cpu_tables(self, pool: Union[ProcessPoolExecutor, None],):
        
        for Lx, Ly, lx, ly in zip(self.Lx_vec, self.Ly_vec, self.lx_vec, self.ly_vec,):
            
            pw0, pw_table = self._build_pw_table(float(Lx), float(Ly), float(lx), float(ly), pool,)
            
            self.pw0_list.append(pw0)
            self.pw_table_list.append(pw_table)
            self.pw_table += pw_table

    @staticmethod
    def _format_output(value, return_numpy: bool):
        
        return cp.asnumpy(value) if return_numpy else value

    def get_grad_cp(self,
                    phases,
                    *,
                    return_numpy: bool = True,):
        
        phi = self._as_phase_vector(phases)
        z = cp.exp(1j * phi)
        weighted = self.pw_table_cp @ cp.conj(z)

        grad = self.grad_prefix_cp * cp.imag(z * weighted)
        
        return self._format_output(grad, return_numpy)

    def get_spatial_analytic_at_z(self,
                                  xx,
                                  yy,
                                  phases,
                                  z: Union[Number, None] = None,
                                  *,
                                  max_batch_elements: int = 2_000_000,
                                  return_numpy: bool = True,):

        if not self.off_fourier_plane:
            raise RuntimeError("Construct GradCal with 'f', 'z', and 'K' to evaluate the field at z.")

        z = self.z if z is None else float(z)

        if not np.isfinite(z):
            raise ValueError("'z' must be finite.")
        if max_batch_elements < 1:
            raise ValueError("'max_batch_elements' must be at least 1.")

        xx_cp = cp.asarray(xx, dtype=cp.float64)
        yy_cp = cp.asarray(yy, dtype=cp.float64)
        phi = self._as_phase_vector(phases)

        if xx_cp.shape != yy_cp.shape:
            raise ValueError("'xx' and 'yy' must have the same shape.")

        wavelength = 2.0 * np.pi / self.K
        beta = np.pi**2 / self.alpha + 1j * np.pi * wavelength * z

        if not np.isfinite(beta) or beta == 0:
            raise ValueError("The supplied 'z' produces an invalid propagation factor.")

        output_shape = xx_cp.shape
        xx_flat = xx_cp.ravel()
        yy_flat = yy_cp.ravel()
        output = cp.empty(xx_flat.size, dtype=cp.complex128)

        radial = self.x_cp * self.x_cp + self.y_cp * self.y_cp
        channel_exponent = (1j * phi - radial / (self.w * self.w) + radial / (self.w**4 * self.alpha))
        x_centres = self.x_cp / (self.w * self.w * self.alpha)
        y_centres = self.y_cp / (self.w * self.w * self.alpha)
        profile_scale = cp.pi**2 / beta
        field_scale = (self.A * (cp.pi / self.alpha) * (cp.pi / beta)
                       * cp.exp(1j * self.K * z))
        points_per_batch = max(1, int(max_batch_elements) // self.num_beamlets,)

        for start in range(0, xx_flat.size, points_per_batch):
            
            stop = min(start + points_per_batch, xx_flat.size,)
            dx = xx_flat[start:stop][None, :] - x_centres[:, None]
            dy = yy_flat[start:stop][None, :] - y_centres[:, None]
            exponent = (channel_exponent[:, None] - profile_scale * (dx * dx + dy * dy))
            output[start:stop] = field_scale * cp.sum(cp.exp(exponent), axis=0,)

        result = output.reshape(output_shape)

        return self._format_output(result, return_numpy)

    def get_intensity_at_z(self,
                           xx,
                           yy,
                           phases,
                           z: Union[Number, None] = None,
                           *,
                           max_batch_elements: int = 2_000_000,
                           return_numpy: bool = True,):

        field = self.get_spatial_analytic_at_z(xx,
                                               yy,
                                               phases,
                                               z,
                                               max_batch_elements=max_batch_elements,
                                               return_numpy=False,)
        result = cp.real(field * cp.conj(field))

        return self._format_output(result, return_numpy)

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 500
    import matplotlib.patches as patches
    
    cmap = plt.cm.twilight
    
    np.random.seed(0) # change seed for different phases, may not produce the exact same results shown in paper due to how random seed works, nothing I can do :(
    
    intensity_size = 2000
    intensity_distance = 0.15
    dl = (2 * intensity_distance) / (intensity_size - 1)
    xx, yy = cp.meshgrid(cp.linspace(-intensity_distance, intensity_distance, intensity_size), 
                         cp.linspace(-intensity_distance, intensity_distance, intensity_size))
    
    R = 6
    d = 0.029757 # cm, value taken from the SLM setup
    beamlet_waist = 0.01365 # cm, value taken from the SLM setup
    
    f = 50.0 # cm, value taken from the SLM setup
    z = 40.0 # cm, value taken from the SLM setup
    wl = 632.8e-7 # cm, value taken from the SLM setup
    K = 2 * np.pi / wl
    
    ### beam combining ###
    L1  = 1 / (4 * R * d) * f * wl
    lr1 = 2e-1
    ######################
    
    ### beam steering ###
    # you can change lx_offset and ly_offset here to move the beam around (but do not do crazy numbers), have fun!
    L2  = 1 / (4 * R * d) * f * wl
    lr2 = 4e-1
    lx_offset = 0.03
    ly_offset = 0.03
    #####################
    
    ### beam shaping ###
    L3 = 1 / (640 * R * d) * f * wl
    lr3 = 2e3
    distance = 0.75e-2
    angle = np.linspace(0, 2*np.pi, 24)[:-1]
    cosine, sine = np.cos(angle), np.sin(angle)
    ####################
    
    x, y = [], []
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            if abs(i + j) <= R:
                x.append( (i  + 0.5 * j) * d )
                y.append( (np.sin(np.pi/3) * j) * d)

    app1 = GradCal(x=x,
                   y=y,
                   beamlet_waist = beamlet_waist,
                   beamlet_amplitude = 1,
                   PIB_windows_size_x = L1, 
                   PIB_windows_size_y = L1, 
                   PIB_windows_displace_x = 0, 
                   PIB_windows_displace_y = 0,
                   n_workers=None,
                   chunk_size=1024,
                   f = f,
                   z = z,
                   K = (2*np.pi)/wl)
    
    app2 = GradCal(x=x,
                   y=y,
                   beamlet_waist = beamlet_waist,
                   beamlet_amplitude = 1,
                   PIB_windows_size_x = L2, 
                   PIB_windows_size_y = L2, 
                   PIB_windows_displace_x = lx_offset, 
                   PIB_windows_displace_y = ly_offset,
                   n_workers=None,
                   chunk_size=1024,
                   f = f,
                   z = z,
                   K = (2*np.pi)/wl)
    
    app3 = GradCal(x=x,
                   y=y,
                   beamlet_waist = beamlet_waist,
                   beamlet_amplitude = 1,
                   PIB_windows_size_x = [L3] * cosine.shape[0], 
                   PIB_windows_size_y = [L3] * cosine.shape[0], 
                   PIB_windows_displace_x = distance * cosine, 
                   PIB_windows_displace_y = distance * sine,
                   n_workers=None,
                   chunk_size=1024,
                   f = f,
                   z = z,
                   K = (2*np.pi)/wl)
    
    phases1 = np.random.uniform(low = -np.pi, high = np.pi, size = len(x))
    phases2 = phases1.copy()
    phases3 = phases1.copy()
    
    for _ in range(20):
        
        phases1 += lr1 * app1.get_grad_cp(phases = phases1)
        
    for _ in range(20):
        
        phases2 += lr2 * app2.get_grad_cp(phases = phases2)
        
    for _ in range(20):
        
        phases3 += lr3 * app3.get_grad_cp(phases = phases3)
        
    fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize = (9, 3), constrained_layout = True)
    
    ax1.imshow(app1.get_intensity_at_z(xx = xx, yy = yy, phases = phases1, z = z), cmap = "jet", origin = "lower", vmin = 0)
    ax2.imshow(app2.get_intensity_at_z(xx = xx, yy = yy, phases = phases2, z = z), cmap = "jet", origin = "lower", vmin = 0)
    ax3.imshow(app3.get_intensity_at_z(xx = xx, yy = yy, phases = phases3, z = z), cmap = "jet", origin = "lower", vmin = 0)
    
    for _ax, _phi in zip([ax1, ax2, ax3], [phases1, phases2, phases3]):
        
        _ax_0 = _ax.inset_axes([0.585, 0.015, 0.4, 0.4])
        _ax_0.set_xticks([])
        _ax_0.set_yticks([])
        _ax_0.set_xlim([-4, 4])
        _ax_0.set_ylim([-4, 4])
        _ax_0.set_aspect(1)
        
        _arr_phi = np.mod(_phi, 2*np.pi)/(2*np.pi)
        _arr_phi = cmap(_arr_phi)
        
        for idx, (_x, _y) in enumerate(zip(x, y)):
            
            c = patches.Circle((_x * 20, _y * 20), 0.2, fill=True, linewidth=2, fc = _arr_phi[idx])
            _ax_0.add_patch(c)
            
        _ax.set_xticks([0, 999, 1999])
        _ax.set_xticklabels([])
        _ax.set_yticks([0, 999, 1999])
        _ax.set_yticklabels([])
        _ax.set_aspect(1)
        