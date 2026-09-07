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
                 deduplicate_displacements: bool = True,):
        
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

        self.obj_prefix = (self.A * self.A * self.w * self.w * np.pi) / 8.0
        self.grad_prefix = -1

        (self.Lx_vec,
         self.Ly_vec,
         self.lx_vec,
         self.ly_vec,) = self._normalize_pib_parameters(PIB_windows_size_x,
                                                        PIB_windows_size_y,
                                                        PIB_windows_displace_x,
                                                        PIB_windows_displace_y,)
                                                        
        self.n_pib = int(self.Lx_vec.size)
        self._prepare_displacement_pairs()

        self.pw0_list: List[complex] = []
        self.pw_table_list: List[np.ndarray] = []
        self.pw_table = np.zeros((self.num_beamlets, self.num_beamlets), dtype=np.complex128,)

        use_pool = ( self.mp_dps is not None and self.n_workers > 1 and self._evaluation_deltas.shape[0] > self.chunk_size)

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

        self.anchor_pos = int(np.argmin(self.x * self.x + self.y * self.y))
        self.W_cp = self.pw_table_cp

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

    def _build_pw_table(self,
                        Lx: float,
                        Ly: float,
                        lx: float,
                        ly: float,
                        pool: Union[ProcessPoolExecutor, None],) -> Tuple[complex, np.ndarray]:
        
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

    def get_PIB_analytic(self,
                         phases,
                         pib_index: int = 0,
                         *,
                         return_numpy: bool = True,):
        
        if not 0 <= pib_index < self.n_pib:
            
            raise IndexError(f"'pib_index' must be between 0 and {self.n_pib - 1}.")

        phi = self._as_phase_vector(phases)
        z = cp.exp(1j * phi)

        pw_table = self.pw_table_cp_list[pib_index]
        weighted = pw_table @ cp.conj(z)
        s_offdiag = z @ weighted
        total = (s_offdiag + self.num_beamlets * self.pw0_cp_list[pib_index])

        result = cp.real(self.obj_prefix_cp * total)
        
        return self._format_output(result, return_numpy)

    def get_grad_cp(self,
                    phases,
                    *,
                    return_numpy: bool = True,):
        
        phi = self._as_phase_vector(phases)
        z = cp.exp(1j * phi)
        weighted = self.pw_table_cp @ cp.conj(z)

        grad = self.grad_prefix_cp * cp.imag(z * weighted)
        
        return self._format_output(grad, return_numpy)

    def get_intensity(self,
                      xx,
                      yy,
                      phases,
                      *,
                      max_batch_elements: int = 2_000_000,
                      return_numpy: bool = True,):

        xx_cp = cp.asarray(xx, dtype=cp.float64)
        yy_cp = cp.asarray(yy, dtype=cp.float64)
        phi = self._as_phase_vector(phases)

        if xx_cp.shape != yy_cp.shape:
            
            raise ValueError("'xx' and 'yy' must have the same shape.")
            
        if max_batch_elements < 1:
            
            raise ValueError("'max_batch_elements' must be at least 1.")

        output_shape = xx_cp.shape
        xx_flat = xx_cp.ravel()
        yy_flat = yy_cp.ravel()
        output = cp.empty(xx_flat.size, dtype=cp.float64)

        points_per_batch = max(1, int(max_batch_elements) // self.num_beamlets,)
        phase_weights = cp.exp(1j * phi)
        spatial_scale = 2.0 * cp.pi
        envelope_scale = self.A * cp.pi * self.w * self.w
        gaussian_scale = cp.pi * cp.pi * self.w * self.w

        for start in range(0, xx_flat.size, points_per_batch):
            
            stop = min(start + points_per_batch, xx_flat.size,)
            x_batch = xx_flat[start:stop]
            y_batch = yy_flat[start:stop]

            spatial_phase = spatial_scale * ( self.x_cp[:, None] * x_batch[None, :] + self.y_cp[:, None] * y_batch[None, :])
            steering = cp.exp(1j * spatial_phase)
            field = phase_weights @ steering

            envelope = envelope_scale * cp.exp(-gaussian_scale * (x_batch * x_batch + y_batch * y_batch))
            output[start:stop] = cp.square( cp.abs(envelope * field))

        result = output.reshape(output_shape)
        
        return self._format_output(result, return_numpy)

    def get_intensity_near_field(self,
                                 xx,
                                 yy,
                                 phases,
                                 *,
                                 max_batch_elements: int = 2_000_000,
                                 return_numpy: bool = True,):

        xx_cp = cp.asarray(xx, dtype=cp.float64)
        yy_cp = cp.asarray(yy, dtype=cp.float64)
        phi = self._as_phase_vector(phases)

        if xx_cp.shape != yy_cp.shape:
            
            raise ValueError("'xx' and 'yy' must have the same shape.")
            
        if max_batch_elements < 1:
            
            raise ValueError("'max_batch_elements' must be at least 1.")

        output_shape = xx_cp.shape
        xx_flat = xx_cp.ravel()
        yy_flat = yy_cp.ravel()
        output = cp.empty(xx_flat.size, dtype=cp.float64)

        points_per_batch = max(1, int(max_batch_elements) // self.num_beamlets,)
        phase_weights = cp.exp(1j * phi)

        for start in range(0, xx_flat.size, points_per_batch):
            
            stop = min(start + points_per_batch, xx_flat.size,)
            x_batch = xx_flat[start:stop]
            y_batch = yy_flat[start:stop]

            dx = x_batch[None, :] - self.x_cp[:, None]
            dy = y_batch[None, :] - self.y_cp[:, None]
            profiles = cp.exp(-(dx * dx + dy * dy) / self.w ** 2)

            field = self.A * cp.sum(phase_weights[:, None] * profiles, axis=0,)
            output[start:stop] = cp.square(cp.abs(field))

        result = output.reshape(output_shape)
        
        return self._format_output(result, return_numpy)
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 500
    
    intensity_size = 2000
    intensity_distance = 1.5
    dl = (2 * intensity_distance) / (intensity_size - 1)
    xx, yy = cp.meshgrid(cp.linspace(-intensity_distance, intensity_distance, intensity_size), 
                         cp.linspace(-intensity_distance, intensity_distance, intensity_size))

    d = 1
    
    max_pibs = []
    data_mbga = []
    data_spgd = []

    for R, lr_mbga, lr_spgd, step_spgd in zip([1, 2, 3, 4, 5, 6],
                                              np.linspace(1, 2.5, 6),
                                              [1e3, 0.5e3, 0.25e3, 0.125e3, 0.0625e3, 0.03125e3],
                                              [75, 150, 300, 600, 1200, 2400]):
        
        print(R)

        x, y = [], []
        for i in range(-R, R+1):
            for j in range(-R, R+1):
                if abs(i + j) <= R:
                    x.append( (i  + 0.5 * j) * d )
                    y.append( (np.sin(np.pi/3) * j) * d)
                    
        datum_mbga = []
        datum_spgd = []
        
        L = 1/(4 * d * R) # Eq. 12
        pib_l = L / dl
        
        app = GradCal(x=x,
                      y=y,
                      beamlet_waist = 0.2,
                      beamlet_amplitude = 1,
                      PIB_windows_size_x = L, 
                      PIB_windows_size_y = L, 
                      PIB_windows_displace_x = 0, 
                      PIB_windows_displace_y = 0,
                      n_workers=None,
                      chunk_size=1024)
        
        max_pibs.append(app.get_intensity(xx = xx, yy = yy, phases = np.zeros(shape = (len(x))))[int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1),
                                                                                                 int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1)].sum() * dl ** 2)
        
        for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        
            np.random.seed(seed) # random generator could vary from environments, do not be surperise if you cannot get the exact same results from the paper
            phases1 = np.random.uniform(low = -np.pi, high = np.pi, size = len(x))
            phases2 = phases1.copy()
            
            ints_mbga = []
            ints_spgd = []
            
            ints_mbga.append(app.get_intensity(xx = xx, yy = yy, phases = phases1)[int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1),
                                                                                   int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1)].sum() * dl ** 2)
            
            ints_spgd.append(app.get_intensity(xx = xx, yy = yy, phases = phases2)[int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1),
                                                                                   int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1)].sum() * dl ** 2)
        
            
            for i in range(20):
                
                phases1 += lr_mbga * app.get_grad_cp(phases = phases1)
                ints_mbga.append(app.get_intensity(xx = xx, yy = yy, phases = phases1)[int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1),
                                                                                       int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1)].sum() * dl ** 2)
            
            datum_mbga.append(ints_mbga)
            
            for i in range(step_spgd):
        
                delta_phases = np.random.choice(a = [-0.1, 0.1], size = len(x))
                j1 = app.get_intensity(xx = xx, yy = yy, phases = phases2 - delta_phases)[int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1),
                                                                                          int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1)].sum() * dl ** 2
                j2 = app.get_intensity(xx = xx, yy = yy, phases = phases2 + delta_phases)[int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1),
                                                                                          int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1)].sum() * dl ** 2
                
                phases2 += lr_spgd * (j2 - j1) * (delta_phases / 2)
                
                ints_spgd.append(app.get_intensity(xx = xx, yy = yy, phases = phases2)[int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1),
                                                                                       int((intensity_size-1)/2 - pib_l):int((intensity_size-1)/2 + pib_l+1)].sum() * dl ** 2)
            
            datum_spgd.append(ints_spgd)
            
        data_mbga.append(datum_mbga)
        data_spgd.append(datum_spgd)

    threshold = 0.90
    
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize = (6, 6), constrained_layout = True)
    
    mbga_curves = np.mean(data_mbga, axis = 1)
    mbga_curves /= np.max(max_pibs)
    
    spgd_curves = [np.mean(datum_spgd, axis = 0)/np.max(max_pibs) for datum_spgd in data_spgd]
    
    vmax = np.asarray(max_pibs)/np.max(max_pibs)
    
    for i, _c in zip(range(6), ["r", "g", "b", "c", "m", "y"]):
        
        ax1.plot(np.linspace(1, mbga_curves.shape[1], mbga_curves.shape[1]), mbga_curves[i], ls = "-",  c = _c, alpha = 0.75, lw = 1, label = "Ours, " + r"$R={}$".format(i))
        ax1.plot(np.linspace(1, spgd_curves[i].shape[0], spgd_curves[i].shape[0]), spgd_curves[i], ls = "--", c = _c, alpha = 0.75, lw = 1, label = "Ours, " + r"$R={}$".format(i))
        
        ax1.plot([0, 2700], [vmax[i], vmax[i]], ls = ":", alpha = 0.5, c = _c, lw = 1, label = "Max., " + r"$R={}$".format(i))
        
        if i == 0:
        
            ax2.scatter(i + 1, np.where(mbga_curves[i] >= vmax[i] * threshold)[0][0], label = "Ours", s = 3, c = "r", alpha = 1)
            ax2.scatter(i + 1, np.where(spgd_curves[i] >= vmax[i] * threshold)[0][0], label = "SPGD", s = 3, c = "b", alpha = 1)
            
        else:
            
            ax2.scatter(i + 1, np.where(mbga_curves[i] >= vmax[i] * threshold)[0][0], s = 3, c = "r", alpha = 1)
            ax2.scatter(i + 1, np.where(spgd_curves[i] >= vmax[i] * threshold)[0][0], s = 3, c = "b", alpha = 1)
        
    ax1.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1., .102), fancybox=False, 
                shadow=False, ncol=6, mode="expand", borderaxespad=0., columnspacing=0.05, handletextpad=0.1, handlelength=2)
    
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(0.9e-0, 2.7e3)
    ax1.set_ylim(0.8e-2, 1.2e0)
    
    ax1.set_xlabel("Optimisation step " + r"$t$")
    ax1.set_ylabel("Normalised PIB")
    
    ax2.set_yscale("log")
    ax2.set_xlim(0, 7)
    ax2.set_ylim(1e0, 1e4)
    ax2.set_xticks([1, 2, 3, 4, 5, 6])
    ax2.set_xticklabels(["1", "2", "3", "4", "5", "6"])
    ax2.set_aspect(1.8)
    
    ax2.set_xlabel(r"Array size $R$")
    ax2.set_ylabel("Avg. step " + r"$t$" + " at 90% Max.")