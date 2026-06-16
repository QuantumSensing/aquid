//! Momentum-cutoff projector for the SPGPE.
//!
//! Enforces the classical-field energy cutoff by masking Fourier modes
//! with \(|\mathbf{k}| > k_{\mathrm{cut}}\) after each timestep.
//! Required for post-thermalisation dynamics (bias sweeps, rotation);
//! optional during pure thermalisation (thesis §3.4.4–3.4.5).

use ndarray::{Array1, Array2};
use num_complex::Complex;
use rustfft::FftPlanner;

/// Precomputed Fourier-space mask for the classical-field projector.
///
/// Stores a 2D mask of shape \((n_x, n_y)\) where `mask[[i, j]]` is
/// \(1\) if \(|\mathbf{k}| \le k_{\mathrm{cut}}\) and \(0\) otherwise.
/// The mask is applied by forward FFT, elementwise multiplication,
/// then inverse FFT with normalisation \(1/(n_x n_y)\).
pub struct Projector {
    mask: Array2<f64>,
}

impl Projector {
    /// Build a projector for the given temperature and chemical potential.
    ///
    /// The energy cutoff and corresponding wavenumber are
    /// \[
    /// \varepsilon_{\mathrm{cut}} = \tilde{T}\ln 2 + \tilde{\mu},
    /// \qquad
    /// k_{\mathrm{cut}} = \sqrt{2\,\varepsilon_{\mathrm{cut}}}.
    /// \]
    ///
    /// All inputs are dimensionless (thesis Eqs. 3.58, 3.65).
    pub fn new(
        temperature: f64,
        chemical_potential: f64,
        kx: &Array1<f64>,
        ky: &Array1<f64>,
    ) -> Self {
        let eps_cut = temperature * std::f64::consts::LN_2 + chemical_potential;
        let k_cut_sq = 2.0 * eps_cut;

        let nx = kx.len();
        let ny = ky.len();

        let kx_sq = kx.mapv(|k| k * k).into_shape((nx, 1)).expect("kx reshape");
        let ky_sq = ky.mapv(|k| k * k).into_shape((1, ny)).expect("ky reshape");
        let k_sq = kx_sq + ky_sq;

        let mask = k_sq.mapv(|k2| if k2 <= k_cut_sq { 1.0_f64 } else { 0.0_f64 });

        Self { mask }
    }

    /// Apply the projector to a wavefunction.
    ///
    /// Forward FFT → multiply by mask → inverse FFT → normalise.
    /// This removes all Fourier components with
    /// \(|\mathbf{k}| > k_{\mathrm{cut}}\).
    pub fn apply(&self, phi: &Array2<Complex<f64>>) -> Array2<Complex<f64>> {
        let (nx, ny) = (phi.shape()[0], phi.shape()[1]);

        let mut planner = FftPlanner::new();
        let fft_axis1 = planner.plan_fft_forward(ny);
        let fft_axis0 = planner.plan_fft_forward(nx);
        let ifft_axis1 = planner.plan_fft_inverse(ny);
        let ifft_axis0 = planner.plan_fft_inverse(nx);

        let mut spectrum = phi.clone();

        // Forward FFT axis 1 (columns).
        for mut row in spectrum.rows_mut() {
            if let Some(row_slice) = row.as_slice_mut() {
                fft_axis1.process(row_slice);
            }
        }

        // Forward FFT axis 0 (rows).
        let mut column = vec![Complex::new(0.0, 0.0); nx];
        for col_idx in 0..ny {
            for (row_idx, value) in column.iter_mut().enumerate() {
                *value = spectrum[[row_idx, col_idx]];
            }
            fft_axis0.process(&mut column);
            for (row_idx, value) in column.iter().enumerate() {
                spectrum[[row_idx, col_idx]] = *value;
            }
        }

        // Apply mask.
        for ((i, j), value) in spectrum.indexed_iter_mut() {
            *value *= self.mask[[i, j]];
        }

        // Inverse FFT axis 0.
        for col_idx in 0..ny {
            for (row_idx, value) in column.iter_mut().enumerate() {
                *value = spectrum[[row_idx, col_idx]];
            }
            ifft_axis0.process(&mut column);
            for (row_idx, value) in column.iter().enumerate() {
                spectrum[[row_idx, col_idx]] = *value;
            }
        }

        // Inverse FFT axis 1.
        for mut row in spectrum.rows_mut() {
            if let Some(row_slice) = row.as_slice_mut() {
                ifft_axis1.process(row_slice);
            }
        }

        let norm_factor = 1.0 / ((nx * ny) as f64);
        spectrum.mapv_inplace(|v| v * norm_factor);

        spectrum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::Distribution;

    fn make_k_space(nx: usize, ny: usize, dx: f64, dy: f64) -> (Array1<f64>, Array1<f64>) {
        let kx = Array1::from_shape_fn(nx, |i| {
            let f = i as f64 / (nx as f64 * dx);
            if i > nx / 2 {
                (f - 1.0 / dx) * 2.0 * std::f64::consts::PI
            } else {
                f * 2.0 * std::f64::consts::PI
            }
        });
        let ky = Array1::from_shape_fn(ny, |i| {
            let f = i as f64 / (ny as f64 * dy);
            if i > ny / 2 {
                (f - 1.0 / dy) * 2.0 * std::f64::consts::PI
            } else {
                f * 2.0 * std::f64::consts::PI
            }
        });
        (kx, ky)
    }

    #[test]
    fn projector_mask_shape() {
        let (kx, ky) = make_k_space(32, 64, 1.0, 1.0);
        let p = Projector::new(1.0, 1.0, &kx, &ky);
        assert_eq!(p.mask.shape(), &[32, 64]);
    }

    #[test]
    fn projector_mask_is_binary() {
        let (kx, ky) = make_k_space(32, 32, 1.0, 1.0);
        let p = Projector::new(1.0, 1.0, &kx, &ky);
        for &v in p.mask.iter() {
            assert!(v == 0.0 || v == 1.0, "mask value {} not binary", v);
        }
    }

    #[test]
    fn projector_k0_is_always_retained() {
        let (kx, ky) = make_k_space(32, 32, 1.0, 1.0);
        // Even with very low cutoff, k=0 must stay.
        let eps_cut = 0.01_f64;
        let p = Projector::new(eps_cut, 0.0, &kx, &ky);
        // The (0,0) mode is the first index in standard FFT ordering.
        assert!(p.mask[[0, 0]] == 1.0, "k=0 mode must always be retained");
    }

    #[test]
    fn projector_preserves_low_k_plane_wave() {
        let nx = 32;
        let ny = 32;
        let (kx, ky) = make_k_space(nx, ny, 1.0, 1.0);

        // Use a large cutoff so the plane wave is well inside.
        let p = Projector::new(100.0, 0.0, &kx, &ky);

        // Plane wave at k = (kx[1], ky[0]) — low spatial frequency.
        let psi = Array2::from_shape_fn((nx, ny), |(i, j)| {
            let phase = kx[1] * i as f64 + ky[0] * j as f64;
            Complex::new(phase.cos(), phase.sin())
        });

        let projected = p.apply(&psi);

        // The projected field should closely match the input.
        let max_diff = (&projected - &psi)
            .iter()
            .map(|c| c.norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 1e-10,
            "low-k plane wave should be preserved, max_diff = {:.2e}",
            max_diff
        );
    }

    #[test]
    fn projector_removes_high_k_plane_wave() {
        let nx = 32;
        let ny = 32;
        let (kx, ky) = make_k_space(nx, ny, 1.0, 1.0);

        // Cutoff below the Nyquist frequency.
        let p = Projector::new(1.0, 1.0, &kx, &ky);

        // Plane wave at Nyquist frequency.
        let k_nyq_x = kx[nx / 2];
        let k_nyq_y = ky[ny / 2];
        let psi = Array2::from_shape_fn((nx, ny), |(i, j)| {
            let phase = k_nyq_x * i as f64 + k_nyq_y * j as f64;
            Complex::new(phase.cos(), phase.sin())
        });

        let projected = p.apply(&psi);
        let norm = projected.iter().map(|c| c.norm_sqr()).sum::<f64>();

        // After projection, high-k modes are removed — the field norm
        // should be substantially reduced.
        let original_norm = psi.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert!(
            norm < 1e-10 * original_norm,
            "high-k plane wave should be removed, norm = {:.2e}",
            norm
        );
    }

    #[test]
    fn projector_roundtrip_idempotent() {
        let nx = 32;
        let ny = 32;
        let (kx, ky) = make_k_space(nx, ny, 1.0, 1.0);
        let p = Projector::new(1.0, 1.0, &kx, &ky);

        // Random field.
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        let psi = Array2::from_shape_fn((nx, ny), |_| {
            Complex::new(normal.sample(&mut rng), normal.sample(&mut rng))
        });

        let once = p.apply(&psi);
        let twice = p.apply(&once);

        let diff = (&twice - &once)
            .iter()
            .map(|c| c.norm())
            .fold(0.0_f64, f64::max);
        assert!(
            diff < 1e-12,
            "projector should be idempotent, max_diff = {:.2e}",
            diff
        );
    }

    #[test]
    fn projector_preserves_norm_of_low_k_state() {
        let nx = 32;
        let ny = 32;
        let (kx, ky) = make_k_space(nx, ny, 1.0, 1.0);

        // Construct a state with only low-k modes populated.
        let k_cut_sq = 2.0 * (1.0_f64 * std::f64::consts::LN_2 + 1.0);
        let mut psi_k = Array2::from_elem((nx, ny), Complex::new(0.0, 0.0));
        let mut rng = rand::thread_rng();
        let normal = rand_distr::StandardNormal;
        for i in 0..nx {
            for j in 0..ny {
                let k2 = kx[i] * kx[i] + ky[j] * ky[j];
                if k2 <= k_cut_sq {
                    psi_k[[i, j]] = Complex::new(normal.sample(&mut rng), normal.sample(&mut rng));
                }
            }
        }

        // Transform to real space.
        let p = Projector::new(1.0, 1.0, &kx, &ky);
        // Build real-space field via inverse FFT of the band-limited k-space field.
        let mut planner = FftPlanner::new();
        let ifft0 = planner.plan_fft_inverse(nx);
        let ifft1 = planner.plan_fft_inverse(ny);
        let mut psi = psi_k.clone();
        let mut col = vec![Complex::new(0.0, 0.0); nx];
        for j in 0..ny {
            for (i, v) in col.iter_mut().enumerate() {
                *v = psi[[i, j]];
            }
            ifft0.process(&mut col);
            for (i, v) in col.iter().enumerate() {
                psi[[i, j]] = *v;
            }
        }
        for mut row in psi.rows_mut() {
            if let Some(s) = row.as_slice_mut() {
                ifft1.process(s);
            }
        }
        psi.mapv_inplace(|v| v / ((nx * ny) as f64));

        let norm_before = psi.iter().map(|c| c.norm_sqr()).sum::<f64>();
        let projected = p.apply(&psi);
        let norm_after = projected.iter().map(|c| c.norm_sqr()).sum::<f64>();

        let rel_diff = (norm_after - norm_before).abs() / norm_before;
        assert!(
            rel_diff < 1e-10,
            "norm of band-limited state should be preserved, rel_diff = {:.2e}",
            rel_diff
        );
    }
}
