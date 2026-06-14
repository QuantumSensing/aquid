//! Generates the wave vectors for the Fourier-space representation of the grid.

use crate::types::Simulation;
use ndarray::{Array1, Array2};

/// Generates the wave vectors (k-space) for the grid.
///
/// # Arguments
///
/// * `simulation` - The simulation parameters, including grid size and grid points.
///
/// # Returns
///
/// A tuple containing:
/// - `kx`: A 1D array of wave vectors in the x-direction.
/// - `ky`: A 1D array of wave vectors in the y-direction.
/// - `k_sq`: A 2D array of the squared magnitude of the wave vectors.
pub fn generate_k_space(simulation: &Simulation) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    let grid_points_x = simulation.gridpoints.0;
    let grid_points_y = simulation.gridpoints.1;
    let step_size_x = simulation.step_size.0;
    let step_size_y = simulation.step_size.1;

    // Generate wave vectors for x and y directions
    let kx = Array1::from_shape_fn(grid_points_x, |i| {
        let freq = i as f64 / (grid_points_x as f64 * step_size_x);
        if i > grid_points_x / 2 {
            freq - 1.0 / step_size_x
        } else {
            freq
        }
    }) * 2.0
        * std::f64::consts::PI;

    let ky = Array1::from_shape_fn(grid_points_y, |i| {
        let freq = i as f64 / (grid_points_y as f64 * step_size_y);
        if i > grid_points_y / 2 {
            freq - 1.0 / step_size_y
        } else {
            freq
        }
    }) * 2.0
        * std::f64::consts::PI;

    // Create k_sq grid
    let kx_sq = kx.mapv(|v| v.powi(2));
    let ky_sq = ky.mapv(|v| v.powi(2));
    let k_sq = kx_sq.into_shape((grid_points_x, 1)).unwrap() + ky_sq;

    (kx, ky, k_sq)
}

/// Check the anti-aliasing condition for the cubic nonlinearity.
///
/// Blakie et al. require \(dx \le \pi/(2 k_{\max})\) where \(k_{\max}\) is the
/// largest physically retained wavenumber (typically the projector cutoff).
/// This is twice as strict as the naive Nyquist bound \(dx \le \pi/k_{\max}\).
pub fn check_anti_aliasing(dx: f64, k_cut: f64) -> bool {
    dx <= std::f64::consts::PI / (2.0 * k_cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Simulation;
    use ndarray::Array2;
    use num_complex::Complex;
    use rustfft::FftPlanner;

    fn test_simulation() -> Simulation {
        let grid_size = 100e-6;
        let n = 64.0;
        Simulation {
            grid_size,
            gridpoints: (64, 64),
            step_size: (2.0 * grid_size / n, 2.0 * grid_size / n),
            timesteps: 10,
            timestep: 1e-3,
            runs: 1,
            noise_realisations: 1,
        }
    }

    #[test]
    fn k_space_correct_shape() {
        let sim = test_simulation();
        let (kx, ky, k_sq) = generate_k_space(&sim);
        assert_eq!(kx.len(), 64);
        assert_eq!(ky.len(), 64);
        assert_eq!(k_sq.shape(), &[64, 64]);
    }

    #[test]
    fn k_sq_is_nonnegative() {
        let sim = test_simulation();
        let (_, _, k_sq) = generate_k_space(&sim);
        assert!(k_sq.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn k_sq_zero_mode_is_zero() {
        let sim = test_simulation();
        let (_, _, k_sq) = generate_k_space(&sim);
        assert!((k_sq[[0, 0]]).abs() < 1e-12);
    }

    #[test]
    fn kx_max_equals_nyquist_frequency() {
        let sim = test_simulation();
        let (kx, ky, _) = generate_k_space(&sim);
        let dx = sim.step_size.0;
        let dy = sim.step_size.1;
        let expected_kx_max = std::f64::consts::PI / dx;
        let expected_ky_max = std::f64::consts::PI / dy;
        let kx_max_abs = kx.iter().map(|k| k.abs()).fold(0.0_f64, f64::max);
        let ky_max_abs = ky.iter().map(|k| k.abs()).fold(0.0_f64, f64::max);
        let rel_err_x = (kx_max_abs - expected_kx_max).abs() / expected_kx_max;
        let rel_err_y = (ky_max_abs - expected_ky_max).abs() / expected_ky_max;
        assert!(
            rel_err_x < 1e-10,
            "max |kx| = {:.6e}, expected pi/dx = {:.6e}",
            kx_max_abs,
            expected_kx_max
        );
        assert!(
            rel_err_y < 1e-10,
            "max |ky| = {:.6e}, expected pi/dy = {:.6e}",
            ky_max_abs,
            expected_ky_max
        );
    }

    #[test]
    fn fft_round_trip_normalisation() {
        // Forward FFT + inverse FFT must recover the original field
        // up to the 1/(nx*ny) normalisation factor applied on inverse.
        let (nx, ny) = (32, 32);
        let mut field = Array2::from_shape_fn((nx, ny), |(i, j)| {
            let x = (i as f64) - (nx as f64) / 2.0;
            let y = (j as f64) - (ny as f64) / 2.0;
            Complex::new((-0.5 * (x * x + y * y)).exp(), 0.0)
        });

        let original = field.clone();

        let mut planner = FftPlanner::new();
        let fft1 = planner.plan_fft_forward(ny);
        let fft0 = planner.plan_fft_forward(nx);
        let ifft1 = planner.plan_fft_inverse(ny);
        let ifft0 = planner.plan_fft_inverse(nx);

        // Forward
        for mut row in field.rows_mut() {
            if let Some(s) = row.as_slice_mut() {
                fft1.process(s);
            }
        }
        let mut col_buf = vec![Complex::new(0.0, 0.0); nx];
        for j in 0..ny {
            for (i, v) in col_buf.iter_mut().enumerate() {
                *v = field[[i, j]];
            }
            fft0.process(&mut col_buf);
            for (i, v) in col_buf.iter().enumerate() {
                field[[i, j]] = *v;
            }
        }

        // Inverse
        for j in 0..ny {
            for (i, v) in col_buf.iter_mut().enumerate() {
                *v = field[[i, j]];
            }
            ifft0.process(&mut col_buf);
            for (i, v) in col_buf.iter().enumerate() {
                field[[i, j]] = *v;
            }
        }
        for mut row in field.rows_mut() {
            if let Some(s) = row.as_slice_mut() {
                ifft1.process(s);
            }
        }

        let norm = 1.0 / ((nx * ny) as f64);
        field.mapv_inplace(|v| v * norm);

        let max_err = (&field - &original)
            .iter()
            .map(|c| c.norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-10,
            "FFT round-trip max error = {:.2e}, expected < 1e-10",
            max_err
        );
    }

    #[test]
    fn anti_aliasing_bound_check() {
        // With k_cut at half Nyquist, the condition should pass.
        let dx = 0.1;
        let k_nyquist = std::f64::consts::PI / dx;
        let k_cut = k_nyquist / 2.0;
        assert!(check_anti_aliasing(dx, k_cut));

        // With k_cut above Nyquist/2, the condition should fail.
        let k_cut_too_high = k_nyquist * 0.51;
        assert!(!check_anti_aliasing(dx, k_cut_too_high));
    }
}
