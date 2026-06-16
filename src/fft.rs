//! Shared 2D FFT pipeline for applying operators in Fourier space.
//!
//! The kinetic-energy term, the angular-momentum operator \(\hat{L}_z\), and the
//! classical-field projector all transform a field to Fourier space, multiply by
//! a k-space operator, transform back, and normalise by \(1/(n_x n_y)\). This
//! module holds that pipeline once so the transform code is not repeated.
//!
//! `rustfft` produces unnormalised transforms; the \(1/(n_x n_y)\) factor is
//! applied on the inverse so a forward followed by an inverse is the identity.

use ndarray::Array2;
use num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    /// Per-thread FFT planner. Planning a transform is costly relative to
    /// executing one, and these routines run in the inner timestep loop, so the
    /// planner (and the plans it caches internally) is built once per thread and
    /// reused across every call rather than rebuilt each invocation. It is
    /// thread-local because the simulation evolves noise realisations in
    /// parallel via Rayon and each worker keeps its own planner.
    static PLANNER: RefCell<FftPlanner<f64>> = RefCell::new(FftPlanner::new());
}

/// Message used when an array is not contiguous and cannot be transformed.
const CONTIGUITY_MSG: &str = "2D FFT requires a contiguous (standard-layout) array";

/// The four transforms needed for a forward and inverse 2D FFT on an
/// \((n_x, n_y)\) grid: forward and inverse along each axis.
struct Plans2d {
    forward_axis0: Arc<dyn Fft<f64>>,
    forward_axis1: Arc<dyn Fft<f64>>,
    inverse_axis0: Arc<dyn Fft<f64>>,
    inverse_axis1: Arc<dyn Fft<f64>>,
}

/// Fetch the plans for an \((n_x, n_y)\) grid from the thread-local planner.
///
/// `rustfft` caches plans by length inside the planner, so repeated requests for
/// the same lengths return cheap `Arc` clones without re-planning.
fn plans_for(nx: usize, ny: usize) -> Plans2d {
    PLANNER.with(|planner| {
        let mut planner = planner.borrow_mut();
        Plans2d {
            forward_axis0: planner.plan_fft_forward(nx),
            forward_axis1: planner.plan_fft_forward(ny),
            inverse_axis0: planner.plan_fft_inverse(nx),
            inverse_axis1: planner.plan_fft_inverse(ny),
        }
    })
}

/// Forward 2D FFT in place.
///
/// Transforms along axis 1 (rows, contiguous in standard layout) then along
/// axis 0 (columns, via a scratch buffer). No normalisation is applied; the
/// \(1/(n_x n_y)\) factor lives on the inverse in [`inverse_2d_normalised`].
pub fn forward_2d(field: &mut Array2<Complex<f64>>) {
    let (nx, ny) = (field.shape()[0], field.shape()[1]);
    let plans = plans_for(nx, ny);
    let mut column = vec![Complex::new(0.0, 0.0); nx];

    for mut row in field.rows_mut() {
        let row_slice = row.as_slice_mut().expect(CONTIGUITY_MSG);
        plans.forward_axis1.process(row_slice);
    }
    for col_idx in 0..ny {
        for (row_idx, value) in column.iter_mut().enumerate() {
            *value = field[[row_idx, col_idx]];
        }
        plans.forward_axis0.process(&mut column);
        for (row_idx, value) in column.iter().enumerate() {
            field[[row_idx, col_idx]] = *value;
        }
    }
}

/// Inverse 2D FFT in place, including the \(1/(n_x n_y)\) normalisation.
///
/// Transforms along axis 0 (columns) then axis 1 (rows), mirroring
/// [`forward_2d`], so a forward followed by an inverse recovers the input.
pub fn inverse_2d_normalised(field: &mut Array2<Complex<f64>>) {
    let (nx, ny) = (field.shape()[0], field.shape()[1]);
    let plans = plans_for(nx, ny);
    let mut column = vec![Complex::new(0.0, 0.0); nx];

    for col_idx in 0..ny {
        for (row_idx, value) in column.iter_mut().enumerate() {
            *value = field[[row_idx, col_idx]];
        }
        plans.inverse_axis0.process(&mut column);
        for (row_idx, value) in column.iter().enumerate() {
            field[[row_idx, col_idx]] = *value;
        }
    }
    for mut row in field.rows_mut() {
        let row_slice = row.as_slice_mut().expect(CONTIGUITY_MSG);
        plans.inverse_axis1.process(row_slice);
    }

    let norm_factor = 1.0 / ((nx * ny) as f64);
    field.mapv_inplace(|value| value * norm_factor);
}

/// Apply an operator in Fourier space: forward FFT, run `op` on the spectrum,
/// inverse FFT with normalisation.
///
/// `op` receives the forward-transformed field and mutates it in place — for
/// example multiplying by the kinetic operator \(k^2/2\) or by the projector
/// mask. The returned array is the operator applied to `phi` in real space.
pub fn apply_in_k_space(
    phi: &Array2<Complex<f64>>,
    op: impl FnOnce(&mut Array2<Complex<f64>>),
) -> Array2<Complex<f64>> {
    let mut field = phi.clone();
    forward_2d(&mut field);
    op(&mut field);
    inverse_2d_normalised(&mut field);
    field
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Distribution, StandardNormal};

    fn random_field(nx: usize, ny: usize) -> Array2<Complex<f64>> {
        let mut rng = rand::thread_rng();
        let normal = StandardNormal;
        Array2::from_shape_fn((nx, ny), |_| {
            Complex::new(normal.sample(&mut rng), normal.sample(&mut rng))
        })
    }

    fn max_abs_diff(a: &Array2<Complex<f64>>, b: &Array2<Complex<f64>>) -> f64 {
        (a - b).iter().map(|c| c.norm()).fold(0.0_f64, f64::max)
    }

    #[test]
    fn forward_then_inverse_is_identity() {
        let field = random_field(32, 48);
        let mut transformed = field.clone();
        forward_2d(&mut transformed);
        inverse_2d_normalised(&mut transformed);
        let err = max_abs_diff(&transformed, &field);
        assert!(
            err < 1e-12,
            "round-trip error = {:.2e}, expected < 1e-12",
            err
        );
    }

    #[test]
    fn apply_identity_operator_recovers_input() {
        let field = random_field(64, 64);
        // A no-op in k-space must leave the field unchanged once normalised.
        let result = apply_in_k_space(&field, |_| {});
        let err = max_abs_diff(&result, &field);
        assert!(
            err < 1e-12,
            "identity error = {:.2e}, expected < 1e-12",
            err
        );
    }

    #[test]
    fn apply_zeroing_operator_gives_zero() {
        let field = random_field(32, 32);
        let result = apply_in_k_space(&field, |spectrum| spectrum.fill(Complex::new(0.0, 0.0)));
        let max = result.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        assert!(max < 1e-14, "zeroed field max = {:.2e}, expected ~0", max);
    }

    #[test]
    fn handles_multiple_grid_sizes_via_cached_planner() {
        // Reusing the thread-local planner across differing sizes must not
        // panic and must stay correct for each size.
        for (nx, ny) in [(16, 16), (32, 8), (8, 64)] {
            let field = random_field(nx, ny);
            let result = apply_in_k_space(&field, |_| {});
            let err = max_abs_diff(&result, &field);
            assert!(
                err < 1e-12,
                "round-trip error = {:.2e} for ({}, {})",
                err,
                nx,
                ny
            );
        }
    }
}
