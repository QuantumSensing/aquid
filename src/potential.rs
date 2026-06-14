//! Potential energy functions for the SGPE simulation.
//!
//! This module provides implementations of the harmonic and toroidal trapping
//! potentials, together with two Gaussian weak-link barriers for atomtronic
//! SQUID simulations. The barrier potentials are time-dependent, with the
//! junction positions advancing according to the bias kinematics.
//!
//! All potentials are returned as \(\text{Array2}<\text{Complex}<\text{f64}>>\) with
//! zero imaginary part, since the potentials are real-valued.

use crate::constants::PI;
use crate::types::{Trap, TrapType};
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// Configuration for the two weak-link barriers forming the Josephson junctions.
///
/// The barriers are placed at angular positions \(\theta_1, \theta_2\) on a ring
/// of radius \(R\). Their kinematics are:
/// \(\dot\theta_1 = \Omega_{\mathrm{ext}} + 2\pi f\),
/// \(\dot\theta_2 = \Omega_{\mathrm{ext}} - 2\pi f\).
pub struct BarrierConfig {
    /// Barrier height \(U_b\) (dimensionless)
    pub height: f64,
    /// Gaussian width \(\sigma_b\) (dimensionless, in units of \(\ell_x\))
    pub width: f64,
    /// Angular position of junction 1 [rad]
    pub theta_1: f64,
    /// Angular position of junction 2 [rad]
    pub theta_2: f64,
    /// Common rotation rate \(\Omega_{\mathrm{ext}}\) [rad/sim-time]
    pub omega_ext: f64,
    /// Differential frequency \(f\) (dimensionless, bias current \(= 4f\))
    pub f: f64,
}

impl BarrierConfig {
    /// Create a new barrier configuration.
    ///
    /// Junction 1 starts at \(\theta_1 = 0\), junction 2 at \(\theta_2 = \pi\)
    /// (diametrically opposed).
    pub fn new(height: f64, width: f64, omega_ext: f64, f: f64) -> Self {
        Self {
            height,
            width,
            theta_1: 0.0,
            theta_2: PI,
            omega_ext,
            f,
        }
    }

    /// Advance both junction angles by one timestep.
    ///
    /// \(\theta_1 \mathrel{+}= (\Omega_{\mathrm{ext}} + 2\pi f)\,\mathrm{d}t\),
    /// \(\theta_2 \mathrel{+}= (\Omega_{\mathrm{ext}} - 2\pi f)\,\mathrm{d}t\).
    pub fn step(&mut self, dt: f64) {
        self.theta_1 += (self.omega_ext + 2.0 * PI * self.f) * dt;
        self.theta_2 += (self.omega_ext - 2.0 * PI * self.f) * dt;
    }
}

/// Calculates the harmonic potential in dimensionless units.
///
/// The coordinates \(x, y\) are scaled by \(\ell_x = \sqrt{\hbar/(m\omega_x)}\), so the
/// dimensionless harmonic potential is
/// \[
/// \tilde{V}(\tilde x,\tilde y) = \frac{1}{2}\tilde x^2 + \frac{1}{2}\left(\frac{\omega_y}{\omega_x}\right)^2 \tilde y^2 .
/// \]
/// Energy is in units of \(\hbar\omega_x\).
pub fn harmonic_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    let aspect_ratio_y = trap.frequency_y / trap.frequency_x;

    let potential_x = 0.5 * x.mapv(|x| x.powi(2)).into_shape((x.len(), 1)).unwrap();
    let potential_y =
        0.5 * aspect_ratio_y.powi(2) * y.mapv(|y| y.powi(2)).into_shape((1, y.len())).unwrap();

    let potential = potential_x + potential_y;
    potential.mapv(|val| Complex::new(val, 0.0))
}

/// Calculates the toroidal potential.
///
/// \[
/// V(r) = V_0\left(1 - e^{-\frac{(r - R)^2}{2\sigma^2}}\right)
/// \]
pub fn toroidal_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    let depth = trap.depth.expect("Depth is required for a toroidal trap");
    let ring_radius = trap
        .ring_radius
        .expect("Ring radius is required for a toroidal trap");
    let trap_radius = trap
        .trap_radius
        .expect("Trap radius is required for a toroidal trap");

    debug_assert!(trap_radius > 0.0, "trap_radius must be positive");

    let x_sq = x.mapv(|v| v.powi(2)).into_shape((x.len(), 1)).unwrap(); // (nx, 1)
    let y_sq = y.mapv(|v| v.powi(2)).into_shape((1, y.len())).unwrap(); // (1, ny)
    let r = (&x_sq + &y_sq).mapv(f64::sqrt); // (nx, ny) via broadcast

    let potential = depth
        * (1.0
            - (-0.5 / trap_radius.powi(2) * (r - ring_radius).mapv(|v| v.powi(2))).mapv(f64::exp));

    potential.mapv(|val| Complex::new(val, 0.0))
}

/// Selects and calculates the appropriate potential based on trap type.
pub fn calculate_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    match trap.trap_type {
        TrapType::Harmonic | TrapType::Cigar => harmonic_potential(x, y, trap),
        TrapType::Toroidal => toroidal_potential(x, y, trap),
    }
}

/// Computes the barrier potential from two Gaussian weak links.
///
/// \[
/// V_{\mathrm{bar}}(\mathbf{r}, t) = \sum_{j=1,2} U_b
///     \exp\!\left(-\frac{\|\mathbf{r} - \mathbf{r}_j(t)\|^2}{2\sigma_b^2}\right)
/// \]
/// where \(\mathbf{r}_j(t) = R(\cos\theta_j(t), \sin\theta_j(t))\).
///
/// # Panics
///
/// Panics if `trap.ring_radius` is `None` (programmer error: the barrier potential
/// requires a ring geometry).
pub fn barrier_potential(
    x: &Array1<f64>,
    y: &Array1<f64>,
    trap: &Trap,
    config: &BarrierConfig,
) -> Array2<Complex<f64>> {
    let ring_radius = trap
        .ring_radius
        .expect("ring_radius required for barrier potential");

    debug_assert!(config.width > 0.0, "barrier width must be positive");

    let x_sq = x.mapv(|v| v.powi(2)).into_shape((x.len(), 1)).unwrap(); // (nx, 1)
    let y_sq = y.mapv(|v| v.powi(2)).into_shape((1, y.len())).unwrap(); // (1, ny)
    let r_sq = &x_sq + &y_sq; // (nx, ny) — x[i]^2 + y[j]^2 at each grid point
    let x_col = x.clone().into_shape((x.len(), 1)).unwrap(); // (nx, 1)
    let y_row = y.clone().into_shape((1, y.len())).unwrap(); // (1, ny)

    let r0 = ring_radius;
    let inv_two_sigma_sq = 1.0 / (2.0 * config.width.powi(2));

    let mut result = Array2::zeros((x.len(), y.len()));
    for &theta in &[config.theta_1, config.theta_2] {
        let cross = &x_col * theta.cos() + &y_row * theta.sin(); // (nx, ny)
        let dist_sq = &r_sq + (r0 * r0) - &(&cross * (2.0 * r0)); // (nx, ny)
        result = &result + &dist_sq.mapv(|d2| (-d2 * inv_two_sigma_sq).exp());
    }
    (result * config.height).mapv(|val| Complex::new(val, 0.0))
}

/// Computes the total toroidal potential including weak-link barriers.
///
/// Returns \(\texttt{toroidal\_potential}(x, y, \texttt{trap}) + \texttt{barrier\_potential}(x, y, \texttt{trap}, \texttt{config})\).
///
/// # Panics
///
/// Panics if the trap type is not `TrapType::Toroidal`.
pub fn total_potential(
    x: &Array1<f64>,
    y: &Array1<f64>,
    trap: &Trap,
    config: &BarrierConfig,
) -> Array2<Complex<f64>> {
    match trap.trap_type {
        TrapType::Toroidal => {
            toroidal_potential(x, y, trap) + barrier_potential(x, y, trap, config)
        }
        _ => panic!("total_potential is only valid for Toroidal trap type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn test_trap() -> Trap {
        Trap {
            trap_type: TrapType::Harmonic,
            frequency_x: 2.0 * PI * 25.0,
            frequency_y: 2.0 * PI * 25.0,
            frequency_z: 2.0 * PI * 100.0,
            depth: None,
            ring_radius: None,
            trap_radius: None,
        }
    }

    fn test_toroidal_trap() -> Trap {
        Trap {
            trap_type: TrapType::Toroidal,
            frequency_x: 2.0 * PI * 570.0,
            frequency_y: 2.0 * PI * 570.0,
            frequency_z: 2.0 * PI * 300.0,
            depth: Some(1.0),
            ring_radius: Some(10.0),
            trap_radius: Some(2.0),
        }
    }

    fn test_barrier_config() -> BarrierConfig {
        BarrierConfig::new(1.0, 1.5, 0.0, 0.0)
    }

    fn test_x() -> Array1<f64> {
        Array1::linspace(-10.0, 10.0, 33)
    }

    fn test_y() -> Array1<f64> {
        Array1::linspace(-10.0, 10.0, 33)
    }

    // --- Moved harmonic tests ---

    #[test]
    fn harmonic_potential_at_origin_is_zero() {
        let trap = test_trap();
        let x = test_x();
        let y = test_y();
        let v = harmonic_potential(&x, &y, &trap);
        // With 33 points, the midpoint index 16 is exactly 0.0
        let mid = 16;
        let val = v[[mid, mid]].re;
        assert!(val.abs() < 1e-10, "V(0,0) = {} should be ~0", val);
    }

    #[test]
    fn harmonic_potential_is_positive() {
        let trap = test_trap();
        let x = test_x();
        let y = test_y();
        let v = harmonic_potential(&x, &y, &trap);
        assert!(v.iter().all(|c| c.re >= 0.0));
    }

    #[test]
    fn harmonic_potential_is_real() {
        let trap = test_trap();
        let x = test_x();
        let y = test_y();
        let v = harmonic_potential(&x, &y, &trap);
        assert!(
            v.iter().all(|c| c.im.abs() < 1e-15),
            "harmonic potential should be purely real"
        );
    }

    #[test]
    fn harmonic_potential_isotropic_when_omega_y_equals_omega_x() {
        let trap = test_trap(); // frequency_y == frequency_x == 2*PI*25
        let x = test_x();
        let y = test_y();
        let v = harmonic_potential(&x, &y, &trap);

        let step = 20.0 / 32.0; // linspace(-10, 10, 33)
        let idx_of = |val: f64| -> usize { ((val + 10.0) / step).round() as usize };

        let mid = idx_of(0.0);
        let i = idx_of(5.0);
        // V(x, 0) should equal V(0, x) when omega_y == omega_x
        let v_x0 = v[[i, mid]].re;
        let v_0x = v[[mid, i]].re;
        let rel_diff = (v_x0 - v_0x).abs() / v_x0;
        assert!(
            rel_diff < 1e-10,
            "V({:.3}, 0) = {} != V(0, {:.3}) = {}",
            x[i],
            v_x0,
            y[i],
            v_0x
        );
    }

    #[test]
    fn harmonic_potential_grows_quadratically() {
        let trap = test_trap();
        let x = test_x();
        let y = test_y();
        let v = harmonic_potential(&x, &y, &trap);

        let step = 20.0 / 32.0;
        let idx_of = |val: f64| -> usize { ((val + 10.0) / step).round() as usize };

        let mid = idx_of(0.0);
        let i1 = idx_of(2.5);
        let i2 = idx_of(5.0);
        // V(2 * x, 0) should equal 4 * V(x, 0) for a harmonic potential
        let v1 = v[[i1, mid]].re;
        let v2 = v[[i2, mid]].re;
        let ratio = v2 / v1;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "V(5.0) / V(2.5) = {}, expected 4.0",
            ratio
        );
    }

    // --- New toroidal and barrier tests ---

    #[test]
    fn toroidal_potential_is_nonnegative() {
        let trap = test_toroidal_trap();
        let x = test_x();
        let y = test_y();
        let v = toroidal_potential(&x, &y, &trap);
        assert!(v.iter().all(|c| c.re >= 0.0));
    }

    #[test]
    fn toroidal_potential_minimum_on_ring() {
        let trap = test_toroidal_trap();
        let x = test_x();
        let y = test_y();
        let v = toroidal_potential(&x, &y, &trap);

        // The ring minimum is at r = R = 10. Grid points at (0, +/-10) lie on the ring.
        // v[i, j] = V(x[i], y[j]) with row = x-index, column = y-index,
        // so v[[i_x, i_y]] corresponds to (x[i_x], y[i_y]).
        let step = 20.0 / 32.0;
        let idx_of = |val: f64| -> usize { ((val + 10.0) / step).round() as usize };

        let i_x = idx_of(0.0);
        let i_y = idx_of(10.0);

        // V at (0, 10) should be ~0 (r = R)
        let val = v[[i_x, i_y]].re;
        assert!(
            val.abs() < 1e-10,
            "V at ring (r = R) should be ~0, got {}",
            val
        );

        // V at (0, -10) should also be ~0
        let i_y_neg = idx_of(-10.0);
        let val2 = v[[i_x, i_y_neg]].re;
        assert!(
            val2.abs() < 1e-10,
            "V at ring (r = R) should be ~0, got {}",
            val2
        );
    }

    #[test]
    fn barrier_potential_peak_height() {
        let trap = test_toroidal_trap();
        let config = test_barrier_config(); // height=1.0, width=1.5
        let x = test_x();
        let y = test_y();
        let v = barrier_potential(&x, &y, &trap, &config);

        // With theta_1 = 0, the first barrier centre is exactly at a grid point
        // (R, 0) = (10, 0), so the Gaussian evaluates to U_b at that point.
        let max_val = v.iter().map(|c| c.re).fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_val - config.height).abs() < 1e-10,
            "barrier peak should be U_b = {}, got {}",
            config.height,
            max_val
        );
    }

    #[test]
    fn barrier_potential_peak_at_theta() {
        let trap = test_toroidal_trap();
        let config = test_barrier_config(); // theta_1 = 0
        let x = test_x();
        let y = test_y();
        let v = barrier_potential(&x, &y, &trap, &config);

        // v[i, j] = V(x[i], y[j]) so v[[i_x, i_y]] = V(x[i_x], y[i_y]).
        let step = 20.0 / 32.0;
        let idx_of = |val: f64| -> usize { ((val + 10.0) / step).round() as usize };

        let i_x_r = idx_of(10.0); // x-index of R
        let i_y_0 = idx_of(0.0); // y-index of 0

        // Barrier 1 is at (R, 0) = (10.0, 0.0) when theta_1 = 0.
        let val_at_peak = v[[i_x_r, i_y_0]].re;
        let global_max = v.iter().map(|c| c.re).fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (val_at_peak - global_max).abs() < 1e-10,
            "barrier 1 peak at theta = 0 should equal the global maximum"
        );
    }

    #[test]
    fn total_potential_is_nonnegative() {
        let trap = test_toroidal_trap();
        let config = test_barrier_config();
        let x = test_x();
        let y = test_y();
        let v = total_potential(&x, &y, &trap, &config);
        assert!(v.iter().all(|c| c.re >= 0.0));
    }

    #[test]
    fn barrier_step_advances_angles() {
        let mut config = BarrierConfig::new(1.0, 1.5, 2.0, 0.5);
        let dt = 0.1;

        let theta_1_before = config.theta_1;
        let theta_2_before = config.theta_2;

        let expected_dtheta_1 = (2.0 + 2.0 * PI * 0.5) * dt;
        let expected_dtheta_2 = (2.0 - 2.0 * PI * 0.5) * dt;

        config.step(dt);

        assert!(
            (config.theta_1 - theta_1_before - expected_dtheta_1).abs() < 1e-15,
            "theta_1 advance incorrect"
        );
        assert!(
            (config.theta_2 - theta_2_before - expected_dtheta_2).abs() < 1e-15,
            "theta_2 advance incorrect"
        );
    }

    // --- BarrierConfig initial angle invariants ---

    #[test]
    fn barrier_config_new_sets_initial_angles() {
        let config = BarrierConfig::new(1.0, 1.5, 0.0, 0.0);
        assert!(
            (config.theta_1 - 0.0).abs() < 1e-15,
            "theta_1 should be 0, got {}",
            config.theta_1
        );
        assert!(
            (config.theta_2 - PI).abs() < 1e-15,
            "theta_2 should be PI, got {}",
            config.theta_2
        );
    }

    #[test]
    fn barrier_step_no_drive_no_rotation() {
        let mut config = BarrierConfig::new(1.0, 1.5, 0.0, 0.0);
        let dt = 0.1;
        let theta_1_before = config.theta_1;
        let theta_2_before = config.theta_2;
        config.step(dt);
        assert!(
            (config.theta_1 - theta_1_before).abs() < 1e-15,
            "theta_1 should not change when omega_ext = 0 and f = 0"
        );
        assert!(
            (config.theta_2 - theta_2_before).abs() < 1e-15,
            "theta_2 should not change when omega_ext = 0 and f = 0"
        );
    }

    #[test]
    fn barrier_step_f_only() {
        let mut config = BarrierConfig::new(1.0, 1.5, 0.0, 0.5);
        let dt = 0.1;
        let theta_1_before = config.theta_1;
        let theta_2_before = config.theta_2;
        let expected_dtheta_1 = 2.0 * PI * 0.5 * dt;
        let expected_dtheta_2 = -2.0 * PI * 0.5 * dt;
        config.step(dt);
        assert!(
            (config.theta_1 - theta_1_before - expected_dtheta_1).abs() < 1e-15,
            "theta_1 advance with f only should be {}, got {}",
            expected_dtheta_1,
            config.theta_1 - theta_1_before
        );
        assert!(
            (config.theta_2 - theta_2_before - expected_dtheta_2).abs() < 1e-15,
            "theta_2 advance with f only should be {}, got {}",
            expected_dtheta_2,
            config.theta_2 - theta_2_before
        );
    }

    #[test]
    fn barrier_step_omega_ext_only() {
        let mut config = BarrierConfig::new(1.0, 1.5, 2.0, 0.0);
        let dt = 0.1;
        let theta_1_before = config.theta_1;
        let theta_2_before = config.theta_2;
        let expected_dtheta = 2.0 * dt;
        config.step(dt);
        assert!(
            (config.theta_1 - theta_1_before - expected_dtheta).abs() < 1e-15,
            "theta_1 advance with omega_ext only should be {}, got {}",
            expected_dtheta,
            config.theta_1 - theta_1_before
        );
        assert!(
            (config.theta_2 - theta_2_before - expected_dtheta).abs() < 1e-15,
            "theta_2 advance with omega_ext only should be {}, got {}",
            expected_dtheta,
            config.theta_2 - theta_2_before
        );
    }

    // --- calculate_potential dispatch tests ---

    #[test]
    fn calculate_potential_harmonic_matches_harmonic_potential() {
        let x = test_x();
        let y = test_y();
        let trap = test_trap();
        let expected = harmonic_potential(&x, &y, &trap);
        let result = calculate_potential(&x, &y, &trap);
        assert_eq!(result.shape(), expected.shape());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r.re - e.re).abs() < 1e-15, "real part mismatch");
            assert!((r.im - e.im).abs() < 1e-15, "imag part mismatch");
        }
    }

    #[test]
    fn calculate_potential_toroidal_matches_toroidal_potential() {
        let x = test_x();
        let y = test_y();
        let trap = test_toroidal_trap();
        let expected = toroidal_potential(&x, &y, &trap);
        let result = calculate_potential(&x, &y, &trap);
        assert_eq!(result.shape(), expected.shape());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r.re - e.re).abs() < 1e-15, "real part mismatch");
            assert!((r.im - e.im).abs() < 1e-15, "imag part mismatch");
        }
    }

    // --- total_potential error handling ---

    #[test]
    #[should_panic(expected = "total_potential is only valid for Toroidal trap type")]
    fn total_potential_panics_for_harmonic_trap() {
        let x = test_x();
        let y = test_y();
        let trap = test_trap();
        let config = test_barrier_config();
        let _v = total_potential(&x, &y, &trap, &config);
    }

    // --- Array shape invariants ---

    #[test]
    fn harmonic_potential_shape() {
        let x = Array1::linspace(-10.0, 10.0, 50);
        let y = Array1::linspace(-10.0, 10.0, 30);
        let trap = test_trap();
        let v = harmonic_potential(&x, &y, &trap);
        assert_eq!(v.shape(), &[50, 30]);
    }

    #[test]
    fn toroidal_potential_shape() {
        let x = Array1::linspace(-10.0, 10.0, 50);
        let y = Array1::linspace(-10.0, 10.0, 30);
        let trap = test_toroidal_trap();
        let v = toroidal_potential(&x, &y, &trap);
        assert_eq!(v.shape(), &[50, 30]);
    }

    #[test]
    fn barrier_potential_shape() {
        let x = Array1::linspace(-10.0, 10.0, 50);
        let y = Array1::linspace(-10.0, 10.0, 30);
        let trap = test_toroidal_trap();
        let config = test_barrier_config();
        let v = barrier_potential(&x, &y, &trap, &config);
        assert_eq!(v.shape(), &[50, 30]);
    }
}
