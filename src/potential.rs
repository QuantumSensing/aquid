//! Trap and barrier potentials for the SGPE simulation.
//!
//! Provides analytic forms for the harmonic and toroidal ring traps, and the
//! two Gaussian weak-link barriers that form the atomtronic SQUID.

use super::constants::*;
use super::types::*;
use ndarray::{Array1, Array2};
use num_complex::Complex;

/// Configuration for the two weak-link laser barriers forming the atomtronic SQUID.
///
/// The junctions co-rotate at \(\Omega_{\mathrm{ext}}\) and are driven differentially
/// to inject bias. Kinematics:
///
/// \[
/// \dot\theta_1 = \Omega_{\mathrm{ext}} + 2\pi f(t), \qquad
/// \dot\theta_2 = \Omega_{\mathrm{ext}} - 2\pi f(t)
/// \]
///
/// The differential frequency \(f\) is ramped linearly from 0 to `f_target` over
/// `ramp_duration`, then held at `f_target` for `hold_duration`.
#[derive(Debug, Clone)]
pub struct BarrierConfig {
    /// Initial angular position of junction 1 [rad]
    pub theta_1_init: f64,
    /// Initial angular position of junction 2 [rad] (typically \(\theta_1 + \pi\))
    pub theta_2_init: f64,
    /// Common-mode rotation rate \(\Omega_{\mathrm{ext}}\) [rad/s]
    pub omega_ext: f64,
    /// Differential drive frequency set-point [Hz]
    pub f_target: f64,
    /// Ramp duration for \(f\): 0 → f_target [s]
    pub ramp_duration: f64,
    /// Hold duration after ramp completes [s]
    pub hold_duration: f64,
}

impl BarrierConfig {
    /// Create a new barrier configuration with diametrically opposed junctions.
    ///
    /// Junction 2 is initialised at \(\theta_1 + \pi\).
    pub fn new(
        theta_1_init: f64,
        omega_ext: f64,
        f_target: f64,
        ramp_duration: f64,
        hold_duration: f64,
    ) -> Self {
        Self {
            theta_1_init,
            theta_2_init: theta_1_init + PI,
            omega_ext,
            f_target,
            ramp_duration,
            hold_duration,
        }
    }

    /// Integrated differential phase contribution at time \(t\):
    /// \(\Delta\phi(t) = 2\pi\int_0^t f(s)\,ds\).
    ///
    /// For a linear ramp \(f(s) = f_{\mathrm{target}}\,s / t_{\mathrm{ramp}}\) on
    /// \([0, t_{\mathrm{ramp}}]\) and constant \(f_{\mathrm{target}}\) thereafter:
    ///
    /// \[
    /// \Delta\phi(t) =
    /// \begin{cases}
    /// 0 & t \le 0,\\[4pt]
    /// \pi\,f_{\mathrm{target}}\,t^2 / t_{\mathrm{ramp}} & 0 < t \le t_{\mathrm{ramp}},\\[4pt]
    /// \pi\,f_{\mathrm{target}}\,t_{\mathrm{ramp}}
    ///   + 2\pi\,f_{\mathrm{target}}\,(t - t_{\mathrm{ramp}}) & t > t_{\mathrm{ramp}}.
    ///
    /// \end{cases}
    ///
    /// \]
    fn differential_phase(&self, t: f64) -> f64 {
        if t <= 0.0 {
            0.0
        } else if t <= self.ramp_duration {
            PI * self.f_target * t * t / self.ramp_duration
        } else {
            PI * self.f_target * self.ramp_duration
                + 2.0 * PI * self.f_target * (t - self.ramp_duration)
        }
    }

    /// Return the angular positions \((\theta_1, \theta_2)\) of the two barriers at
    /// elapsed time \(t\) [s].
    ///
    /// \[
    /// \begin{aligned}
    /// \theta_1(t) &= \theta_1(0) + \Omega_{\mathrm{ext}}\,t + \Delta\phi(t),\\
    /// \theta_2(t) &= \theta_2(0) + \Omega_{\mathrm{ext}}\,t - \Delta\phi(t),
    /// \end{aligned}
    /// \]
    ///
    /// where \(\Delta\phi(t) = 2\pi\int_0^t f(s)\,ds\) is the integrated differential
    /// phase from [`differential_phase`].
    pub fn angles_at(&self, t: f64) -> (f64, f64) {
        let delta = self.differential_phase(t);
        let common = self.omega_ext * t;
        let theta_1 = self.theta_1_init + common + delta;
        let theta_2 = self.theta_2_init + common - delta;
        (theta_1, theta_2)
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
///
/// The returned array has shape \((n_x, n_y)\) with axis 0 = \(x\), axis 1 = \(y\).
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
/// V(r) = V_0\left(1 - e^{-\frac{(r-R)^2}{2\sigma^2}}\right)
/// \]
///
/// The returned array has shape \((n_x, n_y)\) with axis 0 = \(x\), axis 1 = \(y\).
pub fn toroidal_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    let depth = trap.depth.expect("Depth is required for a toroidal trap");
    assert!(
        depth >= 0.0,
        "toroidal trap depth must be non-negative, got {}",
        depth
    );
    let ring_radius = trap
        .ring_radius
        .expect("Ring radius is required for a toroidal trap");
    let trap_radius = trap
        .trap_radius
        .expect("Trap radius is required for a toroidal trap");

    let nx = x.len();
    let ny = y.len();

    // Build separable (nx, ny) grid matching the wavefunction convention:
    // axis 0 = x, axis 1 = y.
    let x_sq = x.mapv(|v| v.powi(2)).into_shape((nx, 1)).unwrap();
    let y_sq = y.mapv(|v| v.powi(2)).into_shape((1, ny)).unwrap();
    let rho = (x_sq + y_sq).mapv(f64::sqrt);

    let exponent = -1.0 / trap_radius.powi(2) * (&rho - ring_radius).mapv(|r| r.powi(2));
    let potential = depth * (1.0 - exponent.mapv(f64::exp));

    potential.mapv(|val| Complex::new(val, 0.0))
}

/// Selects and calculates the appropriate potential based on trap type.
pub fn calculate_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    match trap.trap_type {
        TrapType::Harmonic | TrapType::Cigar => harmonic_potential(x, y, trap),
        TrapType::Toroidal => toroidal_potential(x, y, trap),
    }
}

/// Computes the two Gaussian barrier potentials at time \(t\).
///
/// \[
/// V_{\mathrm{bar}}(\mathbf{r}, t) = \sum_{j=1,2} U_b\,
/// \exp\!\left[-\frac{\lVert\mathbf{r} - \mathbf{r}_j(t)\rVert^2}{2\sigma_b^2}\right],
/// \qquad
/// \mathbf{r}_j(t) = R\,(\cos\theta_j(t), \sin\theta_j(t))
/// \]
///
/// where \(U_b\) is the barrier height, \(\sigma_b\) the Gaussian width,
/// \(R\) the ring radius, and \(\theta_j(t)\) from [`BarrierConfig::angles_at`].
///
/// The returned array has shape \((n_x, n_y)\) with axis 0 = \(x\), axis 1 = \(y\).
pub fn barrier_potential(
    x: &Array1<f64>,
    y: &Array1<f64>,
    t: f64,
    config: &BarrierConfig,
    barrier_height: f64,
    barrier_width: f64,
    ring_radius: f64,
) -> Array2<Complex<f64>> {
    assert!(
        barrier_width > 0.0,
        "barrier_width must be positive, got {}",
        barrier_width
    );
    assert!(
        barrier_height >= 0.0,
        "barrier_height must be non-negative, got {}",
        barrier_height
    );

    let (theta_1, theta_2) = config.angles_at(t);

    let nx = x.len();
    let ny = y.len();
    let inv_two_sigma_sq = 1.0 / (2.0 * barrier_width.powi(2));

    let x1 = ring_radius * theta_1.cos();
    let y1 = ring_radius * theta_1.sin();
    let x2 = ring_radius * theta_2.cos();
    let y2 = ring_radius * theta_2.sin();

    // Barrier 1: separable (nx, ny) grid via into_shape, axis 0 = x, axis 1 = y
    let dx1_sq = x.mapv(|v| (v - x1).powi(2)).into_shape((nx, 1)).unwrap();
    let dy1_sq = y.mapv(|v| (v - y1).powi(2)).into_shape((1, ny)).unwrap();
    let mut barrier = (-(dx1_sq + dy1_sq) * inv_two_sigma_sq).mapv(f64::exp);

    // Barrier 2: accumulate in-place
    let dx2_sq = x.mapv(|v| (v - x2).powi(2)).into_shape((nx, 1)).unwrap();
    let dy2_sq = y.mapv(|v| (v - y2).powi(2)).into_shape((1, ny)).unwrap();
    let barrier_2 = (-(dx2_sq + dy2_sq) * inv_two_sigma_sq).mapv(f64::exp);
    barrier += &barrier_2;

    (barrier_height * barrier).mapv(|val| Complex::new(val, 0.0))
}

/// Computes the total potential: ring trap + two weak-link barriers.
///
/// \[
/// V_{\mathrm{total}}(\mathbf{r}, t) = V_{\mathrm{trap}}(\mathbf{r}) + V_{\mathrm{bar}}(\mathbf{r}, t)
/// \]
pub fn total_potential(
    x: &Array1<f64>,
    y: &Array1<f64>,
    t: f64,
    trap: &Trap,
    config: &BarrierConfig,
    barrier_height: f64,
    barrier_width: f64,
    ring_radius: f64,
) -> Array2<Complex<f64>> {
    let mut ring = calculate_potential(x, y, trap);
    let barriers = barrier_potential(x, y, t, config, barrier_height, barrier_width, ring_radius);
    ring += &barriers;
    ring
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Test helpers ---

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
            frequency_x: 2.0 * PI * 25.0,
            frequency_y: 2.0 * PI * 25.0,
            frequency_z: 2.0 * PI * 100.0,
            depth: Some(10.0),
            ring_radius: Some(10.0),
            trap_radius: Some(2.0),
        }
    }

    fn test_barrier_config() -> BarrierConfig {
        BarrierConfig::new(0.0, 0.0, 0.0, 1.0, 0.0)
    }

    fn test_x() -> Array1<f64> {
        Array1::linspace(-10.0, 10.0, 33)
    }

    fn test_y() -> Array1<f64> {
        Array1::linspace(-10.0, 10.0, 33)
    }

    // --- BarrierConfig tests ---

    #[test]
    fn barrier_config_diametrically_opposed() {
        let config = BarrierConfig::new(0.0, 1.0, 1.0, 1.0, 1.0);
        let diff = config.theta_2_init - config.theta_1_init - PI;
        assert!(
            diff.abs() < 1e-15,
            "theta_2_init should be theta_1_init + pi; diff = {:.2e}",
            diff
        );
    }

    #[test]
    fn barrier_config_angles_at_t0() {
        let theta_1_init = 0.5;
        let config = BarrierConfig::new(theta_1_init, 1.0, 1.0, 1.0, 1.0);
        let (t1, t2) = config.angles_at(0.0);
        assert!(
            (t1 - theta_1_init).abs() < 1e-15,
            "theta_1(0) = {}, expected {}",
            t1,
            theta_1_init
        );
        assert!(
            (t2 - (theta_1_init + PI)).abs() < 1e-15,
            "theta_2(0) = {}, expected {}",
            t2,
            theta_1_init + PI
        );
    }

    #[test]
    fn barrier_config_angles_ramp() {
        // f_target = 2 Hz, ramp_duration = 2 s, omega_ext = 0.
        // At t = 1.0 s (mid-ramp):
        //   ∆φ = π·f_target·t² / ramp_duration = π·2·1/2 = π
        //   θ₁ = 0 + 0 + π = π
        //   θ₂ = π + 0 − π = 0
        let config = BarrierConfig::new(0.0, 0.0, 2.0, 2.0, 1.0);
        let (t1, t2) = config.angles_at(1.0);
        assert!(
            (t1 - PI).abs() < 1e-10,
            "theta_1 = {}, expected pi",
            t1
        );
        assert!(
            t2.abs() < 1e-10,
            "theta_2 = {}, expected 0",
            t2
        );
    }

    #[test]
    fn barrier_config_angles_hold() {
        // f_target = 1 Hz, ramp_duration = 1 s, omega_ext = 0.
        // At t = 2.0 s (1 s after ramp complete):
        //   ∆φ = π·f_target·t_ramp + 2π·f_target·(t − t_ramp)
        //      = π·1·1 + 2π·1·(2−1) = π + 2π = 3π
        //   θ₁ = 0 + 0 + 3π = 3π
        let config = BarrierConfig::new(0.0, 0.0, 1.0, 1.0, 1.0);
        let (t1, _) = config.angles_at(2.0);
        let expected = 3.0 * PI;
        assert!(
            (t1 - expected).abs() < 1e-10,
            "theta_1 = {}, expected {}",
            t1,
            expected
        );
    }

    #[test]
    fn barrier_config_sweep_small() {
        // Total differential sweep should be small (order few degrees)
        // for realistic parameters.
        // f_target = 0.01 Hz, ramp_duration = 1.0 s, t = 1.0 s
        // ∆φ = π·f_target·t² / ramp_duration = π·0.01 ≈ 0.0314 rad ≈ 1.8°
        let config = BarrierConfig::new(0.0, 0.0, 0.01, 1.0, 0.0);
        let (t1, t2) = config.angles_at(1.0);
        // θ₁ − θ₂ = (θ₁(0) + ∆φ) − (θ₂(0) − ∆φ) = −π + 2∆φ
        let differential_sweep = (t1 - t2 + PI).abs();
        assert!(
            differential_sweep < 0.2,
            "differential sweep = {} rad should be small (< 0.2 rad = 11 deg)",
            differential_sweep
        );
    }

    #[test]
    fn barrier_config_differential_phase_ramp() {
        let config = BarrierConfig::new(0.0, 0.0, 2.0, 2.0, 0.0);
        // At mid-ramp, ∆φ should be half of what it would be at t=ramp_duration
        //   ∆φ(t_ramp) = π·f_target·t_ramp = π·2·2 = 4π
        //   ∆φ(t_ramp/2) = π·2·1²/2 = π
        let delta_mid = config.differential_phase(1.0);
        let delta_end = config.differential_phase(2.0);
        assert!((delta_mid - PI).abs() < 1e-10);
        assert!((delta_end - 4.0 * PI).abs() < 1e-10);
        // Mid-point should be 1/4 of end-point (quadratic, not linear)
        assert!((delta_mid / delta_end - 0.25).abs() < 1e-10);
    }

    #[test]
    fn barrier_potential_is_real() {
        let nx = 64;
        let ny = 64;
        let x = Array1::linspace(-20.0, 20.0, nx);
        let y = Array1::linspace(-20.0, 20.0, ny);
        let config = BarrierConfig::new(0.0, 0.0, 0.0, 1.0, 0.0);
        let v = barrier_potential(&x, &y, 0.0, &config, 1.0, 1.0, 10.0);
        assert!(
            v.iter().all(|c| c.im.abs() < 1e-15),
            "barrier potential should be purely real"
        );
    }

    #[test]
    fn barrier_potential_is_nonnegative() {
        let nx = 64;
        let ny = 64;
        let x = Array1::linspace(-20.0, 20.0, nx);
        let y = Array1::linspace(-20.0, 20.0, ny);
        let config = BarrierConfig::new(0.0, 0.0, 0.0, 1.0, 0.0);
        let v = barrier_potential(&x, &y, 0.0, &config, 1.0, 1.0, 10.0);
        assert!(v.iter().all(|c| c.re >= 0.0));
    }

    #[test]
    fn barrier_potential_two_peaks() {
        let nx = 128;
        let ny = 128;
        let x = Array1::linspace(-20.0, 20.0, nx);
        let y = Array1::linspace(-20.0, 20.0, ny);
        let ring_radius = 10.0;
        let config = BarrierConfig::new(0.0, 0.0, 0.0, 1.0, 0.0);
        let barrier_width = 1.0;
        let v = barrier_potential(&x, &y, 0.0, &config, 1.0, barrier_width, ring_radius);

        // Array shape is (nx, ny) with v[[i, j]] = V(x[i], y[j]).
        let step = 40.0 / ((nx - 1) as f64);
        let idx_of = |val: f64| -> usize { ((val + 20.0) / step).round() as usize };

        let idx_x1 = idx_of(ring_radius);   // x-index of +R
        let idx_x2 = idx_of(-ring_radius);  // x-index of -R
        let idx_mid = idx_of(0.0);          // index of 0

        // Barriers centred at (+R, 0) and (−R, 0); evaluate along y = 0
        let v_c1 = v[[idx_x1, idx_mid]].re;
        let v_c2 = v[[idx_x2, idx_mid]].re;
        let v_mid = v[[idx_mid, idx_mid]].re;

        assert!(
            v_c1 > 0.5,
            "V at (R, 0) = {} should be near barrier_height",
            v_c1
        );
        assert!(
            v_c2 > 0.5,
            "V at (-R, 0) = {} should be near barrier_height",
            v_c2
        );

        // The potential at the origin should be lower than at the peaks
        assert!(
            v_mid < v_c1,
            "V at origin = {} should be less than peak V = {}",
            v_mid,
            v_c1
        );
    }

    #[test]
    fn total_potential_is_real_and_nonnegative() {
        let nx = 64;
        let ny = 64;
        let x = Array1::linspace(-20.0, 20.0, nx);
        let y = Array1::linspace(-20.0, 20.0, ny);
        let trap = Trap {
            trap_type: TrapType::Toroidal,
            frequency_x: 2.0 * PI * 25.0,
            frequency_y: 2.0 * PI * 25.0,
            frequency_z: 2.0 * PI * 100.0,
            depth: Some(10.0),
            ring_radius: Some(10.0),
            trap_radius: Some(2.0),
        };
        let config = BarrierConfig::new(0.0, 0.0, 0.0, 1.0, 0.0);
        let v = total_potential(&x, &y, 0.0, &trap, &config, 1.0, 1.0, 10.0);
        assert!(
            v.iter().all(|c| c.im.abs() < 1e-15),
            "total potential should be purely real"
        );
        assert!(
            v.iter().all(|c| c.re >= 0.0),
            "total potential should be non-negative"
        );
    }

    #[test]
    #[should_panic(expected = "barrier_width must be positive")]
    fn barrier_potential_rejects_zero_width() {
        let x = Array1::linspace(-10.0, 10.0, 32);
        let y = Array1::linspace(-10.0, 10.0, 32);
        let config = BarrierConfig::new(0.0, 0.0, 0.0, 1.0, 0.0);
        barrier_potential(&x, &y, 0.0, &config, 1.0, 0.0, 10.0);
    }

    #[test]
    #[should_panic(expected = "barrier_height must be non-negative")]
    fn barrier_potential_rejects_negative_height() {
        let x = Array1::linspace(-10.0, 10.0, 32);
        let y = Array1::linspace(-10.0, 10.0, 32);
        let config = BarrierConfig::new(0.0, 0.0, 0.0, 1.0, 0.0);
        barrier_potential(&x, &y, 0.0, &config, -1.0, 1.0, 10.0);
    }

    // --- Toroidal trap tests ---

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

        let step = 20.0 / 32.0;
        let idx_of = |val: f64| -> usize { ((val + 10.0) / step).round() as usize };

        // v[[i, j]] = V(x[i], y[j]) — on the ring at r = R = 10, V ≈ 0
        let i_x0 = idx_of(0.0);
        let i_y_r = idx_of(10.0);

        let val = v[[i_x0, i_y_r]].re;
        assert!(
            val.abs() < 1e-10,
            "V at (0, R) should be ~0, got {}",
            val
        );
    }

    // --- Barrier peak tests ---

    #[test]
    fn barrier_potential_peak_height() {
        // Barrier centred on a grid point should produce peak = height.
        // With ring_radius = 10.0 and theta_1 = 0, barrier 1 is at (10, 0).
        let x = Array1::linspace(-20.0, 20.0, 65);
        let y = Array1::linspace(-20.0, 20.0, 65);
        let config = test_barrier_config(); // theta_1 = 0, theta_2 = π
        let v = barrier_potential(&x, &y, 0.0, &config, 2.5, 1.5, 10.0);

        let max_val = v.iter().map(|c| c.re).fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_val - 2.5).abs() < 1e-10,
            "barrier peak should be U_b = 2.5, got {}",
            max_val
        );
    }

    // --- calculate_potential dispatch ---

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

    // --- Shape convention tests ---

    #[test]
    fn harmonic_potential_shape() {
        let x = Array1::linspace(-10.0, 10.0, 50);
        let y = Array1::linspace(-10.0, 10.0, 30);
        let trap = test_trap();
        let v = harmonic_potential(&x, &y, &trap);
        assert_eq!(v.shape(), &[50, 30], "shape should be (nx, ny)");
    }

    #[test]
    fn toroidal_potential_shape() {
        let x = Array1::linspace(-10.0, 10.0, 50);
        let y = Array1::linspace(-10.0, 10.0, 30);
        let trap = test_toroidal_trap();
        let v = toroidal_potential(&x, &y, &trap);
        assert_eq!(v.shape(), &[50, 30], "shape should be (nx, ny)");
    }

    #[test]
    fn barrier_potential_shape() {
        let x = Array1::linspace(-10.0, 10.0, 50);
        let y = Array1::linspace(-10.0, 10.0, 30);
        let config = test_barrier_config();
        let v = barrier_potential(&x, &y, 0.0, &config, 1.0, 1.0, 10.0);
        assert_eq!(v.shape(), &[50, 30], "shape should be (nx, ny)");
    }
}
