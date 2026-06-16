//! Implements the Runge-Kutta fourth-order method for SGPE simulations.

use super::constants::*;
use super::types::*;
use super::utils::*;
use crate::potential::calculate_potential;
use crate::projector::Projector;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use rand_distr::{Distribution, StandardNormal};
use rustfft::FftPlanner;
use std::path::Path;
use std::sync::OnceLock;

fn debug_io_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("SGPE_DEBUG_IO")
            .map(|value| value != "0" && !value.is_empty())
            .unwrap_or(false)
    })
}

/// Computes the noise magnitude from the fluctuation-dissipation relation.
///
/// \[
/// \sigma = \sqrt{\frac{2\gamma\tilde{T}\,\Delta t}{\Delta x\,\Delta y}}
/// \]
///
/// This is the per-step amplitude of the complex Wiener noise increment in the
/// dimensionless SGPE (thesis Eq. 3.56). The same amplitude is used for the
/// initial thermal seed \(\psi_0 = \eta\).
pub fn noise_magnitude(gamma: f64, temperature: f64, dt: f64, dx: f64, dy: f64) -> f64 {
    (2.0 * gamma * temperature * dt / (dx * dy)).sqrt()
}

/// Seeds the initial state from a thermal noise distribution.
///
/// \[
/// \psi_0 = \sigma\,(\xi_1 + i\xi_2) / \sqrt{2},
/// \qquad
/// \sigma = \sqrt{\frac{2\gamma\tilde{T}\,\Delta t}{\Delta x\,\Delta y}}
/// \]
///
/// where \(\xi_1, \xi_2\) are independent standard-normal random fields.
/// The amplitude \(\sigma\) matches the per-step Wiener noise increment
/// (thesis Eq. 3.56).
pub fn seed_initial_state(
    gridpoints: (usize, usize),
    gamma: f64,
    temperature: f64,
    dt: f64,
    dx: f64,
    dy: f64,
) -> Array2<Complex<f64>> {
    let amplitude = noise_magnitude(gamma, temperature, dt, dx, dy);
    let mut rng = rand::thread_rng();
    let normal = StandardNormal;
    let norm_factor = amplitude / std::f64::consts::SQRT_2;
    Array2::from_shape_fn(gridpoints, |_| {
        let re = <StandardNormal as Distribution<f64>>::sample(&normal, &mut rng) * norm_factor;
        let im = <StandardNormal as Distribution<f64>>::sample(&normal, &mut rng) * norm_factor;
        Complex::new(re, im)
    })
}

/// Computes the kinetic energy term in Fourier space using a 2D FFT.
pub fn kinetic_energy_fourier(
    phi: &Array2<Complex<f64>>,
    k_sq: &Array2<f64>,
) -> Array2<Complex<f64>> {
    let (nx, ny) = (phi.shape()[0], phi.shape()[1]);

    let mut planner = FftPlanner::new();
    let fft_axis1 = planner.plan_fft_forward(ny);
    let fft_axis0 = planner.plan_fft_forward(nx);
    let ifft_axis1 = planner.plan_fft_inverse(ny);
    let ifft_axis0 = planner.plan_fft_inverse(nx);

    let mut spectrum = phi.clone();

    // Forward FFT along the fast axis (axis 1 / columns).
    for mut row in spectrum.rows_mut() {
        if let Some(row_slice) = row.as_slice_mut() {
            fft_axis1.process(row_slice);
        }
    }

    // Forward FFT along axis 0 (rows) using a scratch buffer per column.
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

    // Apply the kinetic energy operator in k-space.
    for ((i, j), value) in spectrum.indexed_iter_mut() {
        *value *= 0.5 * k_sq[[i, j]];
    }

    // Inverse FFT along axis 0.
    for col_idx in 0..ny {
        for (row_idx, value) in column.iter_mut().enumerate() {
            *value = spectrum[[row_idx, col_idx]];
        }
        ifft_axis0.process(&mut column);
        for (row_idx, value) in column.iter().enumerate() {
            spectrum[[row_idx, col_idx]] = *value;
        }
    }

    // Inverse FFT along axis 1.
    for mut row in spectrum.rows_mut() {
        if let Some(row_slice) = row.as_slice_mut() {
            ifft_axis1.process(row_slice);
        }
    }

    let norm_factor = 1.0 / ((nx * ny) as f64);
    spectrum.mapv_inplace(|value| value * norm_factor);

    spectrum
}

/// Computes the angular momentum operator \(\hat{L}_z\) acting on a wavefunction.
///
/// \[
/// \hat{L}_z \psi = -i (x \partial_y \psi - y \partial_x \psi)
/// \]
///
/// The spatial gradients are computed via FFT:
/// \(\partial_x \psi = \mathcal{F}^{-1}[i k_x \mathcal{F}[\psi]]\).
///
/// # Arguments
///
/// * `phi` - Wavefunction \(\psi\) with shape \((n_x, n_y)\).
/// * `x` - \(x\)-grid coordinates, length \(n_x\).
/// * `y` - \(y\)-grid coordinates, length \(n_y\).
/// * `kx` - Wave-vector array in \(x\) direction, length \(n_x\).
/// * `ky` - Wave-vector array in \(y\) direction, length \(n_y\).
///
/// # Returns
///
/// Array of shape \((n_x, n_y)\) containing \(\hat{L}_z \psi\).
pub fn angular_momentum_lz(
    phi: &Array2<Complex<f64>>,
    x: &Array1<f64>,
    y: &Array1<f64>,
    kx: &Array1<f64>,
    ky: &Array1<f64>,
) -> Array2<Complex<f64>> {
    let (nx, ny) = (phi.shape()[0], phi.shape()[1]);

    let mut planner = FftPlanner::new();
    let fft_axis1 = planner.plan_fft_forward(ny);
    let fft_axis0 = planner.plan_fft_forward(nx);
    let ifft_axis1 = planner.plan_fft_inverse(ny);
    let ifft_axis0 = planner.plan_fft_inverse(nx);

    // Forward FFT along axis 1 (fast axis / columns).
    let mut spectrum = phi.clone();
    for mut row in spectrum.rows_mut() {
        let row_slice = row
            .as_slice_mut()
            .expect("angular_momentum_lz requires a contiguous (standard-layout) array");
        fft_axis1.process(row_slice);
    }

    // Forward FFT along axis 0 (rows).
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

    let i = Complex::new(0.0, 1.0);

    // Broadcast kx -> (nx, 1) and ky -> (1, ny) for elementwise multiplication.
    let ikx = Array2::from_shape_fn((nx, 1), |(idx, _)| i * kx[idx]);
    let iky = Array2::from_shape_fn((1, ny), |(_, idx)| i * ky[idx]);

    // Gradient in k-space: d/dx -> i*kx, d/dy -> i*ky.
    let mut dpsi_dx_k = &spectrum * &ikx;
    let mut dpsi_dy_k = &spectrum * &iky;

    // Inverse FFT of dpsi/dx along axis 0.
    for col_idx in 0..ny {
        for (row_idx, value) in column.iter_mut().enumerate() {
            *value = dpsi_dx_k[[row_idx, col_idx]];
        }
        ifft_axis0.process(&mut column);
        for (row_idx, value) in column.iter().enumerate() {
            dpsi_dx_k[[row_idx, col_idx]] = *value;
        }
    }
    // Inverse FFT of dpsi/dx along axis 1.
    for mut row in dpsi_dx_k.rows_mut() {
        let row_slice = row
            .as_slice_mut()
            .expect("angular_momentum_lz requires a contiguous (standard-layout) array");
        ifft_axis1.process(row_slice);
    }

    // Inverse FFT of dpsi/dy along axis 0.
    for col_idx in 0..ny {
        for (row_idx, value) in column.iter_mut().enumerate() {
            *value = dpsi_dy_k[[row_idx, col_idx]];
        }
        ifft_axis0.process(&mut column);
        for (row_idx, value) in column.iter().enumerate() {
            dpsi_dy_k[[row_idx, col_idx]] = *value;
        }
    }
    // Inverse FFT of dpsi/dy along axis 1.
    for mut row in dpsi_dy_k.rows_mut() {
        let row_slice = row
            .as_slice_mut()
            .expect("angular_momentum_lz requires a contiguous (standard-layout) array");
        ifft_axis1.process(row_slice);
    }

    let norm_factor = 1.0 / ((nx * ny) as f64);
    dpsi_dx_k.mapv_inplace(|val| val * norm_factor);
    dpsi_dy_k.mapv_inplace(|val| val * norm_factor);

    // Real-space grid for position operators, broadcast to (nx, ny).
    let x_grid = Array2::from_shape_fn((nx, 1), |(idx, _)| Complex::new(x[idx], 0.0));
    let y_grid = Array2::from_shape_fn((1, ny), |(_, idx)| Complex::new(y[idx], 0.0));

    // Lz psi = -i * (x * dpsi/dy - y * dpsi/dx)
    -i * (&x_grid * &dpsi_dy_k - &y_grid * &dpsi_dx_k)
}

/// Computes the right-hand side of the SGPE.
///
/// \[
/// i\hbar\frac{\partial\psi}{\partial t} = (1-i\gamma)\left[-\frac{\hbar^2}{2m}\nabla^2 + V(\mathbf{r}) + g|\psi|^2 - \mu - \Omega \hat{L}_z\right]\psi + \eta(\mathbf{r},t)
/// \]
///
/// The rotation term \(-\Omega \hat{L}_z\psi\) and the optional thermal-cloud
/// Hartree-Fock term \(2g \tilde{n}(\mathbf{r})\) are added inside the
/// \((1-i\gamma)\) factor.
pub fn sgpe(
    phi: &Array2<Complex<f64>>,
    potential: &Array2<Complex<f64>>,
    interaction_strength: &f64,
    condensate: &Condensate,
    k_sq: &Array2<f64>,
    omega_rotation: f64,
    lz_psi: &Array2<Complex<f64>>,
    thermal_cloud_density: Option<&Array2<f64>>,
) -> Array2<Complex<f64>> {
    let i = Complex::new(0.0, 1.0);

    let kinetic = kinetic_energy_fourier(phi, k_sq);
    let interaction = phi.mapv(|p| Complex::new(interaction_strength * p.norm_sqr(), 0.0));

    // Build effective potential including optional Hartree-Fock thermal cloud term.
    let effective_potential = match thermal_cloud_density {
        Some(n_tilde) => {
            let hf_term = (2.0 * interaction_strength * n_tilde).mapv(|v| Complex::new(v, 0.0));
            potential + hf_term
        }
        None => potential.clone(),
    };

    let rhs = (1.0 / i)
        * (Complex::new(1.0, 0.0) - i * condensate.gamma)
        * ((effective_potential + interaction - condensate.chemical_potential) * phi + kinetic
            - Complex::new(omega_rotation, 0.0) * lz_psi);

    if debug_io_enabled() {
        if let Err(e) = write_phi(&rhs, "./src/debug/rhs.txt") {
            eprintln!("Failed to write SGPE RHS debug output: {}", e);
        }
    }

    rhs
}

/// Generates Wiener noise for the stochastic term.
pub fn generate_wiener_noise(gridpoints: &(usize, usize)) -> Array2<f64> {
    // Create a random number generator
    let mut rng = rand::thread_rng();

    // Generate the Wiener noise matrix
    let wiener_noise = Array2::from_shape_fn(*gridpoints, |_| StandardNormal.sample(&mut rng));

    if debug_io_enabled() {
        if let Err(e) = write_real_2d(&wiener_noise, "./src/debug/wiener_noise.txt") {
            eprintln!("Failed to write Wiener noise debug output: {}", e);
        }
    }

    wiener_noise
}

/// Generates phase noise for the stochastic term.
pub fn generate_phase_noise(gridpoints: &(usize, usize)) -> Array2<f64> {
    // Create a random number generator
    let mut rng = rand::thread_rng();

    // Draw random numbers from a uniform distribution between 0 and 1
    let uniform_dist = rand::distributions::Uniform::new(0.0, 1.0);
    let phase_noise = Array2::from_shape_fn(*gridpoints, |_| uniform_dist.sample(&mut rng));

    if debug_io_enabled() {
        if let Err(e) = write_real_2d(&phase_noise, "./src/debug/phase_noise.txt") {
            eprintln!("Failed to write phase noise debug output: {}", e);
        }
    }

    phase_noise
}

/// Calculates the final noise term for the SGPE.
pub fn calculate_noise(
    noise_magnitude: f64,
    wiener_noise: &Array2<f64>,
    phase_noise: &Array2<f64>,
) -> Array2<Complex<f64>> {
    // Convert wiener_noise to Complex<f64>
    let wiener_noise_complex = wiener_noise.mapv(|x| Complex::new(x, 0.0));

    // Convert phase_noise to Complex<f64>
    let phase_noise_complex = phase_noise.mapv(|theta| Complex::new(0.0, 2.0 * PI * theta).exp());

    // Calculate the final noise array using the provided formula
    let noise = Complex::new(noise_magnitude, 0.0) * &wiener_noise_complex * &phase_noise_complex;

    if debug_io_enabled() {
        if let Err(e) = write_phi(&noise, "./src/noise.txt") {
            eprintln!("Failed to write noise debug output: {}", e);
        }
    }

    noise
}

/// Performs a single Runge-Kutta step for the SGPE.
///
/// The angular momentum operator \(\hat{L}_z\) is evaluated at each substage
/// using [`angular_momentum_lz`] and passed to [`sgpe`].
pub fn runge_kutta_step_2d(
    y: &Array2<Complex<f64>>,
    h: &f64,
    gridpoints: &(usize, usize),
    noise_magnitude: f64,
    interaction_strength: &f64,
    potential: &Array2<Complex<f64>>,
    condensate: &Condensate,
    k_sq: &Array2<f64>,
    omega_rotation: f64,
    kx: &Array1<f64>,
    ky: &Array1<f64>,
    x: &Array1<f64>,
    y_coords: &Array1<f64>,
    thermal_cloud_density: Option<&Array2<f64>>,
    projector: Option<&Projector>,
) -> Array2<Complex<f64>> {
    // Generate Wiener noise from a normal distribution
    let wiener_noise: Array2<f64> = generate_wiener_noise(gridpoints);

    // Generate phase noise from a uniform distribution
    let phase_noise: Array2<f64> = generate_phase_noise(gridpoints);

    // Calculate the final noise array
    let noise: Array2<Complex<f64>> = calculate_noise(noise_magnitude, &wiener_noise, &phase_noise);

    let lz_k1 = angular_momentum_lz(y, x, y_coords, kx, ky);
    let k1 = sgpe(
        y,
        potential,
        interaction_strength,
        condensate,
        k_sq,
        omega_rotation,
        &lz_k1,
        thermal_cloud_density,
    );

    let y2 = y + Complex::new(*h / 2.0, 0.0) * &k1;
    let lz_k2 = angular_momentum_lz(&y2, x, y_coords, kx, ky);
    let k2 = sgpe(
        &y2,
        potential,
        interaction_strength,
        condensate,
        k_sq,
        omega_rotation,
        &lz_k2,
        thermal_cloud_density,
    );

    let y3 = y + Complex::new(*h / 2.0, 0.0) * &k2;
    let lz_k3 = angular_momentum_lz(&y3, x, y_coords, kx, ky);
    let k3 = sgpe(
        &y3,
        potential,
        interaction_strength,
        condensate,
        k_sq,
        omega_rotation,
        &lz_k3,
        thermal_cloud_density,
    );

    let y4 = y + Complex::new(*h / 1.0, 0.0) * &k3;
    let lz_k4 = angular_momentum_lz(&y4, x, y_coords, kx, ky);
    let k4 = sgpe(
        &y4,
        potential,
        interaction_strength,
        condensate,
        k_sq,
        omega_rotation,
        &lz_k4,
        thermal_cloud_density,
    );

    let result =
        y + Complex::new(*h / 6.0, 0.0) * (k1 + Complex::new(2.0, 0.0) * (k2 + k3) + k4) + noise;

    match projector {
        Some(p) => p.apply(&result),
        None => result,
    }
}

/// Performs the full Runge-Kutta time evolution for the SGPE.
pub fn runge_kutta_2d(
    t0: f64,
    y0: Array2<Complex<f64>>,
    noise_magnitude: &f64,
    interaction_strength: &f64,
    simulation: &Simulation,
    trap: &Trap,
    condensate: &Condensate,
    x_pos: &Array1<f64>,
    y_pos: &Array1<f64>,
    k_sq: &Array2<f64>,
    omega_rotation: f64,
    kx: &Array1<f64>,
    ky: &Array1<f64>,
    thermal_cloud_density: Option<&Array2<f64>>,
    projector: Option<&Projector>,
    save_full_trajectory: bool,
    dir: &Path,
) -> Array2<Complex<f64>> {
    let mut t = t0;
    let mut y = y0;
    let mut norm: f64;

    let trajectory_dir = dir.join("trajectory");
    if save_full_trajectory {
        if let Err(e) = std::fs::create_dir_all(&trajectory_dir) {
            panic!(
                "Failed to create trajectory directory {}: {}",
                trajectory_dir.display(),
                e
            );
        }
    }

    let potential = calculate_potential(x_pos, y_pos, trap);
    let step_area = simulation.step_size.0 * simulation.step_size.1;

    let mut consecutive_small_changes = 0;
    let mut final_step = 0usize;

    let initial_norm = y.iter().map(|&c| c.norm_sqr()).sum::<f64>() * step_area;
    let mut previous_norm = initial_norm.max(1e-12);

    let mut time_series: Vec<(f64, f64)> =
        Vec::with_capacity(simulation.timesteps.max(0) as usize + 1);
    time_series.push((t, initial_norm));

    if save_full_trajectory {
        let initial_path = trajectory_dir.join("step_000000.txt");
        let initial_path_str = initial_path.display().to_string();
        write_phi(&y, &initial_path_str).expect("Failed to write initial state");
    }

    for i in 0..simulation.timesteps {
        y = runge_kutta_step_2d(
            &y,
            &simulation.timestep,
            &simulation.gridpoints,
            *noise_magnitude,
            interaction_strength,
            &potential,
            condensate,
            k_sq,
            omega_rotation,
            kx,
            ky,
            x_pos,
            y_pos,
            thermal_cloud_density,
            projector,
        );

        t += simulation.timestep;
        norm = y.iter().map(|&c| c.norm_sqr()).sum::<f64>() * step_area;
        final_step = (i + 1) as usize;

        time_series.push((t, norm));

        if save_full_trajectory {
            let filename = trajectory_dir.join(format!("step_{:06}.txt", final_step));
            let filename_str = filename.display().to_string();
            write_phi(&y, &filename_str).expect("Failed to write trajectory step");
        }

        let relative_difference = if previous_norm.abs() > f64::EPSILON {
            (norm - previous_norm).abs() / previous_norm
        } else {
            norm.abs()
        };

        if relative_difference < 1e-4 {
            consecutive_small_changes += 1;
        } else {
            consecutive_small_changes = 0;
        }

        if consecutive_small_changes >= 5 {
            break;
        }

        previous_norm = norm;
    }

    let final_state_path = dir.join("final_state.txt");
    let final_state_path_str = final_state_path.display().to_string();
    write_phi(&y, &final_state_path_str).expect("Failed to write final state");

    let atom_series_path = dir.join("atom_number.csv");
    if let Err(e) = write_time_series(&time_series, &atom_series_path) {
        eprintln!(
            "Failed to write atom number data to {}: {}",
            atom_series_path.display(),
            e
        );
    }

    let metadata = format!(
        "save_full_trajectory = {}\ncompleted_steps = {}\ntimestep = {:.8e}\ntotal_time = {:.8e}\n",
        save_full_trajectory,
        final_step,
        simulation.timestep,
        simulation.timestep * final_step as f64,
    );

    if let Err(e) = std::fs::write(dir.join("metadata.toml"), metadata) {
        eprintln!("Failed to write metadata for {}: {}", dir.display(), e);
    }

    y
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
            omega_rotation: 0.0,
            depth: None,
            ring_radius: None,
            trap_radius: None,
        }
    }

    fn test_x() -> Array1<f64> {
        Array1::linspace(-10.0, 10.0, 33)
    }

    fn test_y() -> Array1<f64> {
        Array1::linspace(-10.0, 10.0, 33)
    }

    /// Build k-space arrays for a uniform grid with spacing `dx`, `dy`.
    fn make_k_space(
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
        let kx = Array1::from_shape_fn(nx, |i| {
            let freq = i as f64 / (nx as f64 * dx);
            if i > nx / 2 {
                freq - 1.0 / dx
            } else {
                freq
            }
        }) * 2.0
            * std::f64::consts::PI;

        let ky = Array1::from_shape_fn(ny, |i| {
            let freq = i as f64 / (ny as f64 * dy);
            if i > ny / 2 {
                freq - 1.0 / dy
            } else {
                freq
            }
        }) * 2.0
            * std::f64::consts::PI;

        let kx_sq = kx.mapv(|v| v.powi(2));
        let ky_sq = ky.mapv(|v| v.powi(2));
        let k_sq = kx_sq
            .into_shape((nx, 1))
            .expect("kx_sq shape conversion failed")
            + ky_sq;

        (kx, ky, k_sq)
    }

    #[test]
    fn wiener_noise_correct_shape() {
        let gp = (64, 64);
        let noise = generate_wiener_noise(&gp);
        assert_eq!(noise.shape(), &[64, 64]);
    }

    #[test]
    fn wiener_noise_zero_mean() {
        let gp = (128, 128);
        let noise = generate_wiener_noise(&gp);
        let mean = noise.mean().unwrap();
        // Standard normal should have mean ≈ 0 with std err \(\sim 1/\sqrt{N}\)
        assert!(mean.abs() < 0.02, "mean = {} should be ~0", mean);
    }

    #[test]
    fn wiener_noise_unit_variance() {
        let gp = (128, 128);
        let noise = generate_wiener_noise(&gp);
        let mean = noise.mean().unwrap();
        let var = noise.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        assert!((var - 1.0).abs() < 0.05, "variance = {} should be ~1", var);
    }

    #[test]
    fn phase_noise_in_unit_interval() {
        let gp = (64, 64);
        let noise = generate_phase_noise(&gp);
        assert!(noise.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn calculate_noise_correct_shape() {
        let gp = (64, 64);
        let wiener = generate_wiener_noise(&gp);
        let phase = generate_phase_noise(&gp);
        let noise = calculate_noise(1.0, &wiener, &phase);
        assert_eq!(noise.shape(), &[64, 64]);
    }

    #[test]
    fn rk4_step_preserves_shape() {
        let gp = (33, 33);
        let y = Array2::from_elem(gp, Complex::new(0.1, 0.0));
        let trap = test_trap();
        let x = test_x();
        let y_coords = test_y();
        let potential = calculate_potential(&x, &y_coords, &trap);
        let condensate = Condensate {
            temperature: 0.5,
            gamma: 0.1,
            scattering_length: 100.0 * BOHR_RADIUS,
            chemical_potential: 1.0,
        };
        let kx = Array1::from_shape_fn(gp.0, |i| {
            let f = i as f64 / (gp.0 as f64);
            if i > gp.0 / 2 {
                f - 1.0
            } else {
                f
            }
        } * 2.0 * PI);
        let ky = Array1::from_shape_fn(gp.1, |i| {
            let f = i as f64 / (gp.1 as f64);
            if i > gp.1 / 2 {
                f - 1.0
            } else {
                f
            }
        } * 2.0 * PI);
        let kx_sq = kx.mapv(|v| v.powi(2));
        let ky_sq = ky.mapv(|v| v.powi(2));
        let k_sq = kx_sq.into_shape((gp.0, 1)).unwrap() + ky_sq;

        let result = runge_kutta_step_2d(
            &y,
            &0.001,
            &gp,
            0.01,
            &1.0,
            &potential,
            &condensate,
            &k_sq,
            0.0,
            &kx,
            &ky,
            &x,
            &y_coords,
            None,
            None,
        );

        assert_eq!(result.shape(), &[33, 33]);
    }

    // --- seed_initial_state tests ---

    #[test]
    fn seed_initial_state_shape() {
        let gp = (64, 128);
        // \(\gamma=1,\ \tilde{T}=0.5,\ \Delta t=1,\ \Delta x=\Delta y=1 \Rightarrow \sigma = 1\)
        let state = seed_initial_state(gp, 1.0, 0.5, 1.0, 1.0, 1.0);
        assert_eq!(state.shape(), &[64, 128]);
    }

    #[test]
    fn seed_initial_state_mean_zero() {
        let gp = (128, 128);
        let state = seed_initial_state(gp, 1.0, 0.5, 1.0, 1.0, 1.0);
        let mean = state.iter().map(|c| c.re).sum::<f64>() / (128.0 * 128.0);
        // Statistical test: on 128² samples the standard error is ~0.008.
        assert!(mean.abs() < 0.03, "real mean = {} should be ~0", mean);
    }

    #[test]
    fn seed_initial_state_variance() {
        let gp = (128, 128);
        let (gamma, temp, dt, dx, dy) = (1.0, 0.5, 1.0, 1.0, 1.0);
        let state = seed_initial_state(gp, gamma, temp, dt, dx, dy);
        let amplitude = noise_magnitude(gamma, temp, dt, dx, dy);
        let n_elements = (gp.0 * gp.1) as f64;
        let mean_sq = state.iter().map(|c| c.norm_sqr()).sum::<f64>() / n_elements;
        let expected = amplitude * amplitude;
        let rel_err = (mean_sq - expected).abs() / expected;
        assert!(
            rel_err < 0.1,
            "⟨|ψ|²⟩ = {:.4e}, expected {:.4e} (rel_err = {:.2e})",
            mean_sq,
            expected,
            rel_err
        );
    }

    // --- angular_momentum_lz tests ---

    #[test]
    fn lz_shape() {
        let nx = 32;
        let ny = 32;
        let dx = 10.0 / 31.0;
        let dy = 10.0 / 31.0;
        let x = Array1::linspace(-5.0, 5.0, nx);
        let y = Array1::linspace(-5.0, 5.0, ny);
        let (kx, ky, _) = make_k_space(nx, ny, dx, dy);

        let psi = Array2::from_shape_fn((nx, ny), |(i, j)| {
            Complex::new(f64::exp(-0.5 * (x[i] * x[i] + y[j] * y[j])), 0.0)
        });

        let lz = angular_momentum_lz(&psi, &x, &y, &kx, &ky);
        assert_eq!(lz.shape(), &[nx, ny]);
    }

    #[test]
    fn lz_gaussian() {
        // A real Gaussian centred at the origin is an s-wave state with
        // L_z eigenvalue 0. The numerical result should be near zero.
        let nx = 32;
        let ny = 32;
        let dx = 10.0 / 31.0;
        let dy = 10.0 / 31.0;
        let x = Array1::linspace(-5.0, 5.0, nx);
        let y = Array1::linspace(-5.0, 5.0, ny);
        let (kx, ky, _) = make_k_space(nx, ny, dx, dy);

        let psi = Array2::from_shape_fn((nx, ny), |(i, j)| {
            Complex::new(f64::exp(-0.5 * (x[i] * x[i] + y[j] * y[j])), 0.0)
        });

        let lz = angular_momentum_lz(&psi, &x, &y, &kx, &ky);
        let max_val = lz.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
        // FFT gradients on a 32² grid have O(1e-5) discretisation error.
        assert!(
            max_val < 2e-5,
            "L_z on real Gaussian should be near zero, max |Lz psi| = {:.2e}",
            max_val
        );
    }

    #[test]
    fn lz_vortex() {
        // \((x + iy)\,e^{-r^2/2}\) is an \(\hat{L}_z = 1\) eigenstate.
        // L_z psi / psi should be ≈ 1 everywhere psi is non-zero.
        let nx = 32;
        let ny = 32;
        let dx = 10.0 / 31.0;
        let dy = 10.0 / 31.0;
        let x = Array1::linspace(-5.0, 5.0, nx);
        let y = Array1::linspace(-5.0, 5.0, ny);
        let (kx, ky, _) = make_k_space(nx, ny, dx, dy);

        let psi = Array2::from_shape_fn((nx, ny), |(i, j)| {
            let r2 = x[i] * x[i] + y[j] * y[j];
            let vortex = Complex::new(x[i], y[j]);
            vortex * Complex::new(f64::exp(-0.5 * r2), 0.0)
        });

        let lz = angular_momentum_lz(&psi, &x, &y, &kx, &ky);

        // Check the eigenvalue at points where |psi| is large.
        let mut max_ratio_err = 0.0_f64;
        for i in 0..nx {
            for j in 0..ny {
                let p = psi[[i, j]];
                let l = lz[[i, j]];
                let norm_p = p.norm();
                if norm_p > 0.1 {
                    // L_z psi should equal psi for this eigenstate.
                    let err = (l - p).norm() / norm_p;
                    if err > max_ratio_err {
                        max_ratio_err = err;
                    }
                }
            }
        }
        // FFT gradients on a 32² grid have O(1e-5) discretisation error.
        assert!(
            max_ratio_err < 2e-5,
            "L_z eigenvalue error = {:.2e}, expected < 2e-5",
            max_ratio_err
        );
    }

    // --- sgpe tests ---

    #[test]
    fn sgpe_with_rotation() {
        // SGPE with omega_rotation != 0 should produce the correct shape.
        let gp = (32, 32);
        let nx = gp.0;
        let ny = gp.1;
        let dx = 10.0 / 31.0;
        let dy = 10.0 / 31.0;
        let x = Array1::linspace(-5.0, 5.0, nx);
        let y = Array1::linspace(-5.0, 5.0, ny);
        let (kx, ky, k_sq) = make_k_space(nx, ny, dx, dy);

        let psi = Array2::from_shape_fn(gp, |(i, j)| {
            Complex::new(f64::exp(-0.5 * (x[i] * x[i] + y[j] * y[j])), 0.0)
        });

        let trap = test_trap();
        let potential = calculate_potential(&x, &y, &trap);
        let condensate = Condensate {
            temperature: 0.0,
            gamma: 0.0,
            scattering_length: 100.0 * BOHR_RADIUS,
            chemical_potential: 0.0,
        };

        let lz_psi = angular_momentum_lz(&psi, &x, &y, &kx, &ky);
        let result = sgpe(
            &psi,
            &potential,
            &1.0,
            &condensate,
            &k_sq,
            1.0,
            &lz_psi,
            None,
        );

        assert_eq!(result.shape(), &[nx, ny]);
    }

    #[test]
    fn sgpe_thermal_cloud_gated() {
        // Passing Some(n_tilde) should change the result relative to None.
        let gp = (16, 16);
        let nx = gp.0;
        let ny = gp.1;
        let dx = 10.0 / 15.0;
        let dy = 10.0 / 15.0;
        let x = Array1::linspace(-5.0, 5.0, nx);
        let y = Array1::linspace(-5.0, 5.0, ny);
        let (kx, ky, k_sq) = make_k_space(nx, ny, dx, dy);

        let psi = Array2::from_shape_fn(gp, |(i, j)| {
            Complex::new(f64::exp(-0.5 * (x[i] * x[i] + y[j] * y[j])), 0.0)
        });

        let trap = test_trap();
        let potential = calculate_potential(&x, &y, &trap);
        let condensate = Condensate {
            temperature: 0.0,
            gamma: 0.0,
            scattering_length: 100.0 * BOHR_RADIUS,
            chemical_potential: 0.0,
        };

        let lz_psi = angular_momentum_lz(&psi, &x, &y, &kx, &ky);
        let n_tilde = Array2::from_elem(gp, 0.1);

        let result_none = sgpe(
            &psi,
            &potential,
            &1.0,
            &condensate,
            &k_sq,
            0.0,
            &lz_psi,
            None,
        );
        let result_some = sgpe(
            &psi,
            &potential,
            &1.0,
            &condensate,
            &k_sq,
            0.0,
            &lz_psi,
            Some(&n_tilde),
        );

        assert_eq!(result_none.shape(), &[nx, ny]);
        assert_eq!(result_some.shape(), &[nx, ny]);

        // The two results should differ (thermal cloud adds an extra potential).
        let max_diff = (&result_some - &result_none)
            .iter()
            .map(|c| c.norm())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 0.0,
            "sgpe with thermal cloud should differ from without"
        );
    }

    #[test]
    fn conservative_gpe_limit() {
        // With gamma = 0, noise_magnitude = 0, omega_rotation = 0,
        // and no thermal cloud, the SGPE reduces to the conservative GPE.
        // Norm should be conserved to high precision over a single RK4 step.
        let gp = (32, 32);
        let nx = gp.0;
        let ny = gp.1;
        let x = Array1::linspace(-5.0, 5.0, nx);
        let y_coords = Array1::linspace(-5.0, 5.0, ny);
        let dx = 10.0 / 31.0;
        let dy = 10.0 / 31.0;
        let (kx, ky, k_sq) = make_k_space(nx, ny, dx, dy);

        let trap = Trap {
            trap_type: TrapType::Harmonic,
            frequency_x: 2.0 * PI * 25.0,
            frequency_y: 2.0 * PI * 25.0,
            frequency_z: 2.0 * PI * 100.0,
            omega_rotation: 0.0,
            depth: None,
            ring_radius: None,
            trap_radius: None,
        };
        let potential = calculate_potential(&x, &y_coords, &trap);
        let condensate = Condensate {
            temperature: 0.0,
            gamma: 0.0,
            scattering_length: 100.0 * BOHR_RADIUS,
            chemical_potential: 0.0,
        };

        // Initial Gaussian state.
        let y_init = Array2::from_shape_fn(gp, |(i, j)| {
            Complex::new(
                f64::exp(-0.5 * (x[i] * x[i] + y_coords[j] * y_coords[j])),
                0.0,
            )
        });

        let step_area = dx * dy;
        let initial_norm = y_init.iter().map(|&c| c.norm_sqr()).sum::<f64>() * step_area;

        let result = runge_kutta_step_2d(
            &y_init,
            &1e-3,
            &gp,
            0.0,
            &1.0,
            &potential,
            &condensate,
            &k_sq,
            0.0,
            &kx,
            &ky,
            &x,
            &y_coords,
            None,
            None,
        );

        let final_norm = result.iter().map(|&c| c.norm_sqr()).sum::<f64>() * step_area;
        let norm_change = (final_norm - initial_norm).abs();
        assert!(
            norm_change < 1e-12,
            "norm change = {:.2e}, expected < 1e-12",
            norm_change
        );
    }
}
