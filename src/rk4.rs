//! Implements the Runge-Kutta fourth-order method for SGPE simulations.

use super::constants::*;
use super::types::*;
use super::utils::*;
use ndarray::{Array1, Array2};
use num_complex::Complex;
use rand_distr::{Distribution, StandardNormal};
use rustfft::FftPlanner;
use std::path::Path;
use std::sync::OnceLock;

/// Calculates the harmonic potential.
///
/// [V(x,y) = \frac{1}{2}m(\omega_x^2 x^2 + \omega_y^2 y^2)]
pub fn harmonic_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    // Calculate aspect ratios
    let aspect_ratio_x = trap.frequency_x / trap.frequency_z;
    let aspect_ratio_y = trap.frequency_y / trap.frequency_z;

    // Compute the potential using broadcasting
    let potential_x =
        0.5 * aspect_ratio_x.powi(2) * x.mapv(|x| x.powi(2)).into_shape((x.len(), 1)).unwrap();
    let potential_y =
        0.5 * aspect_ratio_y.powi(2) * y.mapv(|y| y.powi(2)).into_shape((1, y.len())).unwrap();

    // Combine x and y potentials
    let potential = potential_x + potential_y;

    // Convert the combined potential to Complex<f64>
    potential.mapv(|val| Complex::new(val, 0.0))
}

/// Calculates the toroidal potential.
///
/// [V(r) = V_0(1 - e^{-\frac{(r-R)^2}{2\sigma^2}})]
pub fn toroidal_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    // Unwrap the toroidal trap parameters
    let depth = trap.depth.expect("Depth is required for a toroidal trap");
    let ring_radius = trap
        .ring_radius
        .expect("Ring radius is required for a toroidal trap");
    let trap_radius = trap
        .trap_radius
        .expect("Trap radius is required for a toroidal trap");

    // Create a grid of x and y values
    let x_grid = x.broadcast((y.len(), x.len())).unwrap();
    let y_grid = y.broadcast((x.len(), y.len())).unwrap().reversed_axes();

    // Compute rho for each pair in the grid
    let rho = (&x_grid * &x_grid + &y_grid * &y_grid).mapv(f64::sqrt);

    // Compute the potential using the given equation
    let potential = depth
        * (1.0
            - (-1.0 / trap_radius.powi(2) * (&rho - ring_radius).mapv(|rho_r| rho_r.powi(2)))
                .mapv(f64::exp));

    // Convert the potential to Complex<f64>
    potential.mapv(|val| Complex::new(val, 0.0))
}

/// Selects and calculates the appropriate potential based on trap type.
pub fn calculate_potential(x: &Array1<f64>, y: &Array1<f64>, trap: &Trap) -> Array2<Complex<f64>> {
    match trap.trap_type {
        TrapType::Harmonic | TrapType::Cigar => harmonic_potential(x, y, trap),
        TrapType::Toroidal => toroidal_potential(x, y, trap),
    }
}

fn debug_io_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("SGPE_DEBUG_IO")
            .map(|value| value != "0" && !value.is_empty())
            .unwrap_or(false)
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

/// Computes the right-hand side of the SGPE.
///
/// [i\hbar\frac{\partial\psi}{\partial t} = (1-i\gamma)(H\psi - \mu\psi)]
pub fn sgpe(
    phi: &Array2<Complex<f64>>,
    potential: &Array2<Complex<f64>>,
    interaction_strength: &f64,
    condensate: &Condensate,
    k_sq: &Array2<f64>,
) -> Array2<Complex<f64>> {
    let i = Complex::new(0.0, 1.0);

    let kinetic = kinetic_energy_fourier(phi, k_sq);
    let interaction = phi.mapv(|p| interaction_strength * p.norm_sqr());

    let rhs = (1.0 / i)
        * (Complex::new(1.0, 0.0) - i * condensate.gamma)
        * ((potential.clone() + interaction - condensate.chemical_potential) * phi + kinetic);

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
pub fn runge_kutta_step_2d(
    y: &Array2<Complex<f64>>,
    h: &f64,
    gridpoints: &(usize, usize),
    noise_magnitude: f64,
    interaction_strength: &f64,
    potential: &Array2<Complex<f64>>,
    condensate: &Condensate,
    k_sq: &Array2<f64>,
) -> Array2<Complex<f64>> {
    // Generate Wiener noise from a normal distribution
    let wiener_noise: Array2<f64> = generate_wiener_noise(gridpoints);

    // Generate phase noise from a uniform distribution
    let phase_noise: Array2<f64> = generate_phase_noise(gridpoints);

    // Calculate the final noise array
    let noise: Array2<Complex<f64>> = calculate_noise(noise_magnitude, &wiener_noise, &phase_noise);

    let k1 = sgpe(y, &potential, &interaction_strength, &condensate, k_sq);
    let k2 = sgpe(
        &(y + Complex::new(h / 2.0, 0.0) * &k1),
        &potential,
        &interaction_strength,
        &condensate,
        k_sq,
    );
    let k3 = sgpe(
        &(y + Complex::new(h / 2.0, 0.0) * &k2),
        &potential,
        &interaction_strength,
        &condensate,
        k_sq,
    );
    let k4 = sgpe(
        &(y + Complex::new(h / 1.0, 0.0) * &k3),
        &potential,
        &interaction_strength,
        &condensate,
        k_sq,
    );

    y + Complex::new(h / 6.0, 0.0) * (k1 + Complex::new(2.0, 0.0) * (k2 + k3) + k4) + noise
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
        // Standard normal should have mean ≈ 0 with std err ~ 1/sqrt(N)
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
            if i > gp.0 / 2 { f - 1.0 } else { f }
        } * 2.0 * PI);
        let ky = Array1::from_shape_fn(gp.1, |i| {
            let f = i as f64 / (gp.1 as f64);
            if i > gp.1 / 2 { f - 1.0 } else { f }
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
        );

        assert_eq!(result.shape(), &[33, 33]);
    }
}
