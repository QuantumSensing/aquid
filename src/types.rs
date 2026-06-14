//! Defines the types and structures used in the SGPE simulation.

use super::constants::*;

/// Specifies the type of trap used in the simulation.
#[derive(Debug, Clone, Copy)]
pub enum TrapType {
    /// The simplest confinement of atomic gases is in a harmonic trap.
    Harmonic,
    /// A ring-shaped trap.
    Toroidal,
    /// Deprecated in favour of using `Harmonic` which also permits highly anisotropic potentials.
    Cigar,
}

/// Required and optional trap parameters are specified in `Trap`.
#[derive(Debug, Clone)]
pub struct Trap {
    /// Trap geometry variant
    pub trap_type: TrapType,
    /// Radial trap frequency \(\omega_x\) [rad/s]
    pub frequency_x: f64,
    /// Radial trap frequency \(\omega_y\) [rad/s]
    pub frequency_y: f64,
    /// Vertical trap frequency \(\omega_z\) [rad/s]
    pub frequency_z: f64,
    /// Amplitude of the ring trap
    pub depth: Option<f64>,
    /// Radius of the ring
    pub ring_radius: Option<f64>,
    /// Width of the ring trap
    pub trap_radius: Option<f64>,
}

/// Represents an atomic species in the simulation.
#[derive(Debug, Clone)]
pub struct Species {
    /// Atomic mass [kg]
    pub atomic_mass: f64,
}

/// Defines the finite-temperature parameters and other properties of the condensate.
#[derive(Debug, Clone)]
pub struct Condensate {
    /// Dimensionless temperature \(\tilde{T} = k_B T / (\hbar \omega_x)\)
    pub temperature: f64,
    /// Dimensionless damping parameter \(\gamma\)
    pub gamma: f64,
    /// \(s\)-wave scattering length [m]
    pub scattering_length: f64,
    /// Dimensionless chemical potential \(\tilde{\mu}\)
    pub chemical_potential: f64,
}

/// Simulation-specific parameters for the SGPE simulation.
#[derive(Debug, Clone)]
pub struct Simulation {
    pub grid_size: f64,
    pub gridpoints: (usize, usize),
    pub step_size: (f64, f64),
    pub timesteps: isize,
    pub timestep: f64,
    pub runs: usize,
    pub noise_realisations: i64,
}

/// Dimensionless scaling factors derived from harmonic oscillator units.
///
/// The reference frequency is \(\omega_x\) (radial trap frequency).
/// Lengths are scaled by \(\ell_x = \sqrt{\hbar/(m\omega_x)}\),
/// time by \(1/\omega_x\), energy by \(\hbar\omega_x\).
#[derive(Debug, Clone)]
pub struct Scalings {
    /// \(\ell_x = \sqrt{\hbar/(m\omega_x)}\) — radial length unit [m]
    pub length_x: f64,
    /// \(\ell_y = \ell_x\) (isotropic in-plane scaling)
    pub length_y: f64,
    /// \(\ell_z = \sqrt{\hbar/(m\omega_z)}\) — vertical length unit [m]
    pub length_z: f64,
    /// \(1/\omega_x\) — time unit [s]
    pub time_unit: f64,
    /// \(k_B/(\hbar\omega_x)\) — converts K to dimensionless \(\tilde{T}\) [K⁻¹]
    pub temperature_scale: f64,
    /// \(1/(\hbar\omega_x)\) — converts J to dimensionless \(\tilde{\mu}\) [J⁻¹]
    pub chemical_potential_scale: f64,
}

impl Scalings {
    /// Construct dimensioning scaling factors from a species and trap.
    ///
    /// All quantities are referenced to the radial frequency \(\omega_x\).
    /// The vertical length scale \(\ell_z\) uses \(\omega_z\) from the trap.
    pub fn new(species: &Species, trap: &Trap) -> Self {
        let hbar = REDUCED_PLANCK_CONSTANT;
        let m = species.atomic_mass;
        let omega_x = trap.frequency_x;
        let omega_z = trap.frequency_z;

        let length_x = (hbar / (m * omega_x)).sqrt();
        let length_y = length_x;
        let length_z = (hbar / (m * omega_z)).sqrt();
        let time_unit = 1.0 / omega_x;
        let temperature_scale = BOLTZMANN_CONSTANT / (hbar * omega_x);
        let chemical_potential_scale = 1.0 / (hbar * omega_x);

        Self {
            length_x,
            length_y,
            length_z,
            time_unit,
            temperature_scale,
            chemical_potential_scale,
        }
    }

    /// Convert a physical temperature in Kelvin to the dimensionless
    /// \(\tilde{T} = k_B T / (\hbar \omega_x)\) used in the SGPE.
    pub fn dimensionless_temperature(&self, t_kelvin: f64) -> f64 {
        t_kelvin * self.temperature_scale
    }

    /// Convert a physical chemical potential in Joules to the dimensionless
    /// \(\tilde{\mu} = \mu / (\hbar \omega_x)\) used in the SGPE.
    pub fn dimensionless_chemical_potential(&self, mu_joules: f64) -> f64 {
        mu_joules * self.chemical_potential_scale
    }

    /// Convert a dimensionless length (in units of \(\ell_x\)) to physical metres.
    pub fn physical_length(&self, dim_length: f64) -> f64 {
        dim_length * self.length_x
    }
}

impl Condensate {
    /// Computes the dimensionless 2D interaction strength
    /// \(g_{\mathrm{2D}} = \sqrt{8\pi}\, a_s / \ell_z\).
    ///
    /// This matches the thesis formula derived from integrating out the
    /// tightly-confined \(z\)-direction.
    ///
    /// # Arguments
    ///
    /// * `oscillator_length_z` — the vertical harmonic oscillator length \(\ell_z\) [m]
    pub fn interaction_strength(&self, oscillator_length_z: f64) -> f64 {
        (8.0_f64 * PI).sqrt() * self.scattering_length / oscillator_length_z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trap with \(\omega_x/2\pi = 570\) Hz, \(\omega_z/2\pi = 300\) Hz
    /// (2013-like parameters for rubidium-87).
    fn trap_570_300() -> Trap {
        Trap {
            trap_type: TrapType::Harmonic,
            frequency_x: 2.0 * PI * 570.0,
            frequency_y: 2.0 * PI * 570.0,
            frequency_z: 2.0 * PI * 300.0,
            depth: None,
            ring_radius: None,
            trap_radius: None,
        }
    }

    /// Trap with \(\omega_x/2\pi = 520\) Hz, \(\omega_z/2\pi = 300\) Hz
    /// (2020-like parameters for rubidium-87).
    fn trap_520_300() -> Trap {
        Trap {
            trap_type: TrapType::Harmonic,
            frequency_x: 2.0 * PI * 520.0,
            frequency_y: 2.0 * PI * 520.0,
            frequency_z: 2.0 * PI * 300.0,
            depth: None,
            ring_radius: None,
            trap_radius: None,
        }
    }

    #[test]
    fn species_rb87_mass() {
        let s = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let expected = 86.9092 * ATOMIC_MASS_UNIT;
        let rel_err = (s.atomic_mass - expected).abs() / expected;
        assert!(
            rel_err < 1e-14,
            "RB87 mass mismatch: rel_err = {:.2e}",
            rel_err
        );
    }

    #[test]
    fn species_rb85_mass() {
        let s = Species {
            atomic_mass: 84.9117 * ATOMIC_MASS_UNIT,
        };
        let expected = 84.9117 * ATOMIC_MASS_UNIT;
        let rel_err = (s.atomic_mass - expected).abs() / expected;
        assert!(
            rel_err < 1e-14,
            "RB85 mass mismatch: rel_err = {:.2e}",
            rel_err
        );
    }

    #[test]
    fn length_z_is_0_62_um() {
        // For rubidium-87 with omega_z/2pi = 300 Hz, l_z ~ 0.62 um (within 2%)
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        let expected = 0.62e-6;
        let rel_err = (s.length_z - expected).abs() / expected;
        assert!(
            rel_err < 0.02,
            "l_z = {:.3e} m, expected ~{:.3e} m (rel_err = {:.2e})",
            s.length_z,
            expected,
            rel_err
        );
    }

    #[test]
    fn length_x_is_0_45_um_570hz() {
        // For rubidium-87 with omega_x/2pi = 570 Hz, l_x ~ 0.45 um (within 2%)
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        let expected = 0.45e-6;
        let rel_err = (s.length_x - expected).abs() / expected;
        assert!(
            rel_err < 0.02,
            "l_x = {:.3e} m, expected ~{:.3e} m (rel_err = {:.2e})",
            s.length_x,
            expected,
            rel_err
        );
    }

    #[test]
    fn length_x_is_0_47_um_520hz() {
        // For rubidium-87 with omega_x/2pi = 520 Hz, l_x ~ 0.47 um (within 2%)
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_520_300();
        let s = Scalings::new(&species, &trap);
        let expected = 0.47e-6;
        let rel_err = (s.length_x - expected).abs() / expected;
        assert!(
            rel_err < 0.02,
            "l_x = {:.3e} m, expected ~{:.3e} m (rel_err = {:.2e})",
            s.length_x,
            expected,
            rel_err
        );
    }

    #[test]
    fn interaction_strength_g_2d_is_0_042() {
        // For rubidium-87 with omega_z/2pi = 300 Hz, a_s = 100 a_0:
        // g_2D ~ 0.042 (within 2%)
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        let condensate = Condensate {
            temperature: 0.0,
            gamma: 0.1,
            scattering_length: 100.0 * BOHR_RADIUS,
            chemical_potential: 1.0,
        };
        let g = condensate.interaction_strength(s.length_z);
        let expected = 0.042;
        let rel_err = (g - expected).abs() / expected;
        assert!(
            rel_err < 0.02,
            "g_2D = {:.4e}, expected ~{:.4e} (rel_err = {:.2e})",
            g,
            expected,
            rel_err
        );
    }

    #[test]
    fn temperature_scaling_27_nk() {
        // At omega_x/2pi = 570 Hz, hbar*omega_x/k_B ~ 27 nK.
        // So T_tilde = 1 when T = 27 nK.
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        let t_dim = s.dimensionless_temperature(27.0e-9);
        let rel_err = (t_dim - 1.0).abs();
        assert!(
            rel_err < 0.05,
            "T_tilde(27 nK) = {:.4e}, expected ~1.0 (err = {:.2e})",
            t_dim,
            rel_err
        );
    }

    #[test]
    fn chemical_potential_30_nk() {
        // At omega_x/2pi = 570 Hz, hbar*omega_x ~ 3.77e-32 J.
        // mu = 30 nK -> mu_tilde ~ 1.1.
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        let mu_joules = 30.0e-9 * BOLTZMANN_CONSTANT;
        let mu_dim = s.dimensionless_chemical_potential(mu_joules);
        let expected = 1.1;
        let rel_err = (mu_dim - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "mu_dim = {:.4e}, expected ~{:.4e} (rel_err = {:.2e})",
            mu_dim,
            expected,
            rel_err
        );
    }

    #[test]
    fn physical_length_round_trip() {
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        let dim = 10.0;
        let phys = s.physical_length(dim);
        let expected = dim * s.length_x;
        assert!((phys - expected).abs() < 1e-20);
    }

    #[test]
    fn time_unit_is_inverse_frequency() {
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        let expected = 1.0 / trap.frequency_x;
        assert!((s.time_unit - expected).abs() < 1e-20);
    }

    #[test]
    fn length_y_equals_length_x() {
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        assert!(
            (s.length_y - s.length_x).abs() < 1e-20,
            "length_y ({:.4e}) != length_x ({:.4e})",
            s.length_y,
            s.length_x
        );
    }

    #[test]
    fn scalings_are_positive() {
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        assert!(s.temperature_scale > 0.0);
        assert!(s.chemical_potential_scale > 0.0);
        assert!(s.time_unit > 0.0);
    }

    #[test]
    fn different_traps_give_different_scalings() {
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let s1 = Scalings::new(&species, &trap_570_300());
        let s2 = Scalings::new(&species, &trap_520_300());
        assert!(
            (s1.length_x - s2.length_x).abs() > 1e-20,
            "different traps should give different length_x"
        );
        assert!(
            (s1.time_unit - s2.time_unit).abs() > 1e-20,
            "different traps should give different time_unit"
        );
    }

    #[test]
    fn dimensionless_temperature_inverse_of_scale() {
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        // T = 1/temperature_scale (i.e. T = hbar*omega_x/k_B) should give dimensionless T = 1.0
        let t_kelvin = 1.0 / s.temperature_scale;
        let t_dim = s.dimensionless_temperature(t_kelvin);
        let err = (t_dim - 1.0).abs();
        assert!(
            err < 1e-10,
            "dimensionless_temperature(1/scale) = {}, expected 1.0",
            t_dim
        );
    }

    #[test]
    fn dimensionless_chemical_potential_inverse_of_scale() {
        let species = Species {
            atomic_mass: 86.9092 * ATOMIC_MASS_UNIT,
        };
        let trap = trap_570_300();
        let s = Scalings::new(&species, &trap);
        // mu = 1/chemical_potential_scale (i.e. mu = hbar*omega_x) should give dimensionless mu = 1.0
        let mu_joules = 1.0 / s.chemical_potential_scale;
        let mu_dim = s.dimensionless_chemical_potential(mu_joules);
        let err = (mu_dim - 1.0).abs();
        assert!(
            err < 1e-10,
            "dimensionless_chemical_potential(1/scale) = {}, expected 1.0",
            mu_dim
        );
    }
}
