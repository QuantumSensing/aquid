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
    pub trap_type: TrapType,
    pub frequency_x: f64,
    pub frequency_y: f64,
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
    pub atomic_mass: f64,
}

/// Defines the finite-temperature parameters and other properties of the condensate.
#[derive(Debug, Clone)]
pub struct Condensate {
    pub temperature: f64,
    pub gamma: f64,
    pub scattering_length: f64,
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

/// Dimensionless scalings for various quantities in the simulation.
#[derive(Debug, Clone)]
pub struct Scalings {
    pub temperature: f64,
    pub length_x: f64,
    pub length_y: f64,
    pub length_z: f64,
    pub time: f64,
    pub chemical_potential: f64,
}

impl Condensate {
    /// Calculates the interaction strength g = 4πℏ²a_s/m.
    pub fn interaction_strength(&self, oscillator_length_z: f64) -> f64 {
        (8.0_f64 * PI).sqrt() * self.scattering_length / oscillator_length_z
    }
}
