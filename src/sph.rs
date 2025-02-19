use std::f64;

use glam::DVec3;
use indicatif::{ProgressIterator, ProgressStyle};
use rstar::{primitives::GeomWithData, DefaultParams, PointDistance, RTree};

pub trait SphKernel {
    fn eval(&self, r: f64, h: f64) -> f64;
}

pub struct CubicSpline;

impl SphKernel for CubicSpline {
    fn eval(&self, r: f64, h: f64) -> f64 {
        let q = r / h;
        let sigma = f64::consts::FRAC_1_PI / h.cbrt();
        if q > 2. {
            0.
        } else if q > 1. {
            sigma * 0.25 * (2. - q).powi(3)
        } else {
            debug_assert!(q >= 0.);
            sigma * (1. - 1.5 * q * q * (1. - 0.5 * q))
        }
    }
}

#[derive(Clone)]
pub struct SphParticle {
    pub loc: DVec3,
    pub velocity: DVec3,
    pub mass: f64,
    pub internal_energy: f64,
    pub smoothing_length: f64,
}

impl SphParticle {
    pub fn new(
        loc: DVec3,
        velocity: DVec3,
        mass: f64,
        internal_energy: f64,
        smoothing_length: f64,
    ) -> Self {
        Self {
            loc,
            velocity,
            mass,
            internal_energy,
            smoothing_length,
        }
    }

    pub fn init_background(loc: DVec3, density: f64, internal_energy: f64, volume: f64) -> Self {
        Self::new(
            loc,
            DVec3::ZERO,
            density * volume,
            internal_energy,
            3. / 4. * f64::consts::FRAC_1_PI * volume.cbrt(),
        )
    }

    pub fn momentum(&self) -> DVec3 {
        self.velocity * self.mass
    }

    pub fn energy(&self) -> f64 {
        self.internal_energy * self.mass
    }
}

type SphLocId = GeomWithData<[f64;3], usize>;


pub struct SphInterpolator<'a, K: SphKernel> {
    particles: &'a [SphParticle],
    rtree: RTree<SphLocId, DefaultParams>,
    kernel: K,
    h_max: f64
}

impl<'a, K: SphKernel> SphInterpolator<'a, K> {
    pub fn init(particles: &'a mut [SphParticle], nngb: Option<usize>, kernel: K) -> Self {
        let rtree = RTree::bulk_load(particles.iter().enumerate().map(|(i, p)| SphLocId::new(p.loc.to_array(), i)).collect());
        if let Some(nngb) = nngb {
            // For each particle set the smoothing length based on its nearest neighbours
            for part in particles.iter_mut() {
                let query_loc = part.loc.to_array();
                let ngb = rtree.nearest_neighbor_iter(&query_loc).take(nngb + 1).last().expect(&format!("There must be at least {:} particles!", nngb + 1));
                part.smoothing_length = 0.5 * ngb.distance_2(&query_loc).sqrt()
            }
        }
        let h_max = particles.iter().map(|p| p.smoothing_length).max_by(|a, b| a.partial_cmp(b).expect("Smoothing lengths must not be NaN")).expect("At least one particle must be present!");
        println!("Hsml max: {:}", h_max);
        Self {
            particles,
            rtree,
            kernel,
            h_max,
        }
    }

    /// TODO:
    /// - Loop over SPH particles and assign weights = Sum kernel(r, sph->h) * cell.volume over the new cells in that sph particle's kernel tot the sph particles
    /// - Loop over new cells and add fraction kernel(r, sph->h) * cell.volume / sph->weight of conserved quantities of neighbouring sph particles to that cell
    /// - Compute mass/velocity/internal energy for each cell from conserved quantities.
    pub fn interpolate(&self, generators: &[DVec3], volumes: &[f64], centroids: &[DVec3], background_density: f64, background_internal_energy: f64) -> Vec<SphParticle> {
        
        let mut sph_weights = vec![0.; self.particles.len()];
        for (c, vol) in centroids.iter().progress_with_style(ProgressStyle::with_template(" -> Computing weights: {bar} {pos:>7}/{len:7} {elapsed_precise}").unwrap()).zip(volumes.iter()) {
            let query_point = c.to_array();
            for sph_loc_id in self.rtree.nearest_neighbor_iter(&query_point) {
                let d2 = sph_loc_id.distance_2(&query_point);
                if d2 > 4. * self.h_max * self.h_max {
                    break;
                }
                let part = &self.particles[sph_loc_id.data];
                let h2 = part.smoothing_length * part.smoothing_length;
                if d2 < 4. * h2 {
                    let wk = self.kernel.eval(d2.sqrt(), part.smoothing_length);
                    sph_weights[sph_loc_id.data] += wk * vol;
                }
            }
        }
        println!(" -> Computing weights: Done!");
        println!(" -> Total mass before redistributing: {:}", self.particles.iter().map(|p| p.mass).sum::<f64>());

        let mut total_mass_after = 0.;
        let mut total_mass_bg = 0.;
        let particles = generators.iter().progress_with_style(ProgressStyle::with_template(" -> Redistributing quantities: {bar} {pos:>7}/{len:7} {elapsed_precise}").unwrap()).zip(volumes.iter().zip(centroids.iter())).map(|(x, (vol, c))| {
            let query_point = c.to_array();
            let mut mass = 0.;
            let mut momentum = DVec3::ZERO;
            let mut energy = 0.;
            let mut n_hit = 0;

            for sph_loc_id in self.rtree.nearest_neighbor_iter(&query_point) {
                let d2 = sph_loc_id.distance_2(&query_point);
                if d2 > 4. * self.h_max * self.h_max {
                    break;
                }

                let part = &self.particles[sph_loc_id.data];
                let h2 = part.smoothing_length * part.smoothing_length;
                let sph_wsum = sph_weights[sph_loc_id.data];
                if d2 < 4. * h2 {
                    let wk = self.kernel.eval(d2.sqrt(), part.smoothing_length);
                    let weight = wk * vol / sph_wsum;
                    n_hit += 1;
                    mass += weight * part.mass;
                    momentum += weight * part.momentum();
                    energy += weight * part.energy();
                }
            }

            let hsml_from_vol = 6. / 4. * f64::consts::FRAC_1_PI * vol.cbrt();
            if n_hit > 0 {
                assert!(mass > 0.);
                total_mass_after += mass;
                let m_inv = 1. / mass;
                SphParticle::new(*x, momentum * m_inv, mass, energy * m_inv, hsml_from_vol)
            } else {
                let part = SphParticle::init_background(*x, background_density, background_internal_energy, *vol);
                total_mass_bg += part.mass;
                part
            }
        }).collect();
        println!(" -> Total mass after redistributing: {:}", total_mass_after);
        println!(" -> Total mass in background: {:}", total_mass_bg);
        particles
    }
}