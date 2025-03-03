use std::f64;

use glam::DVec3;
use indicatif::{ProgressIterator, ProgressStyle};
use rstar::{primitives::GeomWithData, PointDistance, RTree};

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

#[derive(Clone, Copy, Default, Debug)]
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

type LocId = GeomWithData<[f64;3], usize>;


pub struct SphInterpolator<'a, K: SphKernel> {
    particles: &'a [SphParticle],
    kernel: K,
}

#[derive(Clone, Copy, Default)]
struct Conserved {
    mass: f64,
    momentum: DVec3,
    energy: f64,
}

impl<'a, K: SphKernel> SphInterpolator<'a, K> {
    pub fn init(particles: &'a mut [SphParticle], nngb: Option<usize>, kernel: K) -> Self {
        let rtree = RTree::bulk_load(particles.iter().enumerate().map(|(i, p)| LocId::new(p.loc.to_array(), i)).collect());
        if let Some(nngb) = nngb {
            // For each particle set the smoothing length based on its nearest neighbours
            for part in particles.iter_mut() {
                let query_loc = part.loc.to_array();
                let ngb = rtree.nearest_neighbor_iter(&query_loc).take(nngb + 1).last().expect(&format!("There must be at least {:} particles!", nngb + 1));
                part.smoothing_length = 0.5 * ngb.distance_2(&query_loc).sqrt()
            }
        }
        Self {
            particles,
            kernel,
        }
    }

    /// - Loop over SPH particles and assign weights = Sum kernel(r, sph->h) * cell.volume over the new cells in that sph particle's kernel tot the sph particles
    /// - Loop over new cells and add fraction kernel(r, sph->h) * cell.volume / sph->weight of conserved quantities of neighbouring sph particles to that cell
    /// - Compute mass/velocity/internal energy for each cell from conserved quantities.
    pub fn interpolate(&self, generators: &[DVec3], volumes: &[f64], centroids: &[DVec3], background_density: f64, background_internal_energy: f64) -> Vec<SphParticle> {
        
        let rtree = RTree::bulk_load(centroids.iter().enumerate().map(|(i, c)| LocId::new(c.to_array(), i)).collect());
        let mut sph_weights = vec![0.; self.particles.len()];
        for (p, w) in self.particles.iter().progress_with_style(ProgressStyle::with_template(" -> Computing weights: {bar} {pos:>7}/{len:7} {elapsed_precise}").unwrap()).zip(sph_weights.iter_mut()) {
            let h2 = p.smoothing_length * p.smoothing_length;
            let query_point = p.loc.to_array();
            for loc_id in rtree.nearest_neighbor_iter(&query_point) {
                let d2 = loc_id.distance_2(&query_point);
                if d2 > 4. * h2 { break; }
                let wk = self.kernel.eval(d2.sqrt(), p.smoothing_length);
                *w += wk * volumes[loc_id.data];
            }
        } 
        println!(" -> Computing weights: Done!");
        println!(" -> Total mass before redistributing: {:}", self.particles.iter().map(|p| p.mass).sum::<f64>());

        let mut total_mass_after = 0.;
        let mut total_mass_bg = 0.;
        let mut npart_bg = 0;
        let n_part = generators.len();
        let mut conserved = vec![Conserved::default(); n_part];
        let mut n_hit = vec![0; n_part];
        for (p, &w_sum) in self.particles.iter().progress_with_style(ProgressStyle::with_template(" -> Redistributing quantities: {bar} {pos:>7}/{len:7} {elapsed_precise}").unwrap()).zip(sph_weights.iter()) {
            let h2 = p.smoothing_length * p.smoothing_length;
            let query_point = p.loc.to_array();
            for loc_id in rtree.nearest_neighbor_iter(&query_point) {
                let d2 = loc_id.distance_2(&query_point);
                if d2 > 4. * h2 { break; }
                let id = loc_id.data;
                n_hit[id] += 1;
                let wk = self.kernel.eval(d2.sqrt(), p.smoothing_length);
                let weight = wk * volumes[id] / w_sum;
                conserved[id].mass += weight * p.mass;
                conserved[id].momentum += weight * p.momentum();
                conserved[id].energy += weight * p.energy();
            }
        } 
        let particles = conserved.iter().zip(n_hit.iter().zip(generators.iter().zip(volumes.iter()))).map(|(cons, (&n, (x, vol)))| {

            let hsml_from_vol = 6. / 4. * f64::consts::FRAC_1_PI * vol.cbrt();
            if n > 0 {
                assert!(cons.mass > 0.);
                total_mass_after += cons.mass;
                let m_inv = 1. / cons.mass;
                SphParticle::new(*x, cons.momentum * m_inv, cons.mass, cons.energy * m_inv, hsml_from_vol)
            } else {
                let part = SphParticle::init_background(*x, background_density, background_internal_energy, *vol);
                total_mass_bg += part.mass;
                npart_bg += 1;
                part
            }
        }).collect::<Vec<_>>();
        println!(" -> Total mass after redistributing: {:}", total_mass_after);
        println!(" -> Total mass in background: {:}", total_mass_bg);
        println!(" -> Number of background particles / Total: {:} / {:}", npart_bg, particles.len());
        particles
    }
}