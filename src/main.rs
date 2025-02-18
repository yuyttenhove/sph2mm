use std::f64;

use clap::Parser;
use glam::DVec3;
use ndarray::Array1;
use ndarray::Array2;
use octree::TreeNodeLeafData;

mod octree;
use octree::Octree;

#[derive(Clone, Copy)]
struct SphData {
    id: usize,
    loc: DVec3,
}

impl TreeNodeLeafData for SphData {
    fn loc(&self) -> DVec3 {
        self.loc
    }
}

#[derive(Clone)]
struct SphParticle {
    loc: DVec3,
    velocity: DVec3,
    mass: f64,
    internal_energy: f64,
    smoothing_length: f64,
}

impl SphParticle {
    fn new(
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

    fn init_background(loc: DVec3, density: f64, internal_energy: f64, volume: f64) -> Self {
        Self::new(
            loc,
            DVec3::ZERO,
            density * volume,
            internal_energy,
            3. / 4. * f64::consts::FRAC_1_PI * volume.cbrt(),
        )
    }
}

fn read_box_info(fname: &str) -> Result<DVec3, hdf5::Error> {
    let file = hdf5::File::open(fname)?;

    // Read header
    let header = file.group("Header")?;
    let attributes = file.attr_names()?;
    if attributes.contains(&"Dimension".to_string()) {
        let dimension = header.attr("Dimension")?.read_raw::<usize>()?[0];
        assert_eq!(dimension, 3, "Only 3D ICs are supported now!");
    }
    let box_size = header.attr("BoxSize")?.read_raw::<f64>()?;
    let box_size = if box_size.len() == 1 {
        DVec3::splat(box_size[0])
    } else {
        DVec3::from_slice(&box_size)
    };
    Ok(box_size)
}

fn read_particle_data(fname: &str) -> Result<Vec<SphParticle>, hdf5::Error> {
    let file = hdf5::File::open(fname)?;

    // Read particle data
    let data = file.group("PartType0")?;
    let coordinates = data.dataset("Coordinates")?.read_raw::<f64>()?;
    let masses = data.dataset("Masses")?.read_raw::<f64>()?;
    let velocities = data.dataset("Velocities")?.read_raw::<f64>()?;
    let internal_energy = data.dataset("InternalEnergy")?.read_raw::<f64>()?;
    let smoothing_length = data.dataset("SmoothingLength")?.read_raw::<f64>()?;

    let coordinates = coordinates
        .chunks(3)
        .map(DVec3::from_slice)
        .collect::<Vec<_>>();
    let velocities = velocities
        .chunks(3)
        .map(DVec3::from_slice)
        .collect::<Vec<_>>();

    let particles = (0..masses.len())
        .map(|i| {
            SphParticle::new(
                coordinates[i],
                velocities[i],
                masses[i],
                internal_energy[i],
                smoothing_length[i],
            )
        })
        .collect();

    Ok(particles)
}

fn get_part_arrays(
    parts: &[SphParticle],
) -> (
    Array2<f64>,
    Array1<f32>,
    Array2<f32>,
    Array1<f32>,
    Array1<f32>,
) {
    let num_part = parts.len();
    let coordinates = Array2::from_shape_vec(
        (num_part, 3),
        parts
            .iter()
            .map(|p| p.loc.to_array().into_iter())
            .flatten()
            .collect(),
    )
    .expect("unable to create array");
    let masses = Array1::from_vec(parts.iter().map(|p| p.mass as f32).collect());
    let velocities = Array2::from_shape_vec(
        (num_part, 3),
        parts
            .iter()
            .map(|p| p.velocity.as_vec3().to_array().into_iter())
            .flatten()
            .collect(),
    )
    .expect("unable to create array");
    let internal_energy =
        Array1::from_vec(parts.iter().map(|p| p.internal_energy as f32).collect());
    let smoothing_length =
        Array1::from_vec(parts.iter().map(|p| p.smoothing_length as f32).collect());

    (
        coordinates,
        masses,
        velocities,
        internal_energy,
        smoothing_length,
    )
}

fn modify_ics(
    parts: &[SphParticle],
    volumes: &[f64],
    fname_in: &str,
    fname_out: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::copy(fname_in, fname_out)?;
    let file_out = hdf5::File::open_rw(fname_out)?;

    // Modify header
    let header = file_out.group("Header")?;
    let mut num_part_this_file = header.attr("NumPart_ThisFile")?.read_raw::<usize>()?;
    let mut num_part_total = header.attr("NumPart_Total")?.read_raw::<usize>()?;
    assert!(
        num_part_this_file
            .iter()
            .zip(num_part_total.iter())
            .all(|(n1, n2)| n1 == n2),
        "Only single file ICs are supported!",
    );
    let num_part = parts.len();
    num_part_this_file[0] = num_part;
    num_part_total[0] = num_part;
    header
        .attr("NumPart_ThisFile")?
        .as_writer()
        .write_raw(&num_part_this_file)?;
    header
        .attr("NumPart_Total")?
        .as_writer()
        .write_raw(&num_part_total)?;

    // Unlink (delete) hydro particle data
    file_out.unlink("PartType0")?;

    // Write new hdyro particle data
    let (coordinates, masses, velocities, internal_energy, smoothing_length) =
        get_part_arrays(parts);
    let part_data = file_out.create_group("PartType0")?;
    part_data
        .new_dataset_builder()
        .with_data(coordinates.view())
        .create("Coordinates")?;
    part_data
        .new_dataset_builder()
        .with_data(masses.view())
        .create("Masses")?;
    part_data
        .new_dataset_builder()
        .with_data(velocities.view())
        .create("Velocities")?;
    part_data
        .new_dataset_builder()
        .with_data(internal_energy.view())
        .create("InternalEnergy")?;
    part_data
        .new_dataset_builder()
        .with_data(smoothing_length.view())
        .create("SmoothingLength")?;
    part_data
        .new_dataset_builder()
        .with_data(&(0..parts.len()).collect::<Vec<_>>())
        .create("ParticleIDs")?;
    part_data
        .new_dataset_builder()
        .with_data(volumes)
        .create("Volumes")?;

    Ok(())
}

/// Add an AMR background to an existing SPH initial conditions file
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Name of ICs file with sph ICs
    sph: String,

    /// Name to store the modified ICs
    modified: String,

    /// Resolution of background grid
    #[arg(short, long, default_value_t = 32)]
    resolution: u32,

    /// Density of background particles
    #[arg(short, long, default_value_t = 1.)]
    density: f64,

    /// Internal energy of background particles
    #[arg(short, long, default_value_t = 1.)]
    internal_energy: f64,

    /// Center of the box
    #[arg(short, long, num_args = 3, value_delimiter = ' ')]
    center: Option<Vec<f64>>,

    /// Perform <N> interations of mesh relaxation
    #[arg(short = 'R', long, default_value_t = 0)]
    relax: u8,
}

fn main() {
    let args = Args::parse();

    if args.relax > 0 {
        unimplemented!("Mesh relaxation is not supported yet!")
    }

    let box_size = read_box_info(&args.sph).expect("Error while reading box info!");
    let sph_particles = read_particle_data(&args.sph).expect("Error while reading particle data!");

    let mut tree = Octree::init(
        box_size,
        args.resolution,
        args.center.map(|v| DVec3::from_slice(&v)),
    );
    for (id, p) in sph_particles.iter().enumerate() {
        tree.insert(SphData { id, loc: p.loc });
    }

    let mut volumes = Vec::with_capacity(sph_particles.len());
    let mm_particles = tree
        .into_leaves()
        .iter()
        .map(|leaf| match leaf.data() {
            Some(data) => {
                volumes.push(
                    4. / 3. * f64::consts::PI * sph_particles[data.id].smoothing_length.powi(3),
                );
                sph_particles[data.id].clone()
            }
            None => {
                let loc = leaf.anchor() + 0.5 * leaf.width();
                let volume = leaf.width().element_product();
                volumes.push(volume);
                SphParticle::init_background(loc, args.density, args.internal_energy, volume)
            }
        })
        .collect::<Vec<_>>();

    modify_ics(&mm_particles, &volumes, &args.sph, &args.modified)
        .expect("Error while writing ICs");
}
