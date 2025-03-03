use std::f64;

use clap::Parser;
use glam::DVec3;
use meshless_voronoi::integrals::VolumeCentroidIntegral;
use meshless_voronoi::Dimensionality;
use meshless_voronoi::VoronoiIntegrator;
use ndarray::Array1;
use ndarray::Array2;

mod octree;
mod sph;

use octree::Octree;
use rand::rngs::ThreadRng;
use rand::Rng;
use sph::CubicSpline;
use sph::SphInterpolator;
use sph::SphParticle;


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

fn read_particle_data(fname: &str, smoothing_length_factor: f64) -> Result<Vec<SphParticle>, hdf5::Error> {
    let file = hdf5::File::open(fname)?;

    // Read particle data
    let data = file.group("PartType0")?;
    let coordinates = data.dataset("Coordinates")?.read_raw::<f64>()?;
    let masses = data.dataset("Masses")?.read_raw::<f64>()?;
    let velocities = data.dataset("Velocities")?.read_raw::<f64>()?;
    let internal_energy = data.dataset("InternalEnergy")?.read_raw::<f64>()?;
    let mut smoothing_length = data.dataset("SmoothingLength")?.read_raw::<f64>()?;
    for hsml in smoothing_length.iter_mut() {
        *hsml *= smoothing_length_factor;
    }

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
    #[arg(short, long)]
    resolution: Option<u32>,

    /// Read positions of background particles from a .hdf5 file (under /PartType0/Coordinates)
    #[arg(short='f', long)]
    background_file: Option<String>,

    /// Density of background particles
    #[arg(short, long, default_value_t = 0.)]
    density: f64,

    /// Internal energy of background particles
    #[arg(short, long, default_value_t = 0.)]
    internal_energy: f64,

    /// Center of the box
    #[arg(short, long, num_args = 3, value_delimiter = ' ')]
    center: Option<Vec<f64>>,

    /// Smoothing length factor. Smoothing lengths from the SPH ICs are multiplied by this 
    /// factor before redistributing the hydrodynamical quantities.
    #[arg(short, long)]
    smoothing_length_factor: Option<f64>,

    /// Reinitialize the smoothing lengths such that each particle has <N> neighbours in its kernel.
    #[arg(short='N', long)]
    number_of_neighbours: Option<usize>,

    /// Perform <R> iterations of Lloyd mesh relaxation before redistributing the hydrodynamical quantities
    #[arg(short = 'R', long, default_value_t = 0)]
    relax: u8,
}

fn main() {
    let args = Args::parse();

    if args.smoothing_length_factor.is_some() && args.number_of_neighbours.is_some() {
        panic!("At most one of --smoothing-length-factor or --number-of-neighbours may be specified!");
    }

    let box_size = read_box_info(&args.sph).expect("Error while reading box info!");
    let box_center_shift = args.center.map(|v| DVec3::from_slice(&v));
    let mut sph_particles = read_particle_data(&args.sph, args.smoothing_length_factor.unwrap_or(1.)).expect("Error while reading particle data!");

    println!("Building tree...");
    let mut tree = match (args.resolution, args.background_file) {
        (Some(resolution), None) => Octree::init_from_bg_resolution(
            box_size,
            resolution,
            box_center_shift,
        ),
        (None, Some(file_name)) => Octree::init_from_bg_file(
            box_size,
            &file_name,
            box_center_shift,
        ).expect("Error reading background coordinates from file!"),
        _ => panic!("Need to specify exactly one of --resolution and --background-file!"),
    };
    for p in sph_particles.iter() {
        tree.insert(p.loc );
    }

    println!("Computing Voronoi cells and performing mesh relaxation...");
    let tree_leaves = tree.into_leaves();
    let mut rng = ThreadRng::default();
    let mut mm_generators = tree_leaves.iter().map(|l| {
        match l.data() {
            Some(loc) => loc,
            None => l.anchor() + 0.5 * l.width() + 0.01 * (DVec3::new(rng.random(), rng.random(), rng.random()) - 0.5),
        }
    }).collect::<Vec<_>>();
    let mut volumes_centroids = VoronoiIntegrator::build(&mm_generators, None, -box_center_shift.unwrap_or_default(), box_size, Dimensionality::ThreeD, false).compute_cell_integrals::<VolumeCentroidIntegral>();
    for _ in 0..args.relax {
        mm_generators = volumes_centroids.iter().map(|vc| vc.centroid).collect::<Vec<_>>();
        volumes_centroids = VoronoiIntegrator::build(&mm_generators, None, -box_center_shift.unwrap_or_default(), box_size, Dimensionality::ThreeD, false).compute_cell_integrals::<VolumeCentroidIntegral>();
    }
    let mm_volumes = volumes_centroids.iter().map(|vc| vc.volume).collect::<Vec<_>>();
    let mm_centroids = volumes_centroids.iter().map(|vc| vc.centroid).collect::<Vec<_>>();

    println!("Interpolating...");
    let interp = SphInterpolator::init(&mut sph_particles, args.number_of_neighbours, CubicSpline);
    let mm_particles = interp.interpolate(&mm_generators, &mm_volumes, &mm_centroids, args.density, args.internal_energy);

    println!("Writing modified ICs...");
    modify_ics(&mm_particles, &mm_volumes, &args.sph, &args.modified)
        .expect("Error while writing ICs");
}
