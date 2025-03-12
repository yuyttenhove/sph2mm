# sph2mm

Add an AMR background to an existing SPH initial conditions file.

```
Usage: sph2mm [OPTIONS] <SPH> <MODIFIED>

Arguments:
  <SPH>       Name of ICs file with sph ICs
  <MODIFIED>  Name to store the modified ICs

Options:
  -r, --resolution <RESOLUTION>
          Resolution of background grid
  -f, --background-file <BACKGROUND_FILE>
          Read positions of background particles from a .hdf5 file (under /PartType0/Coordinates)
  -d, --density <DENSITY>
          Density of background particles [default: 0]
  -i, --internal-energy <INTERNAL_ENERGY>
          Internal energy of background particles [default: 0]
  -c, --center <CENTER> <CENTER> <CENTER>
          Center of the box
  -s, --smoothing-length-factor <SMOOTHING_LENGTH_FACTOR>
          Smoothing length factor. Smoothing lengths from the SPH ICs are multiplied by this factor before redistributing the hydrodynamical quantities
  -N, --number-of-neighbours <NUMBER_OF_NEIGHBOURS>
          Reinitialize the smoothing lengths such that each particle has <N> neighbours in its kernel
  -p, --prune <PRUNE>
          Background particles with more than <PRUNE> SPH neighbours will be pruned. Set to 0 to disable pruning [default: 0]
  -R, --relax <RELAX>
          Perform <R> iterations of Lloyd mesh relaxation before redistributing the hydrodynamical quantities [default: 0]
  -h, --help
          Print help
  -V, --version
          Print version
```