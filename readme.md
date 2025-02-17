# sph2mm

Add an AMR background to an existing SPH initial conditions file

```bash
Usage: sph2mm [OPTIONS] <SPH> <MODIFIED>

Arguments:
  <SPH>       Name of ICs file with sph ICs
  <MODIFIED>  Name to store the modified ICs

Options:
  -r, --resolution <RESOLUTION>            Resolution of background grid [default: 32]
  -d, --density <DENSITY>                  Density of background particles [default: 1]
  -i, --internal-energy <INTERNAL_ENERGY>  Internal energy of background particles [default: 1]
  -c, --center <CENTER> <CENTER> <CENTER>  Center of the box
  -R, --relax <RELAX>                      Perform <N> interations of mesh relaxation [default: 0]
  -h, --help                               Print help
  -V, --version                            Print version
```