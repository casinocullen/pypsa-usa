# PyPSA-USA

**NOTE: This model is under active development. The western interconnection is stable, however you are likely to find bugs in the workflow as we continue to develop the model. Please file github issues or email ktehranchi@stanford.edu for support**

Model [Documentation](https://pypsa-usa.readthedocs.io/en/latest/)

PyPSA-USA is an open-source power systems model of the bulk transmission systems in the United States. This workflow draws from the work of [pypsa-eur](https://pypsa-eur.readthedocs.io/en/latest/index.html) and [pypsa-meets-earth](https://pypsa-earth.readthedocs.io/en/latest/how_to_contribute.html) to build a highly configurable power systems model that can be used for capacity expansion modeling, production cost simulation, and power flow analysis. This model is currently under development, and is only stable under certain configurations detailed below.

The model draws data from:

- The [TAMU/BreakthroughEnergy](https://www.breakthroughenergy.org/) transmission network model. This model has 82,071 bus network, 41,083 substations, and 104,192 lines across the three interconnections.
- Powerplant Data can be drawn from three options: the Breakthrough Network, the public version of the WECC Anchor Data Set Production Cost Model, or the EIA860
- Historical load data from the EIA via the EIA930.
- Forecasted load data from the [public WECC ADS PCM](https://www.wecc.org/ReliabilityModeling/Pages/AnchorDataSet.aspx).
- Renewable time series based on ERA5, assembled using the atlite tool.
- Geographical potentials for wind and solar generators based on [land use](https://land.copernicus.eu/global/products/lc) and excluding [protected lands](https://www.protectedplanet.net/country/USA) are computed with the atlite library.

Example 500 Node Western Interconnection Network:
![pypsa-usa Base Network](https://github.com/PyPSA/pypsa-usa/blob/master/workflow/repo_data/network_500.jpg)


# Installation

For installation clone this repository and create the dedicated `conda` environment

```
conda env create -f environment.yaml
conda activate pypsa-usa
```

<!-- download submodules after cloning:

```
git submodule update --init
```

update submodule with:

```
git submodule update --remote
``` -->

## Wildcards:
For more detailed definitions of wildcards, please reference [pypsa-eur](https://pypsa-eur.readthedocs.io/en/latest/wildcards.html). Not all wildcards implemented are available for pypsa-usa.

# Execution 
To execute the workflow, go into the `workflow` directory and execute `snakemake` from your terminal. 

```bash
snakemake -j6
```

where 6 indicates the number of used cores, you may change it to your preferred number. This will run the workflow defined in the `Snakefile`.

Note: The `build_renewable_profiles` rule will take ~10-15 minutes to run the first time you run the workflow. After that, changing the number of clusters, load, or generator configurations will not require rebuilding the renewable profiles. Changes to `renewables` configuration will cause re-run of `build_renewable_profiles`.


# Contributing

We welcome your contributions to this project. Please see the [contributions](https://pypsa-usa.readthedocs.io/en/latest/contributing.html) guide in our readthedocs page for more information. Please do not hesitate to reachout to ktehranchi@stanford.edu with specific questions, requests, or feature ideas.

<!-- # Scope -->

# License

The project is licensed under MIT License.
