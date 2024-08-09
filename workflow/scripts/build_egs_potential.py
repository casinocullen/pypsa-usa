# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2023 @LukasFranken, The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
This rule extracts potential and cost for electricity generation through
enhanced geothermal systems.

For this, we use data from "From hot rock to useful energy..." by Aghahosseini, Breyer (2020)
'https://doi.org/10.1016/j.apenergy.2020.115769'
Note that we input data used here is not the same as in the paper, but was passed on by the authors.

The data provides a lon-lat gridded map of Europe (1° x 1°), with each grid cell assigned
a heat potential (in GWh) and a cost (in EUR/MW).

This scripts overlays that map with the network's regions, and builds a csv with CAPEX, OPEX and p_nom_max
"""


import logging
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
import pypsa
from _helpers import configure_logging
from add_electricity import (
    calculate_annuity,
    _add_missing_carriers_from_costs,
    add_nice_carrier_names,
    load_costs,
)

import xarray as xr
from shapely.geometry import Polygon

idx = pd.IndexSlice

logger = logging.getLogger(__name__)



########################################################################
################### NEW SECTION TO ADD GEOTHERMAL ######################
########################################################################
def clean_egs_sc(geo_egs_sc, capex_cap = 1e5): 
    geo_sc = gpd.read_file(geo_egs_sc).to_crs(4326).rename(columns={'name':'county'})
    
    geo_sc['capex_kw'] =(geo_sc['capex_kw']/5000).round()*5000
    geo_sc['fom_kw'] =(geo_sc['fom_kw']/500).round()*500
    geo_sc = geo_sc.loc[geo_sc['capex_kw'] <= capex_cap]
    
    return geo_sc

def join_pypsa_cluser(regions_gpd, geo_egs_sc): 
    geo_sc = clean_egs_sc(geo_egs_sc)

    region_onshore = gpd.read_file(regions_gpd)
    region_onshore_geo = gpd.sjoin(region_onshore, geo_sc, how="left").reset_index(drop=True)

    region_onshore_geo_grp = region_onshore_geo.groupby(['name', 'capex_kw', 'fom_kw'])['potential_mw'].agg('sum').reset_index()
    region_onshore_geo_grp['class'] = region_onshore_geo_grp.groupby(['name']).cumcount()+1
    region_onshore_geo_grp['class'] = "c" + region_onshore_geo_grp['class'].astype(str)
    region_onshore_geo_grp['carrier'] = 'egs'
    region_onshore_geo_grp['carrier_class'] = region_onshore_geo_grp[['carrier', 'class']].agg('_'.join, axis=1)

    region_onshore_geo_grp = region_onshore_geo_grp[['name', 'carrier', 'carrier_class','capex_kw', 'fom_kw', 'potential_mw']].rename(columns={'name':'bus', 'capex_kw':"capital_cost", 'fom_kw':"marginal_cost", 'potential_mw':"p_nom_max"}).sort_values(by=['bus', 'carrier'], ascending=True)
    region_onshore_geo_grp['capital_cost'] = region_onshore_geo_grp['capital_cost'] * 1e3
    region_onshore_geo_grp['marginal_cost'] = region_onshore_geo_grp['marginal_cost'] * 1e3
    region_onshore_geo_grp["fom_pct"] = region_onshore_geo_grp['marginal_cost'] / region_onshore_geo_grp['capital_cost']

    return region_onshore_geo_grp

def add_egs(n: pypsa.Network, regions_gpd, geo_sc, cost_reduction): 
    egs_class = join_pypsa_cluser(regions_gpd, geo_sc)
    egs_class["capital_cost"] = (
            (
                calculate_annuity(30, 0.055)
                + egs_class["fom_pct"]/100
            )
            * egs_class["capital_cost"]
            * n.snapshot_weightings.objective.sum() / 8760.0
        ) * (1-cost_reduction)
    

    egs_class['Generator'] = egs_class['bus'] + ' ' + egs_class['carrier_class']
    egs_class = egs_class.set_index('Generator')
    egs_class['p_nom_min'] = 0
    egs_class['p_nom'] = 0
    egs_class['efficiency'] = 1
    egs_class['weight'] = 1
    egs_class['control'] = 'PQ'
    egs_class['p_min_pu'] = 0
    egs_class['p_max_pu'] = 0.9
    egs_class['p_nom_opt'] = np.nan
  
    n.madd(
    "Generator",
    egs_class.index,
    suffix=" new",
    carrier= egs_class.carrier, 
    bus=egs_class.bus,
    p_nom_min=0,
    p_nom=0,
    p_nom_max=egs_class.p_nom_max,
    p_nom_extendable=True,
    ramp_limit_up=0.15,
    ramp_limit_down=0.15,
    efficiency=egs_class.efficiency,
    marginal_cost=0,
    capital_cost=egs_class.capital_cost,
    lifetime=30,
    p_min_pu = egs_class.p_min_pu,
    p_max_pu = egs_class.p_max_pu,
    )



def attach_geo_storageunits(n, costs, elec_opts, regions_gpd, geo_egs_sc):

    carriers = elec_opts["extendable_carriers"]["StorageUnit"]
    carriers = [k for k in carriers if 'geothermal' in k]
    _add_missing_carriers_from_costs(n, costs, carriers)

    buses_i = n.buses.index

    for carrier in carriers:
        if 'ht' in carrier: 
            geo_sc = clean_egs_sc(geo_egs_sc)

            region_onshore = gpd.read_file(regions_gpd)
            region_onshore_geo = gpd.sjoin(region_onshore, geo_sc, how="left").reset_index(drop=True)
            region_onshore_geo_ht = (region_onshore_geo
                                     .loc[(region_onshore_geo.depth <= 4000) & (region_onshore_geo.temp >= 150)]
                                     .groupby(['name'])['potential_mw']
                                     .agg('sum')
                                     .reset_index()
                                     .set_index('name', drop=False)
            )

            max_hours = int(carrier.split("hr_")[0])
            roundtrip_ef = 1.41421356237 # 1.41421356237^2 = 2

            n.madd(
                "StorageUnit",
                region_onshore_geo_ht.index,
                " " + carrier,
                bus=region_onshore_geo_ht.name,
                carrier=carrier,
                p_nom_max=region_onshore_geo_ht.potential_mw,
                p_nom_extendable=True,
                capital_cost=costs.at[carrier, "capital_cost"],
                marginal_cost=costs.at[carrier, "marginal_cost"],
                efficiency_store=roundtrip_ef,
                efficiency_dispatch=roundtrip_ef,
                max_hours=max_hours,
                cyclic_state_of_charge=True,
            )

        else: 
            max_hours = int(carrier.split("hr_")[0])
            roundtrip_ef = 0.83666002653 # 0.83666002653^2 = 0.75
            
            n.madd(
                "StorageUnit",
                buses_i,
                " " + carrier,
                bus=buses_i,
                carrier=carrier,
                p_nom_extendable=True,
                capital_cost=costs.at[carrier, "capital_cost"],
                marginal_cost=costs.at[carrier, "marginal_cost"],
                efficiency_store=roundtrip_ef,
                efficiency_dispatch=roundtrip_ef,
                max_hours=max_hours,
                cyclic_state_of_charge=True,
            )


def get_capacity_factors(network_regions_file, air_temperatures_file):
    """
    Performance of EGS is higher for lower temperatures, due to more efficient
    air cooling Data from Ricks et al.: The Role of Flexible Geothermal Power
    in Decarbonized Elec Systems.
    """

    # these values are taken from the paper's
    # Supplementary Figure 20 from https://zenodo.org/records/7093330
    # and relate deviations of the ambient temperature from the year-average
    # ambient temperature to EGS capacity factors.
    delta_T = [-15, -10, -5, 0, 5, 10, 15, 20]
    cf = [1.17, 1.13, 1.07, 1, 0.925, 0.84, 0.75, 0.65]

    x = np.linspace(-15, 20, 200)
    y = np.interp(x, delta_T, cf)

    upper_x = np.linspace(20, 25, 50)
    m_upper = (y[-1] - y[-2]) / (x[-1] - x[-2])
    upper_y = upper_x * m_upper - x[-1] * m_upper + y[-1]

    lower_x = np.linspace(-20, -15, 50)
    m_lower = (y[1] - y[0]) / (x[1] - x[0])
    lower_y = lower_x * m_lower - x[0] * m_lower + y[0]

    x = np.hstack((lower_x, x, upper_x))
    y = np.hstack((lower_y, y, upper_y))

    network_regions = gpd.read_file(network_regions_file).set_crs(epsg=4326)
    index = network_regions["name"]

    air_temp = xr.open_dataset(air_temperatures_file)

    snapshots = pd.date_range(freq="h", **snakemake.params.snapshots)
    capacity_factors = pd.DataFrame(index=snapshots)

    # bespoke computation of capacity factors for each bus.
    # Considering the respective temperatures, we compute
    # the deviation from the average temperature and relate it
    # to capacity factors based on the data from above.
    for bus in index:
        temp = air_temp.sel(name=bus).to_dataframe()["temperature"]
        capacity_factors[bus] = np.interp((temp - temp.mean()).values, x, y)

    return capacity_factors

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_egs_potential",
            interconnect="texas",
            clusters=40,
        )
    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    elec_config = snakemake.config["electricity"]

    network_regions = (
        gpd.read_file(snakemake.input.regions_onshore)
        .set_index("name", drop=True)
        .set_crs(epsg=4326)
    )


    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.config["costs"],
        elec_config["max_hours"],
        Nyears,
    )

    n.buses["location"] = n.buses.index


    add_nice_carrier_names(n, snakemake.config)

    regions_gpd = snakemake.input.regions_onshore
    geo_egs_sc = snakemake.input.geo_egs_sc
    cost_reduction = snakemake.params.cost_reduction
    add_egs(n, regions_gpd, geo_egs_sc, cost_reduction)
    if any("geothermal" in s for s in elec_config['extendable_carriers']['StorageUnit']): 
        attach_geo_storageunits(n, costs, elec_config, regions_gpd, geo_egs_sc)


    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])