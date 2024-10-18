"""
Module for building heating and cooling infrastructure.
"""

import logging
from typing import Optional
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from constants import NG_MWH_2_MMCF, STATE_2_CODE, COAL_dol_ton_2_MWHthermal
from eia import FuelCosts
from _helpers import calculate_annuity
from constants import STATE_2_CODE, STATES_INTERCONNECT_MAPPER
from add_sectors import add_sector_foundation

logger = logging.getLogger(__name__)
CODE_2_STATE = {v: k for k, v in STATE_2_CODE.items()}


def build_geothermal_heat(
    n: pypsa.Network,
    cost_direct_use_path: str, 
    region_cluster_path: str, 
    geo_egs_sc_path: str, 
    **kwargs,
) -> None:
    """
    Main funtion to interface with.
    """

    # Add geothermal to bus

    add_sector_foundation(n, 
                          carrier='geothermal', 
                          add_supply = True)


    # Geothermal section
    cost_direct_use = pd.read_csv(cost_direct_use_path)
    region_onshore = gpd.read_file(region_cluster_path)
    geothermal_resources = (
        gpd.read_file(geo_egs_sc_path)
                        .to_crs(4326)
                        .rename(columns={'name':'county'})
    )

    heat_carrier_list = n.loads[n.loads['carrier'].str.endswith('heat')]['carrier'].unique()

    # Loop through all heat if temperature met demand
    for carrier in heat_carrier_list:
        add_industrial_geothermal_tech(
            n,
            region_cluster = region_onshore,
            geothermal_resources = geothermal_resources,
            costs_direct_use = cost_direct_use,
            carrier_name = carrier
        )
    # else: 
    #     raise NotImplementedError


def add_industrial_geothermal_tech(
    n: pypsa.Network,
    region_cluster, 
    geothermal_resources,
    costs_direct_use, 
    carrier_name = None,
    **kwargs,
) -> None:
    
    region_onshore_geo = clean_subsurface_resources(region_cluster,
                                                  geothermal_resources)

    temp = carrier_name.split('-')[2]
    temp_int = int(temp.replace("T", ""))

    # Add heat pump bashed on the temperature provided
    add_industrial_geothermal_direct_use(n, 
                                         temp = temp_int,
                                         costs_direct_use = costs_direct_use,
                                         region_onshore_geo = region_onshore_geo, 
                                         carrier_name = carrier_name,)

def clean_subsurface_resources(
    region_cluster,
    geothermal_resources,
)-> None:
    
    region_onshore_geo = gpd.sjoin(region_cluster, geothermal_resources[['geoid', 'county', 'st_abbr', 'depth', 'potential_mw', 'geometry']], how="left").reset_index(drop=True)

    return region_onshore_geo

def add_industrial_geothermal_direct_use(
    n: pypsa.Network,
    temp: int,
    costs_direct_use: pd.DataFrame,
    region_onshore_geo: pd.DataFrame,
    carrier_name: str, 

) -> None:
    lifetime = 30
    efficiency = 1

    # Round production temperature to nearest 50 degC
    costs_direct_use['temp_resource'] = round(costs_direct_use['prod_temp_degc_round'] / 50) * 50

    costs_direct_use = costs_direct_use[costs_direct_use['temp_resource'].isin(range(0, 1000, 50))]
    costs_direct_use = costs_direct_use[costs_direct_use['capex_mw'] > 0]
    
    # Get CAPEX per MW
    # costs_direct_use['capex_mw'] = costs_direct_use['capex_kw'] * 1e3
    # Find the cheapest depth/temp combination across all temp gradient
    costs_direct_use = costs_direct_use.groupby(['depth', 'temp_resource'])[['capex_mw']].agg('min').reset_index()

    # Combine the cost data with resource data
    region_onshore_geo_county = region_onshore_geo.merge(costs_direct_use, on=['depth'], how='inner')
    
    # Sum up all available resources at the same CAPEX and temperature
    region_onshore_geo_county = region_onshore_geo_county.groupby(['name', 'temp_resource','capex_mw'])[['potential_mw']].agg('sum').reset_index()
    region_onshore_geo_county_cheapest = region_onshore_geo_county.groupby(['name', 'temp_resource', 'capex_mw'])[['potential_mw']].agg('sum').reset_index()
    region_onshore_geo_county_cheapest = region_onshore_geo_county_cheapest.groupby(['name', 'temp_resource'])[['capex_mw','potential_mw']].agg('first').reset_index()
    
    # Clean format
    region_onshore_geo_county_cheapest = region_onshore_geo_county_cheapest.rename(columns={'name':'bus0'})
    region_onshore_geo_county_cheapest['temp_resource'] = region_onshore_geo_county_cheapest['temp_resource'].astype('int')

    # Assume a fixed FOM from GEOPHIRES
    region_onshore_geo_county_cheapest['fom_pct'] = 1.41/41.15

    # Convert CAPEX to annualized capital cost
    region_onshore_geo_county_cheapest["capital_cost"] = (
        (
            calculate_annuity(lifetime, 0.055)
             + region_onshore_geo_county_cheapest["fom_pct"]
        )
        * region_onshore_geo_county_cheapest["capex_mw"]
        * n.snapshot_weightings.objective.sum() / 8760.0
    )


    # Add links
    loads = n.loads[(n.loads.carrier == carrier_name)]

    geo_du = pd.DataFrame(index=loads.bus)
    geo_du["state"] = geo_du.index.map(n.buses.STATE)
    geo_du["bus0"] = geo_du.index.map(lambda x: x.split(carrier_name)[0].strip())
    geo_du["bus1"] = geo_du.index
    geo_du["carrier"] = "ind-geo-du"
    geo_du["temp_resource"] = temp
    geo_du.index = geo_du.index.map(lambda x: x.split("-heat")[0])

    geo_du_region = geo_du.reset_index().merge(region_onshore_geo_county_cheapest, on=['temp_resource', 'bus0'], how='inner')
    geo_du_region.index = geo_du_region.bus1.map(lambda x: x.split("-heat")[0])
    geo_du_region["bus0"] = geo_du_region.state + ' geothermal'


    if not geo_du_region.empty: 
        n.madd(
            "Link",
            geo_du.index,
            suffix="-geo-du",  # 'ind' included in index already
            bus0=geo_du_region.bus0,
            bus1=geo_du_region.bus1,
            carrier=geo_du_region.carrier,
            efficiency=efficiency,
            capital_cost=geo_du_region.capital_cost,
            p_nom_extendable=True,
            p_nom_max = geo_du_region.potential_mw,
            lifetime=lifetime,
        )
    else: 
        logger.warning(
            f"Skip adding geothermal direct use for {carrier_name} since unable to provide required heat."
        )
