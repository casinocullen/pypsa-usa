"""
Builds the demand data for the PyPSA network.

**Relevant Settings**

.. code:: yaml

    snapshots:
        start:
        end:
        inclusive:

    scenario:
    interconnect:
    planning_horizons:


**Inputs**

    - base_network:
    - eia: (GridEmissions data file)
    - efs: (NREL EFS Load Forecasts)

**Outputs**

    - demand: Path to the demand CSV file.
"""

# snakemake is not liking this futures import. Removing type hints in context class
# from __future__ import annotations

import calendar
import logging
import sqlite3
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import os
import constants as const
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import configure_logging
from eia import EnergyDemand, TransportationDemand

logger = logging.getLogger(__name__)

STATE_2_CODE = const.STATE_2_CODE
CODE_2_STATE = {value: key for key, value in STATE_2_CODE.items()}
STATE_TIMEZONE = const.STATE_2_TIMEZONE

FIPS_2_STATE = const.FIPS_2_STATE
NAICS = const.NAICS
TBTU_2_MWH = const.TBTU_2_MWH


EPRI_NERC_2_STATE = {
    "ECAR": ["IL", "IN", "KY", "MI", "OH", "WI", "WV"],
    "ERCOT": ["TX"],
    "MAAC": ["MD"],
    "MAIN": ["IA", "KS", "MN", "NE", "ND", "SD"],
    "MAPP": ["AR", "DE", "MO", "MT", "NM", "OK"],
    "NYPCC_NY": ["NJ", "NY"],
    "NPCC_NE": ["CT", "ME", "MA", "NH", "PA", "RI", "VT"],
    "SERC_STV": ["AL", "GA", "LA", "MS", "NC", "SC", "TN", "VA"],
    "SERC_FL": ["FL"],
    "SPP": [],
    "WSCC_NWP": ["ID", "OR", "WA"],
    "WSCC_RA": ["AZ", "CO", "UT", "WY"],
    "WSCC_CNV": ["CA", "NV"],
}

# https://www.epri.com/research/products/000000003002018167
EPRI_SEASON_2_MONTH = {
    "Peak": [5, 6, 7, 8, 9],  # may -> sept
    "OffPeak": [1, 2, 3, 4, 10, 11, 12],  # oct -> april
}


###
# main entry point
###

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_electrical_demand",
            interconnect="usa",
            end_use="power",
        )

    configure_logging(snakemake)

    # Step 1: def _get_load_allocation_factor

    n = pypsa.Network(snakemake.input.network)
    n.buses.Pd = n.buses.Pd.fillna(0)
    bus_load = n.buses.Pd.to_frame(name="Pd").join(n.buses.county.to_frame(name="zone"))
    zone_loads = bus_load.groupby("zone")["Pd"].transform("sum")
    load_per_bus = (bus_load.Pd / zone_loads).to_frame(name="laf")

    # Step 2: def get_load_buses_per_county
    buses_per_county = n.buses[["Pd", "county"]].fillna(0)
    buses_per_county = buses_per_county[buses_per_county.Pd != 0]

    # Step 3: calculate % of share of all buses in a county
    buses_per_county_share = buses_per_county.merge(load_per_bus, on=['Bus'], how='outer').reset_index()
    buses_per_county_share = buses_per_county_share.groupby(['county','Bus'])[['Pd', 'laf']].agg('first').reset_index()

    # For each Heat ID, assign demands per bus per hour
    # Step 4: def _format_epri_data
    epri = pd.read_csv(snakemake.input.epri)
    epri_heat = epri.loc[epri['End Use'] == 'ProcessHeating']

    epri_heat["state"] = epri_heat.Region.map(EPRI_NERC_2_STATE)
    epri_heat["month"] = epri_heat.Season.map(EPRI_SEASON_2_MONTH)
    epri_heat["end_use"] = epri_heat['End Use']


    epri_heat = (
        epri_heat.explode("state")
        .explode("month")
        .drop(
            columns=[
                "Day Type",
                "Sector",
                "End Use",
                "Scale Multiplier",
                "Region",
                "Season",
            ],
        )
        .groupby(["state", "month", "end_use"])
        .mean()
    )
    
    epri_heat = (
        epri_heat.rename(
            columns={x: int(x.replace("HR", "")) - 1 for x in epri_heat.columns},
        )  # start hour is zero
        .reset_index()
        .melt(id_vars=["state", "month", "end_use"], var_name="hour")
        .pivot(index=["state", "month", "hour"], columns=["end_use"])
    )
    epri_heat.columns = epri_heat.columns.droplevel(0)  # drops name "value"
    epri_heat.columns.name = ""


    # Step 5: def _add_epri_snapshots
    def get_days_in_month(year: int, month: int) -> list[int]:
        return [x for x in range(1, calendar.monthrange(year, month)[1] + 1)]

    # set reading and writitng strategies
    sns = n.snapshots
    investment_periods = n.investment_periods.to_list()

    for year in investment_periods:
        epri_heat = epri_heat.reset_index()
        epri_heat["day"] = 1
        epri_heat["year"] = year
        epri_heat["day"] = epri_heat.month.map(lambda x: get_days_in_month(year, x))
        epri_heat = epri_heat.explode("day")
        epri_heat["snapshot"] = pd.to_datetime(epri_heat[["year", "month", "day", "hour"]])

        epri_heat = epri_heat.set_index(["state", "snapshot"]).drop(
            columns=["year", "month", "day", "hour"],
        )

        # Step 6: def _norm_epri_data
        normed = []
        for state in epri_heat.index.get_level_values("state").unique():
            df = epri_heat.xs(state, level="state", drop_level=False)
            normed.append(df.apply(lambda x: x / x.sum()))

        df_normed = pd.concat(normed)
        df_normed = df_normed[np.in1d(df_normed.index.get_level_values(1), n.snapshots.get_level_values(1))]

        # Step 7: Building heating demand by NAICS
        raw_ind_heat = pd.read_csv(snakemake.input.ind_demand)

        ind_heat_county = raw_ind_heat.groupby(['geoid', 'county', 'st_abbr', 'naics','naics_des', 'temp_deg_c'])[['gas', 'oil', 'coal']].agg('sum').reset_index()
        ind_heat_county['geoid'] = ind_heat_county['geoid'].astype(str).str.zfill(5)

        # Remove biomass since they are assumed to be carbon neutral
        ind_heat_county_long = ind_heat_county.melt(id_vars=['geoid', 'county', 'st_abbr', 'naics','naics_des', 'temp_deg_c'], value_vars=['gas', 'oil', 'coal'], ignore_index=False).reset_index()
        ind_heat_county_long = ind_heat_county_long.loc[ind_heat_county_long['value']>0]
        ind_heat_county_long["temp_round_deg_c"] = round(ind_heat_county_long["temp_deg_c"]/ 50, 0) * 50
        ind_heat_county_long["temp_round_deg_c"] = ind_heat_county_long["temp_round_deg_c"].astype(int)
        ind_heat_county_long['annual_mwh'] = ind_heat_county_long['value'] * 277.78 /6

        # Find capacity by assuming consistent output
        ind_heat_county_long['cap_mw'] = ind_heat_county_long['annual_mwh']/8760/6

        # Step 8: Group by NAICS and temp
        ind_heat_county_long_demand = ind_heat_county_long.groupby(['geoid', 'st_abbr', 'naics', 'naics_des','temp_round_deg_c'])['value'].agg('sum').reset_index()

        ind_heat_county_long_demand['annual_mwh'] = ind_heat_county_long_demand['value'] * 277.78 / 6
        ind_heat_county_long_demand['id'] = 'N'+ind_heat_county_long_demand['naics'].astype(str) +  '_T' + ind_heat_county_long_demand['temp_round_deg_c'].astype(str)

        # Step 9: build existing capacity
        ind_heat_county_long_cap = ind_heat_county_long.groupby(['geoid', 'st_abbr', 'naics', 'naics_des','temp_round_deg_c', 'variable'])['cap_mw'].agg('sum').reset_index().rename(columns={'variable':'fuel'})
        ind_heat_county_long_cap['carrier_name'] = 'ind-N'+ind_heat_county_long_cap['naics'].astype(str) +  '-T' + ind_heat_county_long_cap['temp_round_deg_c'].astype(str) + '-heat'

        ind_heat_county_long_cap = ind_heat_county_long_cap.merge(buses_per_county_share, 
                                                                left_on=['geoid'], right_on=['county'], how = 'inner')
        ind_heat_county_long_cap['cap_mw_bus'] = ind_heat_county_long_cap['cap_mw'] * ind_heat_county_long_cap['laf']

        ind_heat_county_long_cap = ind_heat_county_long_cap[['Bus', 'cap_mw_bus', 'fuel', 'carrier_name']]

        ind_heat_county_long_cap.to_csv(snakemake.output.existing_cap, index=False)

        # Step 10: For each Heat ID, assign demands per bus
        ind_heat_county_long_grp = ind_heat_county_long_demand.groupby(['geoid', 'st_abbr', 'id'])['annual_mwh'].agg('sum').reset_index()

        for heat_id in ind_heat_county_long_grp.id.unique():
            ind_heat_county_long_grp_one = ind_heat_county_long_grp[ind_heat_county_long_grp.id == heat_id]
            ind_heat_county_long_grp_one = ind_heat_county_long_grp_one.merge(buses_per_county_share, 
                                                                        left_on=['geoid'], right_on=['county'], how = 'inner')

            if ind_heat_county_long_grp_one.empty:
                pass
            else: 
                ind_heat_county_long_grp_one['mwh_per_bus'] = ind_heat_county_long_grp_one['annual_mwh']*ind_heat_county_long_grp_one['laf']

                ind_heat_county_long_grp_one = ind_heat_county_long_grp_one[['id','Bus', 'st_abbr','mwh_per_bus']]

                # Get hourly profile
                ind_heat_county_long_grp_one_hour = ind_heat_county_long_grp_one.merge(df_normed.reset_index(), 
                                                                        left_on=['st_abbr'], right_on=['state'], how = 'inner')

                ind_heat_county_long_grp_one_hour['mwh_per_bus_per_hour'] = ind_heat_county_long_grp_one_hour['mwh_per_bus']*ind_heat_county_long_grp_one_hour['ProcessHeating']

                # Step final: write csv
                if not os.path.exists(snakemake.params.demand_output_dir):
                    os.makedirs(snakemake.params.demand_output_dir)

                file_name = "industry_" + heat_id + "_" + "heating.csv"
                final_ind_heat_county_long_grp_one_hour = ind_heat_county_long_grp_one_hour[['snapshot','Bus','mwh_per_bus_per_hour']].pivot(index = 'snapshot',columns='Bus', values='mwh_per_bus_per_hour')
                
                final_ind_heat_county_long_grp_one_hour.to_csv(snakemake.params.demand_output_dir + file_name)

    pd.DataFrame().to_csv(snakemake.output.add_to_demand)