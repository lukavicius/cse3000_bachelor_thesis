import pprint
from collections import OrderedDict

import numpy as np
from pathlib import Path
from gymnasium import Space
from gymnasium.spaces import Box
from core.envs.water_management_system import WaterManagementSystem
from core.models.dam import Dam
from core.models.flow import Flow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_system import IrrigationSystem
from core.models.catchment import Catchment
import csv

make_csv = True


def nile_river_simulation(nu_of_timesteps=240):
    # Create power plant, dam and irrigation system. Initialise with semi-random parameters.
    # Set objective functions to identity for power plant, minimum_water_level for dam and water_deficit_minimised
    # for irrigation system.

    water_management_system = create_nile_river_env()

    if make_csv:
        with open("verification/group13.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Year",
                    "Input",
                    "Gerd_storage",
                    "Gerd_release",
                    "Roseires_storage",
                    "Roseires_release",
                    "Sennar_storage",
                    "Sennar_release",
                    "Had_storage",
                    "Had_release",
                    "Gerd_production",
                ]
            )
            for i in range(nu_of_timesteps):
                action = generateOutput()
                (
                    final_observation,
                    final_reward,
                    final_terminated,
                    final_truncated,
                    final_info,
                ) = water_management_system.step(action)
                writer.writerow(
                    [
                        i,
                        action,
                        ensure_float(final_info.get("GERD")["stored_water"]),
                        ensure_float(final_info.get("GERD")["current_release"]),
                        ensure_float(final_info.get("Roseires")["stored_water"]),
                        ensure_float(final_info.get("Roseires")["current_release"]),
                        ensure_float(final_info.get("Sennar")["stored_water"]),
                        ensure_float(final_info.get("Sennar")["current_release"]),
                        ensure_float(final_info.get("HAD")["stored_water"]),
                        ensure_float(final_info.get("HAD")["current_release"]),
                        ensure_float(final_info.get("GERD_power_plant")["monthly_production"]),
                    ]
                )
    else:
        for _ in range(nu_of_timesteps):
            action = water_management_system.action_space.sample()
            action = np.array(list(action.values())).flatten()
            print("Action:", action, "\n")
            (
                final_observation,
                final_reward,
                final_terminated,
                final_truncated,
                final_info,
            ) = water_management_system.step(action)
            print("Reward:", final_reward)
            pprint.pprint(final_info)
            print("Is finished:", final_truncated, final_terminated)


def create_nile_river_env(directory=2, growth_coef=0.02, irr_lambda_std=0.01, evap_increase_coef=0.004,
                          evap_lambda_mean=0.5, evap_lambda_std=0.5):
    # Ethiopia
    GERD_dam = Dam(
        "GERD",
        Box(low=0, high=80000000000),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=15000000000.0,
        evap_increase_coef=evap_increase_coef,
        lambda_mean=evap_lambda_mean,
        lambda_std=evap_lambda_std,
    )
    GERD_power_plant = PowerPlant(
        "GERD_power_plant",
        Objective.identity,
        "ethiopia_power",
        efficiency=0.93,
        max_turbine_flow=4320,
        head_start_level=507,
        max_capacity=6000,
        dam=GERD_dam,
    )
    data_directory = Path(__file__).parent / "core" / "data"
    # Sudan
    DSSennar_irr_system = IrrigationSystem(
        "DSSennar",
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
        growth_coef,
        irr_lambda_std
    )
    Gezira_irr_system = IrrigationSystem(
        "Gezira",
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
        growth_coef,
        irr_lambda_std
    )
    Hassanab_irr_system = IrrigationSystem(
        "Hassanab",
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
        growth_coef,
        irr_lambda_std
    )
    Tamaniat_irr_system = IrrigationSystem(
        "Taminiat",
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
        growth_coef,
        irr_lambda_std
    )
    USSennar_irr_system = IrrigationSystem(
        "USSennar",
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
        growth_coef,
        irr_lambda_std
    )
    Roseires_dam = Dam(
        "Roseires",
        Box(low=0, high=80000000000),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=4571250000.0,
        evap_increase_coef=evap_increase_coef,
        lambda_mean=evap_lambda_mean,
        lambda_std=evap_lambda_std,
    )
    Sennar_dam = Dam(
        "Sennar",
        Box(low=0, high=80000000000),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=434925000.0,
        evap_increase_coef=evap_increase_coef,
        lambda_mean=evap_lambda_mean,
        lambda_std=evap_lambda_std,
    )
    # Egypt
    Egypt_irr_system = IrrigationSystem(
        "Egypt",
        Objective.water_deficit_minimised,
        "egypt_deficit_minimised",
        growth_coef,
        irr_lambda_std
    )
    HAD_dam = Dam(
        "HAD",
        Box(low=0, high=80000000000),
        Box(0, 4000),
        Objective.minimum_water_level,
        "HAD_minimum_water_level",
        stored_water=137025000000.0,
        evap_increase_coef=evap_increase_coef,
        lambda_mean=evap_lambda_mean,
        lambda_std=evap_lambda_std,
    )
    # Create 'edges' between Facilities.
    # TODO: determine max capacity for flows
    GERD_inflow = Inflow(
        "blueNile",
        GERD_dam,
        float("inf"),
        directoryno=directory
    )

    GerdToRoseires_catchment = Catchment(
        "gerdToRoseires", directory
    )
    # TODO: add catchment 1 inflow to sources of Roseires (inflow with destination Roseires)

    Roseires_flow = Flow("roseires_flow", [GERD_dam, GerdToRoseires_catchment], Roseires_dam, float("inf"))

    RoseiresToAbuNaama_catchment = Catchment(
        "roseiresToAbuNaama", directory
    )

    # TODO: add catchment 2 inflow to sources of USSennar (inflow with destination USSennar)
    upstream_Sennar_received_flow = Flow(
        "upstream_Sennar_received_flow",
        [Roseires_dam, RoseiresToAbuNaama_catchment],
        USSennar_irr_system,
        float("inf"),
    )

    SukiToSennar_catchment = Catchment(
        "sukiToSennar", directory
    )

    # TODO: add catchment 3 inflow to sources of Sennar (inflow with destination USSennar)
    Sennar_flow = Flow("sennar_flow", [USSennar_irr_system, SukiToSennar_catchment], Sennar_dam, float("inf"))

    Gezira_received_flow = Flow("gezira_received_flow", [Sennar_dam], Gezira_irr_system, float("inf"))

    Dinder_catchment = Catchment("dinder", directory)

    Rahad_catchment = Catchment("rahad", directory)

    downstream_Sennar_received_flow = Flow(
        "downstream_sennar_received_flow",
        [Gezira_irr_system, Dinder_catchment, Rahad_catchment],
        DSSennar_irr_system,
        float("inf"),
    )
    WhiteNile_catchment = Catchment(
        "whiteNile",
        directory
    )
    Taminiat_received_flow = Flow(
        "taminiat_received_flow",
        [DSSennar_irr_system, WhiteNile_catchment],
        Tamaniat_irr_system,
        float("inf"),
    )

    Atbara_catchment = Catchment("atbara", directory)

    # TODO: change Hassanab received flow to depend on leftover flow from Taminiat in previous month (see A.2.8)
    Hassanab_received_flow = Flow(
        "hassanab_received_flow",
        [Tamaniat_irr_system, Atbara_catchment],
        Hassanab_irr_system,
        float("inf"),
    )
    HAD_flow = Flow("had_flow", [Hassanab_irr_system], HAD_dam, float("inf"))
    Egypt_flow = Flow("egypt_flow", [HAD_dam], Egypt_irr_system, float("inf"))
    # Create water management system. Add Facilities in the topological order (in the list).
    # Egypt deficit reward goes negative when there is a deficit. Otherwise is 0.
    water_management_system = WaterManagementSystem(
        water_systems=[
            GERD_inflow,
            GERD_dam,
            GERD_power_plant,
            GerdToRoseires_catchment,
            Roseires_flow,
            Roseires_dam,
            RoseiresToAbuNaama_catchment,
            upstream_Sennar_received_flow,
            USSennar_irr_system,
            SukiToSennar_catchment,
            Sennar_flow,
            Sennar_dam,
            Gezira_received_flow,
            Gezira_irr_system,
            Dinder_catchment,
            Rahad_catchment,
            downstream_Sennar_received_flow,
            DSSennar_irr_system,
            WhiteNile_catchment,
            Taminiat_received_flow,
            Tamaniat_irr_system,
            Atbara_catchment,
            Hassanab_received_flow,
            Hassanab_irr_system,
            HAD_flow,
            HAD_dam,
            Egypt_flow,
            Egypt_irr_system,
        ],
        rewards={
            "ethiopia_power": 0,
            "sudan_deficit_minimised": 0,
            "egypt_deficit_minimised": 0,
            "HAD_minimum_water_level": 0,
        },
        step_limit=240,  # Use lower horizon for local training
    )

    return water_management_system


def generateOutput():
    random_values = np.random.rand(
        4,
    ) * [10000, 10000, 10000, 4000]

    return random_values


def ensure_float(value):
    if isinstance(value, np.ndarray):
        return value.item()
    return value


if __name__ == "__main__":
    nile_river_simulation()
