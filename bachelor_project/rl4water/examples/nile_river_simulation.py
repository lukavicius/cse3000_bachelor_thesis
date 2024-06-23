import numpy as np
from pathlib import Path
from gymnasium.spaces import Box
from core.envs.water_management_system import WaterManagementSystem
from core.models.dam import Dam
from core.models.flow import Flow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_system import IrrigationSystem
from core.models.catchment import Catchment


def create_nile_river_env() -> WaterManagementSystem:
    # Ethiopia
    GERD_dam = Dam(
        "GERD",
        Box(low=0, high=80000000000),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=15000000000.0,
    )
    GERD_power_plant = PowerPlant(
        "GERD_power_plant",
        # Objective.identity,
        Objective.scalar_identity,
        "ethiopia_power",
        efficiency=0.93,
        max_turbine_flow=4320,
        head_start_level=507,
        max_capacity=6000,
        dam=GERD_dam,
    )
    data_directory = Path(__file__).parents[1] / "core" / "data"
    # Sudan
    DSSennar_irr_system = IrrigationSystem(
        "DSSennar_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_DSSennar.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    Gezira_irr_system = IrrigationSystem(
        "Gezira_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Gezira.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    Hassanab_irr_system = IrrigationSystem(
        "Hassanab_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Hassanab.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    Tamaniat_irr_system = IrrigationSystem(
        "Tamaniat_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Tamaniat.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    USSennar_irr_system = IrrigationSystem(
        "USSennar_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_USSennar.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    Roseires_dam = Dam(
        "Roseires",
        Box(low=0, high=80000000000),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=4571250000.0,
    )
    Sennar_dam = Dam(
        "Sennar",
        Box(low=0, high=80000000000),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=434925000.0,
    )
    # Egypt
    Egypt_irr_system = IrrigationSystem(
        "Egypt_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Egypt.txt"),
        Objective.water_deficit_minimised,
        "egypt_deficit_minimised",
    )
    HAD_dam = Dam(
        "HAD",
        Box(low=0, high=80000000000),
        Box(0, 4000),
        Objective.minimum_water_level,
        "HAD_minimum_water_level",
        stored_water=137025000000.0,
    )
    # Create 'edges' between Facilities.
    # TODO: determine max capacity for flows
    GERD_inflow = Inflow(
        "gerd_inflow",
        GERD_dam,
        float("inf"),
        np.loadtxt(data_directory / "catchments" / "InflowBlueNile.txt"),
    )

    GerdToRoseires_catchment = Catchment(
        "GerdToRoseires_catchment", np.loadtxt(data_directory / "catchments" / "InflowGERDToRoseires.txt")
    )
    # TODO: add catchment 1 inflow to sources of Roseires (inflow with destination Roseires)

    Roseires_flow = Flow("roseires_flow", [GERD_dam, GerdToRoseires_catchment], Roseires_dam, float("inf"))

    RoseiresToAbuNaama_catchment = Catchment(
        "RoseiresToAbuNaama_catchment", np.loadtxt(data_directory / "catchments" / "InflowRoseiresToAbuNaama.txt")
    )

    # TODO: add catchment 2 inflow to sources of USSennar (inflow with destination USSennar)
    upstream_Sennar_received_flow = Flow(
        "upstream_Sennar_received_flow",
        [Roseires_dam, RoseiresToAbuNaama_catchment],
        USSennar_irr_system,
        float("inf"),
    )

    SukiToSennar_catchment = Catchment(
        "SukiToSennar_catchment", np.loadtxt(data_directory / "catchments" / "InflowSukiToSennar.txt")
    )

    # TODO: add catchment 3 inflow to sources of Sennar (inflow with destination USSennar)
    Sennar_flow = Flow("sennar_flow", [USSennar_irr_system, SukiToSennar_catchment], Sennar_dam, float("inf"))

    Gezira_received_flow = Flow("gezira_received_flow", [Sennar_dam], Gezira_irr_system, float("inf"))

    Dinder_catchment = Catchment("dinder_catchment", np.loadtxt(data_directory / "catchments" / "InflowDinder.txt"))

    Rahad_catchment = Catchment("rahad_catchment", np.loadtxt(data_directory / "catchments" / "InflowRahad.txt"))

    downstream_Sennar_received_flow = Flow(
        "downstream_sennar_received_flow",
        [Gezira_irr_system, Dinder_catchment, Rahad_catchment],
        DSSennar_irr_system,
        float("inf"),
    )
    WhiteNile_catchment = Catchment(
        "whitenile_catchment",
        np.loadtxt(data_directory / "catchments" / "InflowWhiteNile.txt"),
    )
    Taminiat_received_flow = Flow(
        "taminiat_received_flow",
        [DSSennar_irr_system, WhiteNile_catchment],
        Tamaniat_irr_system,
        float("inf"),
    )

    Atbara_catchment = Catchment("atbara_catchment", np.loadtxt(data_directory / "catchments" / "InflowAtbara.txt"))

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
        step_limit=240,  # Use low horizon for local training
    )

    return water_management_system
