from core.models.facility import Facility
from core.models.dam import Dam
from scipy.constants import g
import numpy as np


class PowerPlant(Facility):
    """
    Class to represent Hydro-energy Powerplant

    Attributes:
    ----------
    name : str
        identifier
    efficiency : float
        Efficiency coefficient (mu) used in hydropower formula
    max_turbine_flow : float
        Maximum possible flow that can be passed through the turbines for the
        purpose of hydroenergy production
    head_start_level : float
        Minimum elevation of water level that is used to calculate hydraulic
        head for hydropower production
    max_capacity : float
        Total design capacity (mW) of the plant
    water_level_coeff : float
        Coefficient that determines the water level based on the volume of outflow
        Used to calculate at what level the head of the power plant operates
    water_usage : float
        Amount of water  that is used by plant, decimal coefficient

    Methods:
    ----------
    determine_reward():
        Calculates the reward (power generation) given the values of its attributes
    determine_consumption():
        Determines how much water is consumed by the power plant
    determine_info():
        Returns info about the hydro-energy powerplant
    """

    def __init__(
        self,
        name: str,
        objective_function,
        objective_name: str,
        efficiency: float,
        max_turbine_flow: float,
        head_start_level: float,
        max_capacity: float,
        dam: Dam = None,
        # TODO: determine actual water usage for power plants, 0.0 for ease now
        water_usage: float = 0.0,
    ) -> None:
        super().__init__(name, objective_function, objective_name)
        self.efficiency = efficiency
        self.max_turbine_flow = max_turbine_flow
        self.head_start_level = head_start_level
        self.max_capacity = max_capacity
        self.dam = dam
        self.water_usage = water_usage
        self.production_vector = np.empty(0, dtype=np.float64)

        self.nu_of_days_per_month = [
            31,  # January
            28,  # February (non-leap year)
            31,  # March
            30,  # April
            31,  # May
            30,  # June
            31,  # July
            31,  # August
            30,  # September
            31,  # October
            30,  # November
            31,  # December
        ]

    # Constants are configured as parameters with default values
    def determine_production(self) -> float:
        """
        Calculates power production in MWh.

        Returns:
        ----------
        float
            Plant's power production in mWh
        """
        m3_to_kg_factor: int = 1000
        w_mw_conversion: float = 1e-6
        # Turbine flow is equal to outflow, as long as it does not exceed maximum turbine flow
        turbine_flow = min(self.dam.release_vector[-1], self.max_turbine_flow)

        # Uses water level from dam to determine water level
        water_level = self.dam.level_vector[-1] if self.dam.level_vector else 0
        # Calculate at what level the head will generate power, using water_level of the outflow and head_start_level
        head = max(0.0, water_level - self.head_start_level)

        # Calculate power in mW, has to be lower than or equal to capacity
        power_in_mw = min(
            self.max_capacity,
            turbine_flow * head * m3_to_kg_factor * g * self.efficiency * w_mw_conversion,
        )

        nu_days = self.nu_of_days_per_month[self.determine_month()]
        total_hours = nu_days * 24

        # Hydro-energy power production in mWh
        production = power_in_mw * total_hours

        self.production_vector = np.append(self.production_vector, production)

        return production

    def determine_reward(self) -> float:
        """
        Determines reward for the power plant using the power production.

        Parameters:
        ----------
        objective_function : (float) -> float
            Function calculating the objective given the power production.

        Returns:
        ----------
        float
            Reward.
        """
        # print("Produced", self.determine_production(), "power from", self.dam.release_vector[-1])
        return self.objective_function(self.determine_production())

    def determine_consumption(self) -> float:
        """
        Determines water consumption.

        Returns:
        ----------
        float
            How much water is consumed
        """
        return self.inflow * self.water_usage

    def determine_info(self) -> dict:
        """
        Determines info of hydro-energy power plant

        Returns:
        ----------
        dict
            Info about power plant (name, inflow, outflow, water usage, timestep, total production)
        """
        return {
            "name": self.name,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "monthly_production": self.production_vector[-1],
            "water_usage": self.water_usage,
            "total production (MWh)": sum(self.production_vector),
        }

    def determine_month(self):
        return self.timestep % 12

    def reset(self) -> None:
        super().reset()
        self.production_vector = np.empty(0, dtype=np.float64)
