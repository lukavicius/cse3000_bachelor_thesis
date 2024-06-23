from typing import Tuple
from pathlib import Path
from core.models.facility import ControlledFacility
from gymnasium.spaces import Box, Space
import numpy as np
from numpy.core.multiarray import interp as compiled_interp
from array import array
from bisect import bisect_right
import decimal
import warnings

dam_data_directory = Path(__file__).parents[1] / "data" / "dams"


class Dam(ControlledFacility):
    """
    A class used to represent reservoirs/dams of the problem

    Attributes
    ----------
    name: str
        Lowercase non-spaced name of the reservoir
    storage_vector: np.array (1xH)
        m3
        A vector that holds the volume of the water in the reservoir
        throughout the simulation horizon
    level_vector: np.array (1xH)
        m
        A vector that holds the elevation of the water in the reservoir
        throughout the simulation horizon
    release_vector: np.array (1xH)
        m3/s
        A vector that holds the actual average release per month
        from the reservoir throughout the simulation horizon
    evap_rates: np.array (1x12)
        cm
        Monthly evaporation rates of the reservoir

    Methods
    -------
    determine_info()
        Return dictionary with parameters of the dam.
    storage_to_level(h=float)
        Returns the level(height) based on volume.
    level_to_storage(s=float)
        Returns the volume based on level(height).
    level_to_surface(h=float)
        Returns the surface area based on level.
    integration(
        total_seconds: int,
        release_action: float,
        net_inflow_per_second: float,
        integ_step: int,
        )
        Returns average monthly water release.
    """

    def __init__(
        self,
        name: str,
        observation_space: Space,
        action_space: Box,
        objective_function,
        objective_name: str = "",
        max_capacity: float = float("Inf"),
        stored_water: float = 0,
        evap_increase_coef = 0.004,
        lambda_mean = 0.5,
        lambda_std = 0.5
    ) -> None:
        super().__init__(name, observation_space, action_space, max_capacity)
        self.stored_water: float = stored_water

        self.evap_increase_coef = evap_increase_coef
        self.lambda_std = lambda_std
        self.lambda_mean = lambda_mean

        self.evap_rates_raw = np.loadtxt(dam_data_directory / f"evap_{name}.txt")
        self.evap_rates = self.generate_year_evaps()
        self.storage_to_minmax_rel = np.loadtxt(dam_data_directory / f"store_min_max_release_{name}.txt")
        self.storage_to_level_rel = np.loadtxt(dam_data_directory / f"store_level_rel_{name}.txt")
        self.storage_to_surface_rel = np.loadtxt(dam_data_directory / f"store_sur_rel_{name}.txt")

        self.storage_vector = []
        self.level_vector = []
        self.inflow_vector = []
        self.release_vector = []

        # Initialise storage vector
        self.storage_vector.append(stored_water)

        self.objective_function = objective_function
        self.objective_name = objective_name

        # TODO: Read it from file
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

        # self.water_level = self.storage_to_level(self.stored_water)

    import numpy as np
    import warnings

    def generate_year_evaps(self):
        temps_years = []
        for i in range(20):
            try:
                constant_value = ((1 + self.evap_increase_coef) ** (6 + i)) - 1
                T = self.evap_rates_raw + np.abs(np.array(self.evap_rates_raw) * constant_value)
                T_min = T.min()
                T_max = T.max()
                T_norm = (T - T_min) / (T_max - T_min)
                lambda_val = np.random.normal(self.lambda_mean, self.lambda_std)

                # Apply log transformation within a warnings context manager
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', category=RuntimeWarning)
                    try:
                        T_log = np.log(1 + lambda_val * T_norm)
                    except RuntimeWarning as e:
                        raise ValueError("RuntimeWarning encountered in log transformation")

                # Scale and shift back to the original range
                T_scaled = T_log * (T_max - T_min) / np.log(1 + lambda_val) + T_min

                # Adjust mean to match the original data
                original_mean = T.mean()
                scaled_mean = T_scaled.mean()
                T_final = T_scaled + (original_mean - scaled_mean)
            except Exception as e:
                # If any exception occurs, keep T_final as the original T
                T_final = T

            temps_years = np.append(temps_years, T_final)

        return temps_years

    def determine_reward(self) -> float:
        # Pass water level to reward function
        return self.objective_function(self.storage_to_level(self.stored_water))

    def determine_outflow(self, action: float) -> float:
        # Timestep is one month
        # Get the number of days in the month
        nu_days = self.nu_of_days_per_month[self.determine_month()]
        # Calculate hours
        total_hours = nu_days * 24
        # Calculate seconds
        total_seconds = total_hours * 3600
        # Calculate integration step
        integ_step = total_seconds / (nu_days * 48)

        self.inflow_vector = np.append(self.inflow_vector, self.inflow)

        # Calculate outflow using integration function
        outflow = self.integration(
            total_seconds,
            action,
            self.inflow,
            integ_step,
        )
        # print("Outflow of water:", outflow, "from action", action)
        return outflow

    def determine_info(self) -> dict:
        info = {
            "name": self.name,
            "stored_water": self.stored_water,
            "current_level": self.level_vector[-1] if self.level_vector else None,
            "current_release": self.release_vector[-1] if self.release_vector else None,
            "evaporation_rates": self.evap_rates.tolist(),
        }
        return info

    def determine_observation(self) -> float:
        return self.stored_water

    def is_terminated(self) -> bool:
        return self.stored_water > self.max_capacity or self.stored_water < 0

    def determine_month(self):
        return self.timestep % 12

    def storage_to_level(self, s: float) -> float:
        return self.modified_interp(s, self.storage_to_level_rel[0], self.storage_to_level_rel[1])

    def storage_to_surface(self, s: float) -> float:
        return self.modified_interp(s, self.storage_to_surface_rel[0], self.storage_to_surface_rel[1])

    def level_to_minmax(self, h):
        return (
            np.interp(h, self.rating_curve[0], self.rating_curve[1]),
            np.interp(h, self.rating_curve[0], self.rating_curve[2]),
        )

    def storage_to_minmax(self, s):
        return (
            np.interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[1]),
            np.interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[2]),
        )

    def integration(
        self,
        total_seconds: int,
        release_action: float,
        net_inflow_per_second: float,
        integ_step: int,
    ) -> float:
        """
        Converts the flows of the reservoir into storage. Time step
        fidelity can be adjusted within a for loop. The core idea is to
        arrive at m3 storage from m3/s flows.

        Parameters
        ----------
        total_seconds: int
            Number of seconds in the timestep.
        release_action: float
            How much m3/s of water should be released.
        net_inflow_per_second: float
            Total inflow to this Dam measured in m3/s.
        integ_step: int
            Size of the integration step.

        Returns
        -------
        avg_monthly_release: float
            Average monthly release given in m3.
        """
        current_storage = self.storage_vector[-1]
        in_month_releases = np.empty(0, dtype=np.float64)
        monthly_evap_total = 0
        integ_step_count = total_seconds / integ_step

        for _ in np.arange(0, total_seconds, integ_step):
            surface = self.storage_to_surface(current_storage)

            evaporation = surface * (self.evap_rates[self.timestep] / (100 * integ_step_count))
            monthly_evap_total += evaporation

            min_possible_release, max_possible_release = self.storage_to_minmax(current_storage)

            release_per_second = min(max_possible_release, max(min_possible_release, release_action))

            in_month_releases = np.append(in_month_releases, release_per_second)

            total_addition = net_inflow_per_second * integ_step

            current_storage += total_addition - evaporation - release_per_second * integ_step

        # Update the amount of water in the Dam
        self.storage_vector.append(current_storage)
        self.stored_water = current_storage

        # Calculate the ouflow of water
        avg_monthly_release = np.mean(in_month_releases, dtype=np.float64)
        self.release_vector.append(avg_monthly_release)

        # Record level based on storage for time t
        self.level_vector.append(self.storage_to_level(current_storage))
        return avg_monthly_release

    @staticmethod
    def modified_interp(x: float, xp: float, fp: float, left=None, right=None) -> float:
        fp = np.asarray(fp)

        return compiled_interp(x, xp, fp, left, right)

    def reset(self) -> None:
        super().reset()
        stored_water = self.storage_vector[0]
        self.storage_vector = [stored_water]
        self.stored_water = stored_water
        self.level_vector = []
        self.release_vector = []
        self.evap_rates = self.generate_year_evaps()
