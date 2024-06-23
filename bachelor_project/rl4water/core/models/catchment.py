from core.models.facility import Facility
from typing import List
import os
import numpy as np

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(dir_path, "../data/")

class Catchment(Facility):
    def __init__(self, name: str, directoryno) -> None:
        super().__init__(name)
        self.name = name
        self.directory = ["120Hurst_grouped", "150ias_grouped", "Baseline_grouped"][directoryno]
        randint = np.random.randint(0, 100)
        self.all_water_accumulated: List[float] = np.loadtxt(f"{data_directory}catchmentsData/{self.directory}/trace{randint}/{name}.txt").tolist()

    def determine_reward(self) -> float:
        return 0

    def determine_consumption(self) -> float:
        return -self.all_water_accumulated[self.timestep % len(self.all_water_accumulated)]

    def is_truncated(self) -> bool:
        return self.timestep >= len(self.all_water_accumulated)

    def determine_info(self) -> dict:
        return {"water_consumption": self.determine_consumption()}

    def reset(self) -> None:
        super().reset()
        randint = np.random.randint(0, 100)
        self.all_water_accumulated = np.loadtxt(f"{data_directory}catchmentsData/{self.directory}/trace{randint}/{self.name}.txt").tolist()
