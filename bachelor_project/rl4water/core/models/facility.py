from abc import ABC, abstractmethod
import numpy as np
from gymnasium.spaces import Space
from gymnasium.core import ObsType, ActType
from typing import SupportsFloat, Tuple
from core.models.objective import Objective


class Facility(ABC):
    def __init__(self, name: str, objective_function=Objective.no_objective, objective_name: str = "") -> None:
        self.name: str = name
        self.timestep: int = 0
        self.inflow: float = 0
        self.outflow: float = 0
        self.inflow_vector = np.empty(0, dtype=np.float64)

        self.objective_function = objective_function
        self.objective_name = objective_name

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_consumption(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_info(self) -> dict:
        raise NotImplementedError()

    def is_terminated(self) -> bool:
        return False

    def is_truncated(self) -> bool:
        return False

    def step(self) -> Tuple[ObsType, float, bool, bool, dict]:
        self.inflow_vector = np.append(self.inflow_vector, self.inflow)

        if self.name == "Hassanab_irr":
            # If inflow_vector has only one element, set inflow to the initial value
            if len(self.inflow_vector) == 1:
                self.inflow = 934.2
            else:
                # Get the previous inflow
                self.inflow = self.inflow_vector[-2]

        self.outflow = self.inflow - self.determine_consumption()
        # TODO: Determine if we need to satisfy any terminating conditions for facility.
        reward = self.determine_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.determine_info()

        self.timestep += 1

        return None, reward, terminated, truncated, info

    def reset(self) -> None:
        self.timestep: int = 0
        self.inflow: float = 0
        self.outflow: float = 0
        self.inflow_vector = np.empty(0, dtype=np.float64)


class ControlledFacility(ABC):
    def __init__(
        self,
        name: str,
        observation_space: Space,
        action_space: ActType,
        objective_function=Objective.no_objective,
        objective_name: str = "",
        max_capacity: float = float("Inf"),
    ) -> None:
        self.name: str = name
        self.timestep: int = 0
        self.inflow: float = 0
        self.outflow: float = 0

        self.observation_space: Space = observation_space
        self.action_space: Space = action_space

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.max_capacity: float = max_capacity

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_outflow(self, action: ActType) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_info(self) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def determine_observation(self) -> ObsType:
        raise NotImplementedError()

    @abstractmethod
    def is_terminated(self) -> bool:
        raise NotImplementedError()

    def is_truncated(self) -> bool:
        return False

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, dict]:
        self.outflow = self.determine_outflow(action)
        # TODO: Change stored_water to multiple outflows.

        observation = self.determine_observation()
        reward = self.determine_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.determine_info()

        self.timestep += 1

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self) -> None:
        self.timestep: int = 0
        self.inflow: float = 0
        self.outflow: float = 0
