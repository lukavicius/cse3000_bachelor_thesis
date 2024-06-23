class Objective:
    MINIMUM_WATER_LEVEL = 159

    @staticmethod
    def no_objective(*args):
        return 0.0

    @staticmethod
    def identity(value: float) -> float:
        return value

    @staticmethod
    def minimum_water_level(water_level: float) -> float:
        return 0.0 if water_level < Objective.MINIMUM_WATER_LEVEL else 1.0

    @staticmethod
    def water_deficit_minimised(demand: float, received: float) -> float:
        return -max(0.0, demand - received)

    SCALAR = 1000000000

    @staticmethod
    def scalar_identity(value: float) -> float:
        return value / Objective.SCALAR
