from abc import abstractmethod, ABC
import pandas as pd


class DataInspection(ABC):
    @abstractmethod
    def inspect(self, data: pd.DataFrame) -> None:
        pass
    
class BasicDataInspection(DataInspection):
    def inspect(self, data: pd.DataFrame) -> None:
        print("Data Shape:", data.shape)
        print("\nData Types:\n", data.dtypes)
        
class DataTypesInspection(DataInspection):
    def inspect(self, data: pd.DataFrame) -> None:
        print("\nData Types and Non-null Counts:")
        print(data.info())
        
class MissingValuesInspection(DataInspection):
    def inspect(self, data: pd.DataFrame) -> None:
        print("\nMissing Values Count:")
        print(data.isnull().sum())

class StatisticsInspection(DataInspection):
    def inspect(self, data: pd.DataFrame) -> None:
        print("\nSummary Statistics (Numerical Features):")
        print(data.describe())
        print("\nSummary Statistics (Categorical Features):")
        print(data.describe(include=["O"]))
        
class DataInspector:
    def __init__(self, strategy: DataInspection):
        self._strategy = strategy
    def set_strategy(self, strategy: DataInspection):
        self._strategy = strategy
    def inspect(self, data: pd.DataFrame) -> None:
        self._strategy.inspect(data)
        
        
if __name__ == "__main__":
    pass