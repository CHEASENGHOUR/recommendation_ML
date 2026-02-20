from abc import ABC, abstractmethod

class Laptop(ABC):
    @abstractmethod
    def prepare(self, data=None):
        pass
    
class GamingLaptop(Laptop):
    def prepare(self):
        return "Preparing a high-performance Gaming Laptop with RTX GPU"


class OfficeLaptop(Laptop):
    def prepare(self):
        return "Preparing a lightweight Office Laptop with long battery"


class UltrabookLaptop(Laptop):
    def prepare(self):
        return "Preparing a thin and premium Ultrabook Laptop"
    
class LaptopFactory:

    def create_laptop(self, laptop_type):

        if laptop_type == "gaming":
            return GamingLaptop()

        elif laptop_type == "office":
            return OfficeLaptop()

        elif laptop_type == "ultrabook":
            return UltrabookLaptop()

        else:
            raise ValueError("Unknown laptop type")