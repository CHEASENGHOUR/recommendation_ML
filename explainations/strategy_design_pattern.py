from abc import ABC, abstractmethod

class InputStrategy(ABC):
    @abstractmethod
    def get_input(self, data=None):
        pass
    
    
class PromptInputStrategy(InputStrategy):
    def get_input(self, data=None):
        if data:
            return f"Prompt input for data: {data}"
        return input("Enter your input: ")
    
class FormInputStrategy(InputStrategy):
    def get_input(self, data=None):
        # Simulate form input
        if data:
            return f"Form input for data: {data}"
        return "Form input data"
    
    
class InputContext:
    def __init__(self, strategy: InputStrategy):
        self._strategy = strategy
        
    def set_strategy(self, strategy: InputStrategy):
        self._strategy = strategy
        
    def get_input(self):
        return self._strategy.get_input()
    
if __name__ == "__main__":
    context = InputContext(PromptInputStrategy())
    print("Using PromptInputStrategy:")
    print(context.get_input("i want an gaming laptop!!!"))
    
    context.set_strategy(FormInputStrategy())
    print("Using FormInputStrategy:")
    print(context.get_input(['laptop', 'gaming', '16GB RAM', '512GB SSD', 'NVIDIA GPU', 'i7 processor', 'high spec', 'good cooling system', 'RGB lighting', 'thin and light design', 'long battery life', 'high refresh rate display', 'good build quality', 'affordable price']))