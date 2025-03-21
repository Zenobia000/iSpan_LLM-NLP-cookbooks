from langchain.tools import tool
from pydantic import BaseModel, Field


class CalculatorTools():
    @tool("Make a calculation")
    def calculate(operation):
        """Use this to make a calculation"""
        try:
            return eval(operation)
        except SyntaxError:
            return "Error: Invalid syntax in mathmetical expression"
    

# Define a Pydantic model for the tool's input parameters
class CalculationInput(BaseModel):
    operation: str = Field(..., description="The mathematical operation to perform")
    factor: float = Field(..., description="A factor by which to multiply the result of the operation")


@tool("perform calculation", args_schema=CalculationInput, return_direct=True)
def perform_calculation(operation: str, factor: float) -> str:
    """
    Performs a specified mathematical operation and multiplies the result by a given factor.
    
    Parameters:
    - operation (str): A string representing a mathematical operation (e.g., "10 + 5").
    - factor (float): A factor by which to multiply the result of the operation.
    
    Returns:
    - A string representation of the calculation result.
    """
    try:
        # Safely evaluate the mathematical expression
        result = safe_eval(operation) * factor
        
        # Return the result as a string
        return f"The result of '{operation}' multiplied by {factor} is {result}."
    except Exception as e:
        return f"Error performing calculation: {str(e)}"

def safe_eval(expression: str) -> float:
    """
    Safely evaluates a mathematical expression string.
    Only allows basic arithmetic operations and numbers.
    
    Parameters:
    - expression (str): A string representing a mathematical expression.
    
    Returns:
    - The numerical result of the expression.
    
    Raises:
    - ValueError: If the expression contains disallowed characters.
    """
    # Check if expression contains only allowed characters
    allowed = set("0123456789.+-*/() ")
    if not all(c in allowed for c in expression):
        raise ValueError("Expression contains disallowed characters")
    
    # Evaluate the expression
    return eval(expression)
