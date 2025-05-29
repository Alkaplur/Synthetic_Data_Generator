from dataclasses import dataclass
from enum import Enum
from typing import List, Any, Optional

class DataGenerationMode(Enum):
    FROM_SAMPLE = "from_sample"
    FROM_DEFINITION = "from_definition"

@dataclass
class CustomerVariable:
    name: str
    data_type: str  # 'categorical', 'numerical', 'text', 'date', 'email', 'phone'
    description: str
    possible_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    pattern: Optional[str] = None
    distribution: Optional[str] = 'uniform'  # 'uniform', 'normal', 'exponential'

@dataclass
class ProductDefinition:
    name: str
    category: str
    price_range: tuple
    features: List[str]
    purchase_probability: float = 0.8  # Probabilidad de compra 