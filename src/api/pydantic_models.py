from pydantic import BaseModel

class CustomerData(BaseModel):
    CustomerId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: int
    TransactionHour: int
    TransactionDay: int
    TransactionMonth: int
    TransactionYear: int
    TotalAmount: float
    AvgAmount: float
    TransactionCount: int
    StdAmount: float

class PredictionResponse(BaseModel):
    CustomerId: str
    RiskProbability: float