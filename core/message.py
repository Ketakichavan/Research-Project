from pydantic import BaseModel
from typing import Optional

class Message(BaseModel):
    sender: str
    receiver: str
    content: str
    round_number: Optional[int] = None
    pattern: Optional[str] = None
    network_condition: Optional[str] = "baseline"
