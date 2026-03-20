from pydantic import BaseModel, EmailStr, ConfigDict
from datetime import datetime
from typing import Optional

class TestLeadBase(BaseModel):
    email: EmailStr
    interest_area: Optional[str] = "General"


class TestLeadCreate(TestLeadBase):
    pass

class TestLeadRead(TestLeadBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)