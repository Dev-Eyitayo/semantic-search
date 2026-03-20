from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from db.session import get_db
from db.models.test_lead import TestLead
from schemas.test_lead import TestLeadCreate, TestLeadRead

router = APIRouter()

@router.post("/", response_model=TestLeadRead)
async def create_lead(lead_in: TestLeadCreate, db: AsyncSession = Depends(get_db)):
    
    query = select(TestLead).where(TestLead.email == lead_in.email)
    result = await db.execute(query)
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

   
    new_lead = TestLead(**lead_in.model_dump())
    db.add(new_lead)
    await db.commit()
    await db.refresh(new_lead)
    return new_lead

@router.get("/", response_model=List[TestLeadRead])
async def read_leads(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(TestLead))
    return result.scalars().all()