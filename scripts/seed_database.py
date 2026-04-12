import asyncio
import sys
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from uuid import uuid4
from loguru import logger

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.base import Base
from db.models.user import User
from db.models.property import Property, SavedSearch
from core.config import settings
from core.security import get_password_hash
from core.enums import UserRole, PropertyType, PriceType, PropertyStatus

# Configure logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

async def get_or_create_user(session, email, defaults):
    """Checks if user exists, returns user object, or creates if missing"""
    result = await session.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    
    if not user:
        user = User(email=email, **defaults)
        session.add(user)
        await session.flush()
        logger.info(f"Created user: {email}")
    else:
        logger.info(f"User already exists: {email}")
    return user

async def seed_database():
    """Seed database with realistic Nigerian real estate data"""
    engine = create_async_engine(settings.DATABASE_ASYNC_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        try:
            logger.info("🌱 Starting Sheltly Database Seeding...")

            # 1. CREATE USERS
            admin = await get_or_create_user(session, "admin@sheltly.com", {
                "id": uuid4(),
                "first_name": "Evidence",
                "last_name": "Admin",
                "password_hash": get_password_hash("Admin@12345"),
                "phone": "+2348011112222",
                "role": UserRole.ADMIN,
                "is_verified": True
            })

            lister = await get_or_create_user(session, "agency@blueapril.com", {
                "id": uuid4(),
                "first_name": "Blueapril",
                "last_name": "Properties",
                "password_hash": get_password_hash("Lister@12345"),
                "phone": "+2348022223333",
                "role": UserRole.LISTER,
                "is_verified": True
            })

            seeker = await get_or_create_user(session, "emmanuel@student.com", {
                "id": uuid4(),
                "first_name": "Emmanuel",
                "last_name": "Oboroghene",
                "password_hash": get_password_hash("Seeker@12345"),
                "phone": "+2348033334444",
                "role": UserRole.SEEKER,
                "is_verified": True
            })

            # 2. PROPERTY DATASET
            properties_data = [
                {
                    "title": "Luxury 3 Bedroom Apartment with Ocean View",
                    "description": "High-end apartment in Eko Atlantic. Features automated lighting, smart climate control, and a world-class gym. Perfect for corporate tenants seeking serenity and high-tech security.",
                    "price": 18000000.0, 
                    "price_type": PriceType.RENT,
                    "location": "Eko Atlantic, Victoria Island, Lagos",
                    "latitude": 6.4281, "longitude": 3.4215,
                    "bedrooms": 3, "bathrooms": 3,
                    "property_type": PropertyType.APARTMENT,
                    "amenities": ["24/7 Power", "Elevator", "Pool", "Smart Home", "Security"],
                    "status": PropertyStatus.APPROVED,
                },
                {
                    "title": "Modern 4 Bedroom Semi-Detached Duplex",
                    "description": "Brand new duplex in a gated estate. Fitted kitchen with gas cooker and heat extractor. All rooms ensuite. Located in a flood-free zone with excellent drainage system.",
                    "price": 95000000.0, 
                    "price_type": PriceType.SALE,
                    "location": "Orchid Road, Lekki, Lagos",
                    "latitude": 6.4589, "longitude": 3.5120,
                    "bedrooms": 4, "bathrooms": 4,
                    "property_type": PropertyType.HOUSE,
                    "amenities": ["CCTV", "Fitted Kitchen", "BQ", "Parking"],
                    "status": PropertyStatus.APPROVED,
                },
                {
                    "title": "Self-Contained Studio near Lead City University",
                    "description": "Clean and affordable studio apartment. Walking distance to the university main gate. Constant water supply and gated compound for student safety.",
                    "price": 500000.0, 
                    "price_type": PriceType.RENT,
                    "location": "Toll Gate Area, Ibadan",
                    "latitude": 7.3575, "longitude": 3.9270,
                    "bedrooms": 1, "bathrooms": 1,
                    "property_type": PropertyType.STUDIO,
                    "amenities": ["Water", "Gated", "Near Campus", "Prepaid Meter"],
                    "status": PropertyStatus.APPROVED,
                },
                {
                    "title": "Exquisite 5 Bedroom Mansion with Lagoon View",
                    "description": "A masterpiece in Banana Island. Features a private cinema, rooftop terrace, and infinity pool. Fully automated smart home with biometric access control.",
                    "price": 550000000.0, 
                    "price_type": PriceType.SALE,
                    "location": "Banana Island, Ikoyi, Lagos",
                    "latitude": 6.4631, "longitude": 3.4560,
                    "bedrooms": 5, "bathrooms": 6,
                    "property_type": PropertyType.HOUSE,
                    "amenities": ["Cinema", "Infinity Pool", "BQ", "Biometric Access"],
                    "status": PropertyStatus.APPROVED,
                },
                {
                    "title": "Premium Office Space in Central Business District",
                    "description": "Open-plan commercial space ideal for tech startups or corporate firms. Includes high-speed fiber internet, conference rooms, and 24/7 backup power.",
                    "price": 3500000.0, 
                    "price_type": PriceType.RENT,
                    "location": "Maitama, Abuja",
                    "latitude": 9.0765, "longitude": 7.4985,
                    "bedrooms": 0, "bathrooms": 4,
                    "property_type": PropertyType.COMMERCIAL,
                    "amenities": ["Fiber Internet", "Conference Room", "Cafe", "Generator"],
                    "status": PropertyStatus.APPROVED,
                },
                {
                    "title": "Cozy 2 Bedroom Flat for Young Professionals",
                    "description": "Tiled floors, well-ventilated rooms, and a large kitchen. Located in a quiet, decent neighborhood with easy access to the Island via Third Mainland Bridge.",
                    "price": 1500000.0, 
                    "price_type": PriceType.RENT,
                    "location": "Gbagada Phase 2, Lagos",
                    "latitude": 6.5567, "longitude": 3.3914,
                    "bedrooms": 2, "bathrooms": 2,
                    "property_type": PropertyType.APARTMENT,
                    "amenities": ["Security", "Clean Water", "Parking", "Tiled"],
                    "status": PropertyStatus.APPROVED,
                },
                {
                    "title": "3 Bedroom Bungalow with Large Backyard",
                    "description": "Spacious family bungalow in a serene estate. Features fruit trees in the backyard and enough space for a kitchen garden. Very peaceful environment.",
                    "price": 45000000.0, 
                    "price_type": PriceType.SALE,
                    "location": "Akala Express, Ibadan",
                    "latitude": 7.3167, "longitude": 3.8333,
                    "bedrooms": 3, "bathrooms": 3,
                    "property_type": PropertyType.HOUSE,
                    "amenities": ["Garden", "Fence", "Borehole", "Security"],
                    "status": PropertyStatus.APPROVED,
                }
            ]

            logger.info(f"Syncing {len(properties_data)} properties...")
            for p_data in properties_data:
                # Check for existing property by title
                stmt = select(Property).where(Property.title == p_data["title"])
                existing_p = (await session.execute(stmt)).scalar_one_or_none()
                
                if not existing_p:
                    prop = Property(id=uuid4(), lister_id=lister.id, **p_data)
                    session.add(prop)

            # 3. CREATE SAVED SEARCHES
            search_stmt = select(SavedSearch).where(SavedSearch.user_id == seeker.id)
            existing_search = (await session.execute(search_stmt)).scalar_one_or_none()

            if not existing_search:
                s_search = SavedSearch(
                    id=uuid4(),
                    user_id=seeker.id,
                    query="Modern student apartments near LCU",
                    filters={"price_max": 600000, "property_type": "STUDIO"},
                    notify_on_match=True
                )
                session.add(s_search)

            await session.commit()
            logger.success("✅ Seeding completed successfully!")

        except Exception as e:
            logger.error(f"❌ Error during seeding: {str(e)}")
            await session.rollback()
            raise
        finally:
            await engine.dispose()

if __name__ == "__main__":
    asyncio.run(seed_database())