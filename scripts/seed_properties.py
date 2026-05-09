import asyncio
import uuid
import random
import time as time_module
import httpx
from datetime import datetime, timezone, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy import delete

from db.models.user import User
from db.models.property import Property
from core.config import settings
from core.security import get_password_hash
from core.enums import UserRole, PropertyType, PriceType, PropertyStatus
from services.embedding_service import generate_embeddings_batch
from core.config import settings

# --- CONFIGURATION ---
UNSPLASH_ACCESS_KEY = settings.UNSPLASH_ACCESS_KEY
DEFAULT_PASSWORD = "Pass12345@"

NIGERIAN_LOCATIONS = [
    {"name": "Toll Gate (Lead City), Ibadan", "lat": 7.3575, "lng": 3.9270, "type": "student"},
    {"name": "Agbowo (UI), Ibadan", "lat": 7.4431, "lng": 3.9003, "type": "student"},
    {"name": "Bodija, Ibadan", "lat": 7.4200, "lng": 3.9100, "type": "urban"},
    {"name": "Yaba (Unilag), Lagos", "lat": 6.5244, "lng": 3.3792, "type": "student"},
    {"name": "Lekki Phase 1, Lagos", "lat": 6.4269, "lng": 3.5789, "type": "luxury"},
    {"name": "Banana Island, Lagos", "lat": 6.4631, "lng": 3.4560, "type": "luxury"},
    {"name": "Victoria Island, Lagos", "lat": 6.4281, "lng": 3.4192, "type": "luxury"},
    {"name": "Surulere, Lagos", "lat": 6.4917, "lng": 3.3608, "type": "urban"},
    {"name": "Ikeja GRA, Lagos", "lat": 6.5833, "lng": 3.3333, "type": "urban"},
    {"name": "Maitama, Abuja", "lat": 9.0833, "lng": 7.4000, "type": "luxury"},
    {"name": "Wuse 2, Abuja", "lat": 9.0667, "lng": 7.4500, "type": "urban"},
]

PROPERTY_DESCRIPTIONS = [
    "Clean and secured self-contained apartment. Constant water supply and tiled floors. Very close to the campus gate.",
    "Executive apartment in a gated community. Features modern finishes, automated lighting, and 24/7 power supply.",
    "Newly built student hostel with ensuite rooms, wardrobe, and kitchen cabinets. Secured environment with CCTV.",
    "Luxury mansion with private pool and lagoon view. Features high-end marble floors and smart home systems.",
    "Spacious 3-bedroom flat for young professionals. Well ventilated with ample parking space and tight security.",
]

AMENITIES = [
    "Swimming Pool", "Gym", "Security 24/7", "Backup Generator", "Water Supply",
    "Parking Space", "Garden", "Air Conditioning", "Fitted Kitchen", "Prepaid Meter"
]

async def fetch_unsplash_images(query: str, count: int = 5):
    url = f"https://api.unsplash.com/search/photos?query={query}&per_page={count}&client_id={UNSPLASH_ACCESS_KEY}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return [img["urls"]["regular"] for img in data["results"]]
        except Exception as e:
            print(f"Unsplash API error: {e}")
    return [f"https://placehold.co/800x600?text={query.replace(' ', '+')}"]

async def clear_existing_data(session: AsyncSession):
    print("🗑️ Wiping existing properties and lister accounts...")
    await session.execute(delete(Property))
    await session.execute(delete(User).where(User.role == UserRole.LISTER))
    await session.commit()

async def create_seed_listers(session: AsyncSession) -> list[uuid.UUID]:
    print(f"👤 Creating lister accounts (Password: {DEFAULT_PASSWORD})...")
    lister_names = [
        ("Tunde", "Adeyemi"), ("Adaeze", "Okoro"), ("Emeka", "Okafor"),
        ("UI", "Rentals"), ("LeadCity", "Hostels"), ("Lagos", "Luxury")
    ]
    
    lister_ids = []
    pw_hash = get_password_hash(DEFAULT_PASSWORD)
    
    for first, last in lister_names:
        email = f"{first.lower()}.{last.lower()}@sheltly.com"
        user = User(
            id=uuid.uuid4(),
            email=email,
            password_hash=pw_hash,
            first_name=first,
            last_name=last,
            phone=f"+23480{random.randint(10000000, 99999999)}",
            role=UserRole.LISTER,
            is_verified=True,
            created_at=datetime.now(timezone.utc)
        )
        session.add(user)
        lister_ids.append(user.id)
    
    await session.commit()
    return lister_ids

async def create_seed_properties(session: AsyncSession, lister_ids: list[uuid.UUID]):
    print("🏡 Generating 200 versatile properties...")
    
    # Pre-fetch image pools to avoid rate limits/latency per item
    image_pools = {
        "student": await fetch_unsplash_images("dormitory room hostel", 30),
        "luxury": await fetch_unsplash_images("modern luxury mansion", 30),
        "apartment": await fetch_unsplash_images("apartment interior building", 30),
        "urban": await fetch_unsplash_images("modern house nigeria", 30)
    }

    properties = []
    approved_indices = []

    for i in range(200):
        loc = random.choice(NIGERIAN_LOCATIONS)
        
        if loc["type"] == "student":
            bedrooms = random.choice([0, 1, 1, 2])
            prop_type = random.choice([PropertyType.STUDIO, PropertyType.APARTMENT])
            price = random.uniform(180_000, 750_000) if prop_type == PropertyType.STUDIO else random.uniform(500_000, 1_200_000)
            price_type = PriceType.RENT
            img_pool = image_pools["student"]
        elif loc["type"] == "luxury":
            bedrooms = random.choice([3, 4, 5, 6])
            prop_type = random.choice([PropertyType.HOUSE, PropertyType.DUPLEX])
            price = random.uniform(80_000_000, 850_000_000)
            price_type = PriceType.SALE
            img_pool = image_pools["luxury"]
        else:
            bedrooms = random.choice([2, 3, 4])
            prop_type = random.choice([PropertyType.APARTMENT, PropertyType.HOUSE])
            price = random.uniform(1_500_000, 5_000_000)
            price_type = PriceType.RENT
            img_pool = image_pools["urban"]

        title_prefix = "Self-contained" if bedrooms == 0 else f"{bedrooms}-Bedroom"
        title = f"{title_prefix} {prop_type.value} in {loc['name']}"
        
        # Select images from pool
        item_images = random.sample(img_pool, min(3, len(img_pool)))
        
        prop = Property(
            id=uuid.uuid4(),
            lister_id=random.choice(lister_ids),
            title=title,
            description=random.choice(PROPERTY_DESCRIPTIONS),
            price=price,
            price_type=price_type,
            location=loc["name"],
            latitude=loc["lat"] + random.uniform(-0.002, 0.002),
            longitude=loc["lng"] + random.uniform(-0.002, 0.002),
            bedrooms=bedrooms if bedrooms > 0 else 1,
            bathrooms=max(1, bedrooms),
            property_type=prop_type,
            amenities=random.sample(AMENITIES, random.randint(3, 7)),
            status=random.choice([PropertyStatus.APPROVED, PropertyStatus.APPROVED, PropertyStatus.PENDING_REVIEW]),
            thumbnail=item_images[0],
            images=item_images,
            created_at=datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
        )
        
        if prop.status == PropertyStatus.APPROVED:
            approved_indices.append(len(properties))
        
        properties.append(prop)
        session.add(prop)

    await session.commit()
    
    if approved_indices:
        print(f"🧠 Generating embeddings for {len(approved_indices)} properties...")
        approved_props = [properties[idx] for idx in approved_indices]
        texts = [f"{p.title}. {p.description}" for p in approved_props]
        try:
            embeddings, _ = generate_embeddings_batch(texts=texts, normalize=True)
            for p, emb in zip(approved_props, embeddings):
                p.embedding = emb
            await session.commit()
        except Exception as e:
            print(f"Embedding error: {e}")

async def main():
    engine = create_async_engine(settings.DATABASE_ASYNC_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as session:
        try:
            await clear_existing_data(session)
            lister_ids = await create_seed_listers(session)
            await create_seed_properties(session, lister_ids)
            print("\n✅ Database re-seeded with high-quality static images!")
            print("   Password: Pass12345@")
        except Exception as e:
            print(f"❌ Error: {e}")
            await session.rollback()
        finally:
            await engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())