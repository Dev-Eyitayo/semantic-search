from fastapi import FastAPI
from core.config import settings
from api.v1 import leads

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    Sheltly AI Neural Engine 
    The backend core for AI-verified real estate listings and semantic search.
    
    Verify Listings using AI cross-referencing.
    Semantic Search for intuitive property discovery.
    Secure Messaging between tenants and landlords.
    """,
    docs_url="/docs", 
    redoc_url="/redoc",
)

@app.get("/", tags=["General"])
async def root():
    return {
        "app": settings.PROJECT_NAME,
        "status": "active",
        "docs": "/api/docs"
    }

app.include_router(leads.router, prefix="/api/v1/leads", tags=["Leads"])

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "engine": "running"}