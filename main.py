from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError 
from core.config import settings
from api.v1 import auth, users, properties, media, search, ai, admin
from fastapi.responses import JSONResponse
from core.logger import setup_logging
import cloudinary


setup_logging()

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True
)

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
    swagger_ui_init_oauth={},  # Disable OAuth default behavior
)

# Add Bearer token security scheme documentation
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=app.description,
        routes=app.routes,
    )
    
    # Add Bearer token security scheme
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}
    
    openapi_schema["components"]["securitySchemes"]["HTTPBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }
    
    # Apply to all endpoints (let HTTPBearer dependency handle the actual auth)
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if isinstance(operation, dict) and "responses" in operation:
                # Mark endpoints that likely need auth (not root, health, login, register, verify, etc)
                endpoint_summary = operation.get("summary", "").lower()
                endpoint_tags = operation.get("tags", [])
                
                # Don't require auth for public endpoints
                public_endpoints = {"root", "health", "login", "register", "verify", "forgot", "reset", "refresh", "token", "debug"}
                is_public = any(term in endpoint_summary for term in public_endpoints) or "authentication" in [t.lower() for t in endpoint_tags]
                
                if not is_public and operation.get("security") is None:
                    operation["security"] = [{"HTTPBearer": []}]
                    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "data": None
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    
    error_details = {err["loc"][-1]: err["msg"] for err in errors}

    raw_msg = errors[0].get("msg", "Validation error")
    friendly_msg = raw_msg.replace("Value error, ", "")

    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": friendly_msg, 
            "data": error_details
        }
    )

@app.get("/", tags=["General"])
async def root():
    return {
        "app": settings.PROJECT_NAME,
        "status": "active",
        "docs": "/docs"
    }

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(properties.router, prefix="/api/v1/properties", tags=["Properties"])
app.include_router(media.router, prefix="/api/v1/media", tags=["Media"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(ai.router, prefix="/api/v1/ai", tags=["AI/Ranking"])
app.include_router(admin.router, tags=["Admin"])

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "engine": "running"}