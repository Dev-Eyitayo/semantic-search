from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError 
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from api.v1 import auth, users, properties, media, search, ai, admin
from fastapi.responses import JSONResponse
from core.logger import setup_logging
import cloudinary
from loguru import logger


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
    description="Backend for AI Driven Semantic Search Property Platform",
    docs_url="/", 
    redoc_url="/redoc",
    swagger_ui_init_oauth={},
)




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

    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}
    
    openapi_schema["components"]["securitySchemes"]["HTTPBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }

    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if isinstance(operation, dict) and "responses" in operation:
                endpoint_summary = operation.get("summary", "").lower()
                endpoint_tags = operation.get("tags", [])

                public_endpoints = {"root", "health", "login", "register", "verify", "forgot", "reset", "refresh", "token", "debug"}
                is_public = any(term in endpoint_summary for term in public_endpoints) or "authentication" in [t.lower() for t in endpoint_tags]
                
                if not is_public and operation.get("security") is None:
                    operation["security"] = [{"HTTPBearer": []}]
                    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Application startup: Checking AI Configuration...")

    try:
        from services.embedding_service import get_embedding_model
        _ = get_embedding_model(warmup=False) 
        logger.success("✓ Embedding service initialized")

    except Exception as e:
        logger.warning(f"AI models will load on demand: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown.
    """
    logger.info("🛑 Application shutdown: Cleaning up resources...")
    
    try:
        from services.embedding_service import get_cache_stats, clear_embedding_cache

        cache_info = get_cache_stats()
        if cache_info['cache_size'] > 0:
            cleared = clear_embedding_cache()
            logger.info(f"✓ Cleared {cleared} cached embeddings")
        
        logger.success("✓ Cleanup completed")
    
    except Exception as e:
        logger.warning(f"Shutdown cleanup warning: {e}")


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


origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://sheltly.vercel.app"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,  
    allow_methods=["*"],    
    allow_headers=["*"],   
)


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