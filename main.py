from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError 
from core.config import settings
from api.v1 import auth
from fastapi.responses import JSONResponse


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
        "docs": "/api/docs"
    }

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "engine": "running"}