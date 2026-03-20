from fastapi import FastAPI

app = FastAPI(title="Sheltly AI Backend")

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Sheltly AI Neural Engine is running",
        "version": "1.0.0"
    }