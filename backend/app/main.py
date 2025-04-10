from fastapi import FastAPI
from app.database import Base, engine
from app.api.endpoints import user    # Import your API routes

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Include API routes
app.include_router(user.router)