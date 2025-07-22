from fastapi import FastAPI
from app.database import Base, engine
from app.api.endpoints import user,recommendation, spotify_feature_extract    # Import your API routes

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Include API routes
app.include_router(user.router)
app.include_router(recommendation.router)
app.include_router(spotify_feature_extract.router)
