"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="HireSense AI",
    description="AI-powered semantic candidate ranking system.",
    version="1.0.0",
)
app.include_router(router)
