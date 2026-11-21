"""HTTP API и страница. Запуск из корня проекта:

    .venv/bin/python -m uvicorn powerplant.api:create_app --factory --port 8001
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse

from powerplant import config
from powerplant.artifact import Bundle, load_bundle
from powerplant.schemas import BatchResponse, MetaResponse, PredictRequest, PredictResponse
from powerplant.service import BatchError, ForecastService

PAGE_PATH = Path(__file__).resolve().parent / "web" / "index.html"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def get_service(request: Request) -> ForecastService:
    return request.app.state.service


ServiceDep = Annotated[ForecastService, Depends(get_service)]


def create_app(bundle: Bundle | None = None, bundle_path: Path = config.BUNDLE_PATH) -> FastAPI:
    """Собирает приложение. Модель грузится один раз при старте."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = ForecastService(bundle or load_bundle(bundle_path))
        yield

    app = FastAPI(title="Эталон мощности ПГУ", version="1.0", lifespan=lifespan)
    page_html = PAGE_PATH.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return page_html

    @app.get("/health")
    def health(service: ServiceDep) -> dict[str, str]:
        return {"status": "ok", "model_version": service.bundle.version}

    @app.get("/api/meta", response_model=MetaResponse)
    def meta(service: ServiceDep) -> MetaResponse:
        return service.meta()

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(request: PredictRequest, service: ServiceDep) -> PredictResponse:
        return service.predict(request)

    @app.get("/api/batch/example", response_class=PlainTextResponse)
    def batch_example(service: ServiceDep) -> PlainTextResponse:
        return PlainTextResponse(service.example_csv(), media_type="text/csv; charset=utf-8")

    @app.post("/api/batch", response_model=BatchResponse)
    def batch(service: ServiceDep, file: Annotated[UploadFile, File()]) -> BatchResponse:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Файл больше 5 МБ")
        try:
            return service.batch(file.filename or "", content)
        except BatchError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return app
