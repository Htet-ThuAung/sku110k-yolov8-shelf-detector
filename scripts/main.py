from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.api import router as shelf_router
import csv
import os

app = FastAPI(title="Shelf Detector API")

# Mount static folder for images
app.mount(
    "/static/annotated",
    StaticFiles(directory="outputs/inference_api_results/annotated"),
    name="static-annotated",
)

# Set up template engine
templates = Jinja2Templates(directory="templates")


# Route for HTML upload form
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    csv_path = "outputs/inference_api_results/csvfile/results.csv"
    results = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            results = list(reader)

    return templates.TemplateResponse(
        "index.html", {"request": request, "results": results}
    )


# Include existing API
app.include_router(shelf_router)
