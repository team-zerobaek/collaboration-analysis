from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware
from starlette.responses import RedirectResponse
from dashboard.app import app as dash_app

app = FastAPI()

app.mount("/dash", WSGIMiddleware(dash_app.server))

@app.get("/")
def read_root():
    return {"message": "Welcome to the SNA Dashboard"}

@app.get("/dash")
def redirect_to_dash():
    return RedirectResponse(url="/dash/")
