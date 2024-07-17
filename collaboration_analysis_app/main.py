from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn
from upload.app import upload_app
from behavioral.app import fastapi_app as dashboard_app
from subjective.app import fastapi_app as subjective_app
from abtest.app import fastapi_app as abtest_app
from ml.app import fastapi_app as ml_app

# Initialize the main FastAPI app
main_app = FastAPI()

# Mount the sub-apps to different paths
main_app.mount("/upload", upload_app)
main_app.mount("/dash", dashboard_app)
main_app.mount("/subjective", subjective_app)
main_app.mount("/abtest", abtest_app)
main_app.mount("/ml", ml_app)

@main_app.get("/")
async def root():
    return RedirectResponse(url="/upload")

if __name__ == "__main__":
    uvicorn.run(main_app, host="0.0.0.0", port=8080)
