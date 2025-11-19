from fastapi import FastAPI
import uvicorn

from src.utils.setup_logging import my_logger_setup, get_my_logger
from src.routers.system_router import router as system_router

my_logger_setup()
logger = get_my_logger()
logger.info("Starting the application")

app = FastAPI()

app.include_router(system_router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7009
    )
