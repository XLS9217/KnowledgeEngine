from fastapi import FastAPI
import uvicorn
import os

from src.utils.setup_logging import my_logger_setup, get_my_logger
from src.routers.system_router import router as system_router
from src.routers.service_router import router as service_router
from src.task_orchestrator.orchestrator_interface import OrchestratorInterface

my_logger_setup()
logger = get_my_logger()
logger.info("Starting the application")

# Initialize the orchestrator engine
logger.info("Initializing OrchestratorInterface with single_process_engine")
OrchestratorInterface.initialize("single_process_engine")
logger.info("OrchestratorInterface initialized successfully")

app = FastAPI()

app.include_router(system_router)
app.include_router(service_router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7009))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
