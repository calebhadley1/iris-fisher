from fastapi import FastAPI
from iris_fisher.api.routers import iris_router


app = FastAPI()
app.include_router(iris_router.router)
