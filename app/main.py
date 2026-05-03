                                                                                                    
from fastapi import FastAPI                                                                          
from app.api.routes import health, classify                                                          
                                                                                                    
app = FastAPI(title="VeriKYC CV Service")                                                         

app.include_router(health.router, prefix="/api/v1")
app.include_router(classify.router, prefix="/api/v1")