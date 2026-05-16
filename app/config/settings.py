from pydantic_settings import BaseSettings                                                           
                                                                                                    
class Settings(BaseSettings):                                                                        
    app_name: str = "VeriKYC CV Service"                                                             
    model_path: str = "/models/efficientnet_b0.onnx"                                           
    allowed_types: list = ["image/jpeg", "image/png"]
    max_file_size: int = 10 * 1024 * 1024  # 10MB

settings = Settings()