from pydantic import BaseModel


class Detection(BaseModel):
    timestamp: str
    detected: bool
    video_path: str
