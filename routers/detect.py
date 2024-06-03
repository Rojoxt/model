from fastapi import APIRouter, UploadFile, File
from services.detection_service import process_video

router = APIRouter()


@router.post("/detect")
async def detect_weapon(file: UploadFile = File(...)):
    result = await process_video(file)
    return result
