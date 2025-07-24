from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# images
class Image(BaseModel):
    image_id: Optional[int] = None
    image_url: str
    image_source: str
    created_at: datetime

