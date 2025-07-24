from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# images
class Image(BaseModel):
    image_id: Optional[int] = None
    image_url: str
    image_source: str
    created_at: datetime

# menus
class Menu(BaseModel):
    menu_id: Optional[int] = None
    menu_name: str
    created_at: datetime
    updated_at: datetime

# ingredients
class Ingredient(BaseModel):
    ingredient_id: Optional[int] = None
    ingredient_name: str
    ingredient_category: str
    created_at: datetime

# analysis_results
class AnalysisResult(BaseModel):
    result_id: Optional[int] = None
    image_id: int
    result_type: str
    detected_id: int
    confidence_score: float 
    created_at: datetime

# menu_ingredients
class MenuIngredient(BaseModel):
    menu_ingredient_id: Optional[int] = None
    menu_id: int
    ingredient_id: int
    amount : int
    unit : str
    created_at: datetime

# menu_recipes
class MenuRecipe(BaseModel):
    menu_recipe_id: Optional[int] = None
    menu_id: int
    cooking_steps: str
    cooking_info : str
    flavor_characteristics : str
    nutrition_info : str
    serving_tips : str
    cooking_tiem_minutes : int
    difficulty_level : str
    servings : int
    created_at: datetime
    updated_at: datetime

# brands
class Brand(BaseModel):
    brand_id: Optional[int] = None
    brand_name: str