from pydantic import BaseModel, Field, confloat

class LLMJudgeEvaluatorOuput(BaseModel):
    score: confloat(ge=0.0, le=1.0) = Field(default=0.0, description="Dimension score")
    comment: str = Field(default="", description="Comment")