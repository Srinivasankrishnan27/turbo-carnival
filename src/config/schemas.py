from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str
    base_url: str
    model: str
    api_key_env: str
    temperature: float = Field(ge=0, le=1)
    timeout: int
    stream: bool = False


class EmbeddingConfig(BaseModel):
    model: str


class NLIConfig(BaseModel):
    model: str


class RuntimeConfig(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig
    nli: NLIConfig
