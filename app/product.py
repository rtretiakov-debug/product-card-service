from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

NO_DATA = "«данных нет»"


class ProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    manufacturer: str
    description: str
    specs: dict[str, str] | Literal["«данных нет»"] = NO_DATA

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("manufacturer")
    @classmethod
    def empty_manufacturer_is_unknown(cls, value: str) -> str:
        return NO_DATA if not value.strip() else value

    @field_validator("specs", mode="before")
    @classmethod
    def empty_specs_are_unknown(cls, value: object) -> object:
        if isinstance(value, dict):
            if not value:
                return NO_DATA
            return {
                key: NO_DATA if isinstance(spec, str) and not spec.strip() else spec
                for key, spec in value.items()
            }
        return value
