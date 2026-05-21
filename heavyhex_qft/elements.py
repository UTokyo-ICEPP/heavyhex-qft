from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class BaseElement:
    """Base class for lattice elements."""
    @classmethod
    def from_json(cls, data: str) -> Any:
        return cls.from_dict({key: json.loads(value) for key, value in json.loads(data).items()})


@dataclass
class Vertex(BaseElement):
    """Vertex data."""
    id: int
    position: tuple[float, float]
    plaquettes: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f'V:{self.id}'

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'position': list(self.position),
            'plaquettes': list(self.plaquettes)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Vertex':
        return cls(
            id=data['id'],
            position=tuple(data['position']),
            plaquettes=set(data['plaquettes'])
        )


@dataclass
class Link(BaseElement):
    """Link data."""
    id: int
    position: tuple[float, float]

    @property
    def label(self) -> str:
        return f'L:{self.id}'

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'position': list(self.position)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Link':
        return cls(
            id=data['id'],
            position=tuple(data['position'])
        )


@dataclass
class Plaquette(BaseElement):
    """Plaquette data."""
    id: int
    position: tuple[float, float]
    vertices: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f'P:{self.id}'

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'position': list(self.position),
            'vertices': list(self.vertices)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Plaquette':
        return cls(
            id=data['id'],
            position=tuple(data['position']),
            vertices=set(data['vertices'])
        )


@dataclass
class DummyPlaquette(BaseElement):
    """Dummy plaquette for dual graph."""
    position: tuple[float, float]
    vertices: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f''

    def to_dict(self) -> dict[str, Any]:
        return {
            'position': list(self.position),
            'vertices': list(self.vertices)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DummyPlaquette':
        return cls(
            position=tuple(data['position']),
            vertices=set(data['vertices'])
        )


@dataclass
class Ancilla(BaseElement):
    """Ancilla qubit."""

    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Ancilla':
        return cls()
