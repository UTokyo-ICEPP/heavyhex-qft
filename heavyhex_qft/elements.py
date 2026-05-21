import json
from dataclasses import dataclass, field


@dataclass
class Vertex:
    """Vertex data."""
    id: int
    position: tuple[float, float]
    plaquettes: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f'V:{self.id}'

    def to_json_data(self):
        return {
            'id': json.dumps(self.id),
            'position': json.dumps(list(self.position)),
            'plaquettes': json.dumps(list(self.plaquettes))
        }

    @classmethod
    def from_json_data(cls, data):
        return Vertex(
            id=json.loads(data['id']),
            position=tuple(json.loads(data['position'])),
            plaquettes=set(json.loads(data['plaquettes']))
        )


@dataclass
class Link:
    """Link data."""
    id: int
    position: tuple[float, float]

    @property
    def label(self) -> str:
        return f'L:{self.id}'

    def to_json_data(self):
        return {
            'id': json.dumps(self.id),
            'position': json.dumps(list(self.position))
        }

    @classmethod
    def from_json_data(cls, data):
        return Link(
            id=json.loads(data['id']),
            position=tuple(json.loads(data['position']))
        )


@dataclass
class Plaquette:
    """Plaquette data."""
    id: int
    position: tuple[float, float]
    vertices: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f'P:{self.id}'

    def to_json_data(self):
        return {
            'id': json.dumps(self.id),
            'position': json.dumps(list(self.position)),
            'vertices': json.dumps(list(self.vertices))
        }

    @classmethod
    def from_json_data(cls, data):
        return Plaquette(
            id=json.loads(data['id']),
            position=tuple(json.loads(data['position'])),
            vertices=set(json.loads(data['vertices']))
        )


@dataclass
class DummyPlaquette:
    """Dummy plaquette for dual graph."""
    position: tuple[float, float]
    vertices: set[int] = field(default_factory=set)

    @property
    def label(self) -> str:
        return f''

    def to_json_data(self):
        return {
            'position': json.dumps(list(self.position)),
            'vertices': json.dumps(list(self.vertices))
        }

    @classmethod
    def from_json_data(cls, data):
        return DummyPlaquette(
            position=tuple(json.loads(data['position'])),
            vertices=set(json.loads(data['vertices']))
        )


@dataclass
class Ancilla:
    """Ancilla qubit."""

    def to_json_data(self):
        return {}

    @classmethod
    def from_json_data(cls, data):
        return Ancilla()