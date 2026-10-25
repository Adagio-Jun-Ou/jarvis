"""Four-Dimensional Imagination Space for Jarvis.

This module provides a sophisticated spatial-temporal reasoning system that enables
Jarvis to imagine, simulate, and reason about objects and scenarios in 4D space-time.

Key capabilities:
- 3D spatial representation with object tracking
- Temporal simulation (4th dimension)
- Physics-based prediction
- Scene composition and manipulation
- Trajectory analysis and forecasting
"""
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from core.helpers import atomic_write_json, read_json_with_integrity


class Dimension(Enum):
    """Dimensions in 4D space."""
    X = 0  # Left-Right
    Y = 1  # Up-Down
    Z = 2  # Forward-Backward
    T = 3  # Time


@dataclass
class Vector3:
    """3D vector with operations."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'Vector3':
        mag = self.magnitude()
        if mag > 0:
            return self / mag
        return Vector3()

    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3') -> 'Vector3':
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def to_tuple(self) -> tuple:
        return (self.x, self.y, self.z)

    def to_dict(self) -> dict:
        return {'x': self.x, 'y': self.y, 'z': self.z}

    @classmethod
    def from_dict(cls, data: dict) -> 'Vector3':
        return cls(data.get('x', 0), data.get('y', 0), data.get('z', 0))


@dataclass
class Quaternion:
    """Quaternion for 3D rotation."""
    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(
            w=self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z,
            x=self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y,
            y=self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x,
            z=self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        )

    def rotate_vector(self, v: Vector3) -> Vector3:
        """Rotate a vector by this quaternion."""
        qv = Quaternion(0, v.x, v.y, v.z)
        conjugate = Quaternion(self.w, -self.x, -self.y, -self.z)
        result = self * qv * conjugate
        return Vector3(result.x, result.y, result.z)

    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float) -> 'Quaternion':
        """Create quaternion from Euler angles (in radians)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return Quaternion(
            w=cr * cp * cy + sr * sp * sy,
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy
        )


@dataclass
class Transform:
    """3D transform with position, rotation, and scale."""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)
    scale: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))

    def to_dict(self) -> dict:
        return {
            'position': self.position.to_dict(),
            'rotation': {'w': self.rotation.w, 'x': self.rotation.x, 'y': self.rotation.y, 'z': self.rotation.z},
            'scale': self.scale.to_dict()
        }


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    min: Vector3 = field(default_factory=Vector3)
    max: Vector3 = field(default_factory=Vector3)

    def contains(self, point: Vector3) -> bool:
        return (self.min.x <= point.x <= self.max.x and
                self.min.y <= point.y <= self.max.y and
                self.min.z <= point.z <= self.max.z)

    def intersects(self, other: 'BoundingBox') -> bool:
        return (self.min.x <= other.max.x and self.max.x >= other.min.x and
                self.min.y <= other.max.y and self.max.y >= other.min.y and
                self.min.z <= other.max.z and self.max.z >= other.min.z)

    def center(self) -> Vector3:
        return (self.min + self.max) * 0.5

    def size(self) -> Vector3:
        return self.max - self.min


@dataclass
class PhysicsProperties:
    """Physical properties of an object."""
    mass: float = 1.0  # kg
    velocity: Vector3 = field(default_factory=Vector3)
    acceleration: Vector3 = field(default_factory=Vector3)
    angular_velocity: Vector3 = field(default_factory=Vector3)
    is_static: bool = False
    friction: float = 0.5
    restitution: float = 0.3  # Bounciness
    gravity_scale: float = 1.0


@dataclass
class ImagineObject:
    """An object in the 4D imagination space."""
    id: str
    name: str
    object_type: str  # "entity", "prop", "terrain", "effect"
    transform: Transform = field(default_factory=Transform)
    bounding_box: Optional[BoundingBox] = None
    physics: PhysicsProperties = field(default_factory=PhysicsProperties)
    properties: dict = field(default_factory=dict)
    children: list['ImagineObject'] = field(default_factory=list)
    time_created: datetime = field(default_factory=datetime.now)
    timeline: list[tuple[float, Transform]] = field(default_factory=list)  # (time, transform) history

    def record_state(self, time: float) -> None:
        """Record current state in timeline."""
        self.timeline.append((time, Transform(
            position=Vector3(self.transform.position.x, self.transform.position.y, self.transform.position.z),
            rotation=Quaternion(self.rotation.w, self.rotation.x, self.rotation.y, self.rotation.z),
            scale=Vector3(self.transform.scale.x, self.transform.scale.y, self.transform.scale.z)
        )))

    def get_state_at_time(self, time: float) -> Optional[Transform]:
        """Get interpolated state at a specific time."""
        if not self.timeline:
            return None

        # Find surrounding keyframes
        for i, (t, transform) in enumerate(self.timeline):
            if t >= time:
                if i == 0:
                    return transform
                # Interpolate
                prev_t, prev_transform = self.timeline[i-1]
                alpha = (time - prev_t) / (t - prev_t) if t != prev_t else 0
                return self._interpolate_transforms(prev_transform, transform, alpha)

        return self.timeline[-1][1] if self.timeline else None

    def _interpolate_transforms(self, a: Transform, b: Transform, t: float) -> Transform:
        """Linearly interpolate between two transforms."""
        return Transform(
            position=a.position + (b.position - a.position) * t,
            rotation=Quaternion(),  # Simplified - should use slerp
            scale=a.scale + (b.scale - a.scale) * t
        )

    @property
    def rotation(self):
        return self.transform.rotation


class SimulationState(Enum):
    """States of a simulation."""
    STOPPED = "stopped"
    PAUSED = "paused"
    RUNNING = "running"
    STEPPING = "stepping"


@dataclass
class ImagineScene:
    """A complete 4D scene with objects and timeline."""
    name: str
    objects: dict[str, ImagineObject] = field(default_factory=dict)
    time_scale: float = 1.0  # Speed of simulation
    current_time: float = 0.0  # Current simulation time in seconds
    start_time: datetime = field(default_factory=datetime.now)
    state: SimulationState = SimulationState.STOPPED
    gravity: Vector3 = field(default_factory=lambda: Vector3(0, -9.81, 0))
    bounds: Optional[BoundingBox] = None
    metadata: dict = field(default_factory=dict)

    def add_object(self, obj: ImagineObject) -> None:
        """Add an object to the scene."""
        self.objects[obj.id] = obj

    def remove_object(self, object_id: str) -> Optional[ImagineObject]:
        """Remove an object from the scene."""
        return self.objects.pop(object_id, None)

    def get_object(self, object_id: str) -> Optional[ImagineObject]:
        """Get an object by ID or name."""
        # Try by ID first
        if object_id in self.objects:
            return self.objects[object_id]
        # Try by name
        for obj in self.objects.values():
            if obj.name == object_id:
                return obj
        return None

    def find_objects_by_type(self, object_type: str) -> list[ImagineObject]:
        """Find all objects of a given type."""
        return [obj for obj in self.objects.values() if obj.object_type == object_type]

    def find_objects_in_radius(self, center: Vector3, radius: float) -> list[ImagineObject]:
        """Find all objects within a radius of a point."""
        return [
            obj for obj in self.objects.values()
            if (obj.transform.position - center).magnitude() <= radius
        ]

    def raycast(self, origin: Vector3, direction: Vector3, max_distance: float = 1000) -> Optional[tuple[ImagineObject, Vector3]]:
        """Cast a ray and find the first intersecting object."""
        direction = direction.normalized()
        closest_hit = None
        closest_dist = max_distance

        for obj in self.objects.values():
            if obj.bounding_box:
                # Simplified AABB raycast
                hit, dist = self._raycast_aabb(origin, direction, obj.bounding_box)
                if hit and dist < closest_dist:
                    closest_dist = dist
                    closest_hit = (obj, origin + direction * dist)

        return closest_hit

    def _raycast_aabb(self, origin: Vector3, direction: Vector3, box: BoundingBox) -> tuple[bool, float]:
        """Raycast against AABB."""
        tmin = float('-inf')
        tmax = float('inf')

        for i, (o, d, bmin, bmax) in enumerate([
            (origin.x, direction.x, box.min.x, box.max.x),
            (origin.y, direction.y, box.min.y, box.max.y),
            (origin.z, direction.z, box.min.z, box.max.z)
        ]):
            if abs(d) < 1e-8:
                if o < bmin or o > bmax:
                    return False, 0
            else:
                t1 = (bmin - o) / d
                t2 = (bmax - o) / d
                tmin = max(tmin, min(t1, t2))
                tmax = min(tmax, max(t1, t2))

        if tmax >= tmin and tmax >= 0:
            return True, max(tmin, 0)
        return False, 0


class ImaginationEngine:
    """The core 4D imagination engine for Jarvis.

    This engine enables Jarvis to:
    1. Create and manipulate 3D scenes
    2. Simulate physics and motion over time (4th dimension)
    3. Predict future states
    4. Reason about spatial relationships
    5. Imagine scenarios and outcomes
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "imagination"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.scenes: dict[str, ImagineScene] = {}
        self.active_scene: Optional[ImagineScene] = None

        # Simulation parameters
        self.physics_substeps = 4
        self.max_simulation_time = 3600  # 1 hour max

        # Object templates for common objects
        self.templates: dict[str, dict] = self._load_templates()

    def _load_templates(self) -> dict:
        """Load object templates."""
        return {
            "cube": {
                "object_type": "prop",
                "bounding_box": BoundingBox(
                    min=Vector3(-0.5, -0.5, -0.5),
                    max=Vector3(0.5, 0.5, 0.5)
                ),
                "properties": {"shape": "cube", "color": "gray"}
            },
            "sphere": {
                "object_type": "prop",
                "properties": {"shape": "sphere", "radius": 0.5, "color": "gray"}
            },
            "cylinder": {
                "object_type": "prop",
                "properties": {"shape": "cylinder", "radius": 0.5, "height": 1.0, "color": "gray"}
            },
            "humanoid": {
                "object_type": "entity",
                "bounding_box": BoundingBox(
                    min=Vector3(-0.3, 0, -0.2),
                    max=Vector3(0.3, 1.8, 0.2)
                ),
                "physics": PhysicsProperties(mass=70.0, friction=0.8),
                "properties": {"shape": "humanoid", "height": 1.8}
            },
            "vehicle": {
                "object_type": "entity",
                "bounding_box": BoundingBox(
                    min=Vector3(-2, 0, -1),
                    max=Vector3(2, 1.5, 1)
                ),
                "physics": PhysicsProperties(mass=1500.0),
                "properties": {"shape": "vehicle", "type": "car"}
            },
            "building": {
                "object_type": "terrain",
                "physics": PhysicsProperties(is_static=True),
                "properties": {"shape": "building"}
            },
        }

    def create_scene(self, name: str, **kwargs) -> ImagineScene:
        """Create a new 4D scene."""
        scene = ImagineScene(name=name, **kwargs)
        self.scenes[name] = scene
        self.active_scene = scene
        return scene

    def switch_scene(self, name: str) -> Optional[ImagineScene]:
        """Switch to a different scene."""
        if name in self.scenes:
            self.active_scene = self.scenes[name]
            return self.active_scene
        return None

    def create_object(
        self,
        name: str,
        template: Optional[str] = None,
        position: Optional[Vector3] = None,
        **kwargs
    ) -> ImagineObject:
        """Create a new object, optionally from a template."""
        import uuid

        obj_id = str(uuid.uuid4())[:8]

        if template and template in self.templates:
            template_data = self.templates[template].copy()
            kwargs = {**template_data, **kwargs}

        obj = ImagineObject(
            id=obj_id,
            name=name,
            object_type=kwargs.get("object_type", "prop"),
            transform=Transform(
                position=position or Vector3(),
                rotation=kwargs.get("rotation", Quaternion()),
                scale=kwargs.get("scale", Vector3(1, 1, 1))
            ),
            bounding_box=kwargs.get("bounding_box"),
            physics=kwargs.get("physics", PhysicsProperties()),
            properties=kwargs.get("properties", {})
        )

        if self.active_scene:
            self.active_scene.add_object(obj)

        return obj

    async def simulate(self, duration: float, time_step: float = 0.016) -> dict:
        """Run physics simulation for a duration.

        Args:
            duration: How long to simulate (in seconds)
            time_step: Time step per physics update (default: ~60 FPS)

        Returns:
            Simulation results
        """
        if not self.active_scene:
            return {"error": "No active scene"}

        scene = self.active_scene
        scene.state = SimulationState.RUNNING

        steps = int(duration / time_step)
        results = {
            "steps": steps,
            "duration": duration,
            "objects": {},
            "collisions": [],
            "events": []
        }

        for step in range(steps):
            dt = time_step * scene.time_scale
            scene.current_time += dt

            # Update physics for each object
            for obj_id, obj in scene.objects.items():
                if obj.physics.is_static:
                    continue

                # Apply gravity
                if obj.physics.gravity_scale != 0:
                    gravity = scene.gravity * obj.physics.gravity_scale * dt
                    obj.physics.velocity = obj.physics.velocity + gravity

                # Apply velocity
                obj.physics.velocity = obj.physics.velocity + obj.physics.acceleration * dt
                obj.transform.position = obj.transform.position + obj.physics.velocity * dt

                # Record state for timeline
                obj.record_state(scene.current_time)

                # Check bounds
                if scene.bounds:
                    if not scene.bounds.contains(obj.transform.position):
                        # Bounce or clamp
                        results["events"].append({
                            "type": "boundary_collision",
                            "object": obj_id,
                            "time": scene.current_time
                        })

            # Check collisions
            collisions = self._check_collisions(scene)
            for collision in collisions:
                self._resolve_collision(collision)
                results["collisions"].append({
                    "objects": collision,
                    "time": scene.current_time
                })

        scene.state = SimulationState.STOPPED

        # Collect final states
        for obj_id, obj in scene.objects.items():
            results["objects"][obj_id] = {
                "position": obj.transform.position.to_dict(),
                "velocity": obj.physics.velocity.to_dict(),
                "timeline_length": len(obj.timeline)
            }

        return results

    def _check_collisions(self, scene: ImagineScene) -> list[tuple[str, str]]:
        """Check for collisions between objects."""
        collisions = []
        objects = list(scene.objects.values())

        for i, obj_a in enumerate(objects):
            if not obj_a.bounding_box:
                continue
            for obj_b in objects[i+1:]:
                if not obj_b.bounding_box:
                    continue

                # Transform bounding boxes to world space
                box_a = BoundingBox(
                    min=obj_a.bounding_box.min + obj_a.transform.position,
                    max=obj_a.bounding_box.max + obj_a.transform.position
                )
                box_b = BoundingBox(
                    min=obj_b.bounding_box.min + obj_b.transform.position,
                    max=obj_b.bounding_box.max + obj_b.transform.position
                )

                if box_a.intersects(box_b):
                    collisions.append((obj_a.id, obj_b.id))

        return collisions

    def _resolve_collision(self, collision: tuple[str, str]) -> None:
        """Resolve a collision between two objects."""
        if not self.active_scene:
            return

        obj_a = self.active_scene.objects.get(collision[0])
        obj_b = self.active_scene.objects.get(collision[1])

        if not obj_a or not obj_b:
            return

        # Simple collision response - reverse velocities
        if not obj_a.physics.is_static:
            obj_a.physics.velocity = obj_a.physics.velocity * -obj_a.physics.restitution
        if not obj_b.physics.is_static:
            obj_b.physics.velocity = obj_b.physics.velocity * -obj_b.physics.restitution

    def predict_position(self, object_id: str, time_delta: float) -> Optional[Vector3]:
        """Predict where an object will be after a time delta."""
        if not self.active_scene:
            return None

        obj = self.active_scene.get_object(object_id)
        if not obj:
            return None

        # Simple kinematic prediction
        current_pos = obj.transform.position
        velocity = obj.physics.velocity
        acceleration = self.active_scene.gravity * obj.physics.gravity_scale

        # s = s0 + v0*t + 0.5*a*t^2
        predicted = current_pos + velocity * time_delta + acceleration * (0.5 * time_delta ** 2)

        return predicted

    def analyze_trajectory(self, object_id_or_name: str) -> dict:
        """Analyze an object's trajectory over time."""
        if not self.active_scene:
            return {"error": "No active scene"}

        obj = self.active_scene.get_object(object_id_or_name)
        if not obj or not obj.timeline:
            return {"error": "Object not found or no timeline data"}

        positions = [(t, tf.position) for t, tf in obj.timeline]
        velocities = []

        for i in range(1, len(positions)):
            dt = positions[i][0] - positions[i-1][0]
            if dt > 0:
                vel = (positions[i][1] - positions[i-1][1]) / dt
                velocities.append(vel)

        avg_velocity = Vector3()
        if velocities:
            avg_velocity = sum(velocities, Vector3()) / len(velocities)

        # Calculate path length
        path_length = 0
        for i in range(1, len(positions)):
            path_length += (positions[i][1] - positions[i-1][1]).magnitude()

        return {
            "object_id": obj.id,
            "object_name": obj.name,
            "total_time": positions[-1][0] - positions[0][0] if len(positions) > 1 else 0,
            "path_length": path_length,
            "average_velocity": avg_velocity.to_dict(),
            "data_points": len(positions),
            "start_position": positions[0][1].to_dict() if positions else None,
            "end_position": positions[-1][1].to_dict() if positions else None
        }

    def describe_scene(self) -> str:
        """Generate a natural language description of the current scene."""
        if not self.active_scene:
            return "No active scene."

        scene = self.active_scene
        lines = [
            f"Scene: {scene.name}",
            f"Time: {scene.current_time:.2f}s",
            f"Objects: {len(scene.objects)}",
            ""
        ]

        for obj_id, obj in scene.objects.items():
            pos = obj.transform.position
            vel = obj.physics.velocity
            lines.append(
                f"  {obj.name} ({obj.object_type}): "
                f"pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}), "
                f"vel=({vel.x:.1f}, {vel.y:.1f}, {vel.z:.1f})"
            )

        return "\n".join(lines)

    def save_scene(self, name: Optional[str] = None) -> bool:
        """Save the current scene to disk."""
        scene_name = name or (self.active_scene.name if self.active_scene else None)
        if not scene_name or not self.active_scene:
            return False

        scene = self.active_scene
        data = {
            "name": scene.name,
            "time_scale": scene.time_scale,
            "current_time": scene.current_time,
            "gravity": scene.gravity.to_dict(),
            "objects": {}
        }

        for obj_id, obj in scene.objects.items():
            data["objects"][obj_id] = {
                "name": obj.name,
                "object_type": obj.object_type,
                "transform": obj.transform.to_dict(),
                "physics": {
                    "mass": obj.physics.mass,
                    "velocity": obj.physics.velocity.to_dict(),
                    "is_static": obj.physics.is_static
                },
                "properties": obj.properties
            }

        atomic_write_json(self.storage_path / f"{scene_name}.json", data)

        return True

    def load_scene(self, name: str) -> Optional[ImagineScene]:
        """Load a scene from disk."""
        file_path = self.storage_path / f"{name}.json"
        if not file_path.exists():
            return None

        data = read_json_with_integrity(file_path)

        scene = ImagineScene(
            name=data["name"],
            time_scale=data.get("time_scale", 1.0),
            current_time=data.get("current_time", 0.0),
            gravity=Vector3.from_dict(data.get("gravity", {}))
        )

        for obj_id, obj_data in data.get("objects", {}).items():
            obj = ImagineObject(
                id=obj_id,
                name=obj_data["name"],
                object_type=obj_data["object_type"],
                transform=Transform(
                    position=Vector3.from_dict(obj_data["transform"]["position"])
                ),
                physics=PhysicsProperties(
                    mass=obj_data["physics"].get("mass", 1.0),
                    velocity=Vector3.from_dict(obj_data["physics"].get("velocity", {})),
                    is_static=obj_data["physics"].get("is_static", False)
                ),
                properties=obj_data.get("properties", {})
            )
            scene.add_object(obj)

        self.scenes[name] = scene
        self.active_scene = scene
        return scene


class ImaginationTool:
    """Tool for Jarvis to interact with the 4D imagination space."""

    name = "imagine"
    description = "Create and simulate 4D scenes with objects, physics, and time"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create_scene", "switch_scene", "describe",
                    "create_object", "move_object", "simulate",
                    "predict", "analyze", "save", "load", "raycast"
                ],
                "description": "Action to perform in imagination space"
            },
            "name": {
                "type": "string",
                "description": "Name for scene or object"
            },
            "template": {
                "type": "string",
                "description": "Object template (cube, sphere, humanoid, vehicle, building)"
            },
            "position": {
                "type": "object",
                "description": "3D position {x, y, z}"
            },
            "velocity": {
                "type": "object",
                "description": "Velocity vector {x, y, z}"
            },
            "duration": {
                "type": "number",
                "description": "Simulation duration in seconds"
            },
            "direction": {
                "type": "object",
                "description": "Direction for raycast {x, y, z}"
            }
        },
        "required": ["action"]
    }
    requires_permission = False

    def __init__(self, engine: Optional[ImaginationEngine] = None):
        self.engine = engine or ImaginationEngine()

    def get_schema(self) -> dict:
        """Get the JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    async def execute(self, **params) -> str:
        """Execute an imagination action."""
        # Extract parameters from kwargs
        action = params.get("action")
        name = params.get("name")
        template = params.get("template")
        position = params.get("position")
        velocity = params.get("velocity")
        duration = params.get("duration")
        direction = params.get("direction")
        if not action:
            return "Error: 'action' parameter is required"

        if action == "create_scene":
            scene = self.engine.create_scene(name or "untitled")
            return f"Created scene '{scene.name}'"

        elif action == "switch_scene":
            scene = self.engine.switch_scene(name)
            if scene:
                return f"Switched to scene '{scene.name}'"
            return f"Scene '{name}' not found"

        elif action == "describe":
            return self.engine.describe_scene()

        elif action == "create_object":
            pos = Vector3.from_dict(position) if position else None
            obj = self.engine.create_object(
                name=name or "object",
                template=template,
                position=pos
            )
            if velocity:
                obj.physics.velocity = Vector3.from_dict(velocity)
            return f"Created object '{obj.name}' (id: {obj.id}) at position {obj.transform.position.to_dict()}"

        elif action == "move_object":
            if not self.engine.active_scene or not name:
                return "No active scene or object name"
            obj = self.engine.active_scene.get_object(name) or self.engine.active_scene.objects.get(name)
            if not obj:
                return f"Object '{name}' not found"
            if position:
                obj.transform.position = Vector3.from_dict(position)
            return f"Moved '{obj.name}' to {obj.transform.position.to_dict()}"

        elif action == "simulate":
            results = await self.engine.simulate(duration or 1.0)
            return f"Simulated {results['duration']}s with {len(results['collisions'])} collisions"

        elif action == "predict":
            if not name:
                return "Object name required"
            pos = self.engine.predict_position(name, duration or 1.0)
            if pos:
                return f"Predicted position for '{name}': {pos.to_dict()}"
            return f"Could not predict position for '{name}'"

        elif action == "analyze":
            if not name:
                return "Object name required"
            analysis = self.engine.analyze_trajectory(name)
            return json.dumps(analysis, indent=2)

        elif action == "save":
            success = self.engine.save_scene(name)
            return "Scene saved" if success else "Failed to save scene"

        elif action == "load":
            scene = self.engine.load_scene(name)
            if scene:
                return f"Loaded scene '{scene.name}' with {len(scene.objects)} objects"
            return f"Scene '{name}' not found"

        elif action == "raycast":
            if not self.engine.active_scene or not position or not direction:
                return "Active scene, position, and direction required"
            origin = Vector3.from_dict(position)
            dir_vec = Vector3.from_dict(direction)
            hit = self.engine.active_scene.raycast(origin, dir_vec)
            if hit:
                obj, point = hit
                return f"Hit '{obj.name}' at position {point.to_dict()}"
            return "No hit"

        return f"Unknown action: {action}"
