"""Spatial design and physics simulation tools for Jarvis.

This module provides tools for:
- 3D spatial reasoning and design
- Physics simulations
- Engineering calculations
- CAD-like operations
"""

import math
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum

from core.helpers import safe_error_message
from .base import Tool


class Unit(Enum):
    """Units of measurement."""
    METERS = "m"
    CENTIMETERS = "cm"
    MILLIMETERS = "mm"
    FEET = "ft"
    INCHES = "in"
    KILOGRAMS = "kg"
    GRAMS = "g"
    POUNDS = "lb"
    NEWTONS = "N"
    JOULES = "J"
    WATTS = "W"


@dataclass
class Vector3:
    """3D vector for spatial calculations."""
    x: float
    y: float
    z: float

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return self * (1 / mag)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class Material:
    """Material properties for physics calculations."""
    name: str
    density: float  # kg/m³
    youngs_modulus: float  # Pa (elastic modulus)
    yield_strength: float  # Pa
    poisson_ratio: float = 0.3
    thermal_conductivity: float = 0.0  # W/(m·K)
    specific_heat: float = 0.0  # J/(kg·K)


# Common materials
MATERIALS = {
    "steel": Material("steel", 7850, 200e9, 250e6, 0.3, 50, 500),
    "aluminum": Material("aluminum", 2700, 70e9, 270e6, 0.33, 205, 900),
    "copper": Material("copper", 8960, 120e9, 70e6, 0.34, 385, 385),
    "wood": Material("wood", 600, 12e9, 40e6, 0.35, 0.15, 1700),
    "concrete": Material("concrete", 2400, 30e9, 30e6, 0.2, 1.5, 880),
    "plastic": Material("plastic", 950, 2e9, 30e6, 0.4, 0.2, 1800),
    "glass": Material("glass", 2500, 70e9, 50e6, 0.22, 1.0, 840),
}


class SpatialCalculationTool(Tool):
    """Tool for spatial calculations and geometry."""

    name = "spatial_calc"
    description = "Perform spatial and geometric calculations (area, volume, distances, angles)"
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["area", "volume", "distance", "angle", "centroid", "transform"],
                "description": "Operation to perform",
            },
            "shape": {
                "type": "string",
                "enum": ["rectangle", "circle", "sphere", "cylinder", "cube", "triangle", "polygon"],
                "description": "Shape type for calculation",
            },
            "dimensions": {
                "type": "object",
                "description": "Dimensions of the shape (e.g., width, height, depth, radius)",
            },
            "points": {
                "type": "array",
                "description": "List of 3D points for calculations",
            },
        },
        "required": ["operation"],
    }
    requires_permission = False

    async def execute(
        self,
        operation: str = "",
        shape: Optional[str] = None,
        dimensions: Optional[dict] = None,
        points: Optional[List[List[float]]] = None,
        **kwargs
    ) -> str:
        """Execute a spatial calculation."""
        try:
            if operation == "area":
                return self._calc_area(shape, dimensions or {})
            elif operation == "volume":
                return self._calc_volume(shape, dimensions or {})
            elif operation == "distance":
                return self._calc_distance(points or [])
            elif operation == "angle":
                return self._calc_angle(points or [])
            elif operation == "centroid":
                return self._calc_centroid(points or [])
            elif operation == "transform":
                return self._transform(points or [], dimensions or {})
            else:
                return f"Unknown operation: {operation}. Valid operations: area, volume, distance, angle, centroid, transform"
        except Exception as e:
            return f"Calculation error: {safe_error_message(e, 'Calculation error')}"

    def _calc_area(self, shape: Optional[str], dims: dict) -> str:
        """Calculate area of a shape."""
        shape = shape or "rectangle"
        dims = {k: float(v) for k, v in dims.items()}

        if shape == "rectangle":
            area = dims.get("width", 1) * dims.get("height", 1)
            return f"Rectangle area: {area:.4f} square units"
        elif shape == "circle":
            area = math.pi * dims.get("radius", 1) ** 2
            return f"Circle area: {area:.4f} square units"
        elif shape == "triangle":
            # Using base and height
            area = 0.5 * dims.get("base", 1) * dims.get("height", 1)
            return f"Triangle area: {area:.4f} square units"
        elif shape == "cylinder":
            # Lateral surface area
            radius = dims.get("radius", 1)
            height = dims.get("height", 1)
            area = 2 * math.pi * radius * height
            return f"Cylinder lateral area: {area:.4f} square units"
        elif shape == "sphere":
            radius = dims.get("radius", 1)
            area = 4 * math.pi * radius ** 2
            return f"Sphere surface area: {area:.4f} square units"
        else:
            return f"Unknown shape: {shape}"

    def _calc_volume(self, shape: Optional[str], dims: dict) -> str:
        """Calculate volume of a shape."""
        shape = shape or "cube"
        dims = {k: float(v) for k, v in dims.items()}

        if shape == "cube":
            side = dims.get("side", 1)
            volume = side ** 3
            return f"Cube volume: {volume:.4f} cubic units"
        elif shape == "rectangle" or shape == "box":
            volume = dims.get("width", 1) * dims.get("height", 1) * dims.get("depth", 1)
            return f"Box volume: {volume:.4f} cubic units"
        elif shape == "sphere":
            radius = dims.get("radius", 1)
            volume = (4/3) * math.pi * radius ** 3
            return f"Sphere volume: {volume:.4f} cubic units"
        elif shape == "cylinder":
            radius = dims.get("radius", 1)
            height = dims.get("height", 1)
            volume = math.pi * radius ** 2 * height
            return f"Cylinder volume: {volume:.4f} cubic units"
        elif shape == "cone":
            radius = dims.get("radius", 1)
            height = dims.get("height", 1)
            volume = (1/3) * math.pi * radius ** 2 * height
            return f"Cone volume: {volume:.4f} cubic units"
        else:
            return f"Unknown shape: {shape}"

    def _calc_distance(self, points: List[List[float]]) -> str:
        """Calculate distance between points."""
        if len(points) < 2:
            return "Need at least 2 points for distance calculation"

        results = []
        for i in range(len(points) - 1):
            p1 = Vector3(*points[i])
            p2 = Vector3(*points[i + 1])
            dist = (p2 - p1).magnitude()
            results.append(f"Distance P{i+1} to P{i+2}: {dist:.4f} units")

        # Total distance
        if len(points) > 2:
            total = sum(
                (Vector3(*points[i+1]) - Vector3(*points[i])).magnitude()
                for i in range(len(points) - 1)
            )
            results.append(f"Total path length: {total:.4f} units")

        return "\n".join(results)

    def _calc_angle(self, points: List[List[float]]) -> str:
        """Calculate angle between vectors."""
        if len(points) < 3:
            return "Need at least 3 points for angle calculation"

        results = []
        for i in range(len(points) - 2):
            p1 = Vector3(*points[i])
            p2 = Vector3(*points[i + 1])
            p3 = Vector3(*points[i + 2])

            v1 = (p1 - p2).normalize()
            v2 = (p3 - p2).normalize()

            dot = max(-1, min(1, v1.dot(v2)))
            angle_rad = math.acos(dot)
            angle_deg = math.degrees(angle_rad)

            results.append(f"Angle at P{i+2}: {angle_deg:.2f}° ({angle_rad:.4f} rad)")

        return "\n".join(results)

    def _calc_centroid(self, points: List[List[float]]) -> str:
        """Calculate centroid of points."""
        if not points:
            return "No points provided"

        total = Vector3(0, 0, 0)
        for p in points:
            total = total + Vector3(*p)

        centroid = total * (1 / len(points))
        return f"Centroid: ({centroid.x:.4f}, {centroid.y:.4f}, {centroid.z:.4f})"

    def _transform(self, points: List[List[float]], params: dict) -> str:
        """Apply transformation to points."""
        if not points:
            return "No points provided"

        # Get transformation parameters
        translate = params.get("translate", [0, 0, 0])
        rotate = params.get("rotate", [0, 0, 0])  # degrees around x, y, z
        scale = params.get("scale", [1, 1, 1])

        results = ["Transformed points:"]
        for i, p in enumerate(points):
            # Apply scale
            x, y, z = p[0] * scale[0], p[1] * scale[1], p[2] * scale[2]

            # Apply rotation (simplified - just Z rotation for now)
            if rotate[2] != 0:
                angle = math.radians(rotate[2])
                x_new = x * math.cos(angle) - y * math.sin(angle)
                y_new = x * math.sin(angle) + y * math.cos(angle)
                x, y = x_new, y_new

            # Apply translation
            x += translate[0]
            y += translate[1]
            z += translate[2]

            results.append(f"P{i+1}: ({x:.4f}, {y:.4f}, {z:.4f})")

        return "\n".join(results)


class PhysicsSimulationTool(Tool):
    """Tool for physics simulations and calculations."""

    name = "physics_sim"
    description = "Run physics simulations and calculations (mechanics, forces, energy, motion)"
    parameters = {
        "type": "object",
        "properties": {
            "simulation": {
                "type": "string",
                "enum": ["projectile", "pendulum", "collision", "stress", "beam", "thermal", "fluid"],
                "description": "Type of simulation to run",
            },
            "params": {
                "type": "object",
                "description": "Parameters for the simulation",
            },
            "material": {
                "type": "string",
                "enum": ["steel", "aluminum", "wood", "concrete", "plastic", "glass"],
                "description": "Material name for structural calculations",
            },
        },
        "required": ["simulation"],
    }
    requires_permission = False

    async def execute(
        self,
        simulation: str = "",
        params: Optional[dict] = None,
        material: Optional[str] = None,
        **kwargs
    ) -> str:
        """Execute a physics simulation."""
        params = params or {}

        try:
            if simulation == "projectile":
                return self._projectile_motion(params)
            elif simulation == "pendulum":
                return self._pendulum(params)
            elif simulation == "collision":
                return self._collision(params)
            elif simulation == "stress":
                return self._stress_analysis(params, material)
            elif simulation == "beam":
                return self._beam_analysis(params, material)
            elif simulation == "thermal":
                return self._thermal(params, material)
            elif simulation == "fluid":
                return self._fluid_dynamics(params)
            else:
                return f"Unknown simulation: {simulation}. Valid simulations: projectile, pendulum, collision, stress, beam, thermal, fluid"
        except Exception as e:
            return f"Simulation error: {safe_error_message(e, 'Calculation error')}"

    def _projectile_motion(self, params: dict) -> str:
        """Calculate projectile motion."""
        g = 9.81  # gravity
        v0 = float(params.get("velocity", 10))  # initial velocity m/s
        angle = math.radians(float(params.get("angle", 45)))  # launch angle
        h0 = float(params.get("height", 0))  # initial height

        vx = v0 * math.cos(angle)
        vy = v0 * math.sin(angle)

        # Time of flight
        discriminant = vy ** 2 + 2 * g * h0
        if discriminant < 0:
            return "Projectile cannot reach ground from initial height"
        t_flight = (vy + math.sqrt(discriminant)) / g

        # Maximum height
        t_max_height = vy / g
        max_height = h0 + vy * t_max_height - 0.5 * g * t_max_height ** 2

        # Range
        range_x = vx * t_flight

        results = [
            "Projectile Motion Simulation",
            "=" * 40,
            f"Initial velocity: {v0:.2f} m/s",
            f"Launch angle: {math.degrees(angle):.2f}°",
            f"Initial height: {h0:.2f} m",
            "",
            f"Horizontal velocity: {vx:.2f} m/s",
            f"Vertical velocity: {vy:.2f} m/s",
            "",
            f"Time of flight: {t_flight:.2f} s",
            f"Maximum height: {max_height:.2f} m",
            f"Horizontal range: {range_x:.2f} m",
        ]

        # Trajectory points
        results.append("\nTrajectory (time, x, y):")
        for t in [0, t_flight * 0.25, t_flight * 0.5, t_flight * 0.75, t_flight]:
            x = vx * t
            y = h0 + vy * t - 0.5 * g * t ** 2
            results.append(f"  t={t:.2f}s: ({x:.2f}, {max(0, y):.2f}) m")

        return "\n".join(results)

    def _pendulum(self, params: dict) -> str:
        """Calculate pendulum motion."""
        g = 9.81
        length = float(params.get("length", 1))  # meters
        initial_angle = math.radians(float(params.get("angle", 30)))  # degrees

        # Period (small angle approximation)
        period = 2 * math.pi * math.sqrt(length / g)
        frequency = 1 / period

        # Angular velocity max
        omega_max = initial_angle * math.sqrt(g / length)

        results = [
            "Pendulum Simulation",
            "=" * 40,
            f"Length: {length:.2f} m",
            f"Initial angle: {math.degrees(initial_angle):.2f}°",
            "",
            f"Period: {period:.2f} s",
            f"Frequency: {frequency:.4f} Hz",
            f"Maximum angular velocity: {omega_max:.4f} rad/s",
            "",
            "Position at time t:",
        ]

        # Show positions at different times
        for t in [0, period/4, period/2, 3*period/4, period]:
            theta = initial_angle * math.cos(omega_max * t)
            results.append(f"  t={t:.2f}s: θ = {math.degrees(theta):.2f}°")

        return "\n".join(results)

    def _collision(self, params: dict) -> str:
        """Calculate collision dynamics."""
        m1 = float(params.get("mass1", 1))  # kg
        m2 = float(params.get("mass2", 1))  # kg
        v1 = float(params.get("velocity1", 5))  # m/s
        v2 = float(params.get("velocity2", 0))  # m/s
        elastic = params.get("elastic", True)

        # Initial momentum and energy
        p_initial = m1 * v1 + m2 * v2
        ke_initial = 0.5 * m1 * v1 ** 2 + 0.5 * m2 * v2 ** 2

        if elastic:
            # Elastic collision
            v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
            v2_final = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
        else:
            # Inelastic collision (objects stick together)
            v1_final = v2_final = p_initial / (m1 + m2)

        # Final momentum and energy
        p_final = m1 * v1_final + m2 * v2_final
        ke_final = 0.5 * m1 * v1_final ** 2 + 0.5 * m2 * v2_final ** 2

        collision_type = "Elastic" if elastic else "Inelastic"

        results = [
            f"{collision_type} Collision Simulation",
            "=" * 40,
            f"Object 1: mass={m1:.2f} kg, v={v1:.2f} m/s",
            f"Object 2: mass={m2:.2f} kg, v={v2:.2f} m/s",
            "",
            f"Initial momentum: {p_initial:.4f} kg·m/s",
            f"Final momentum: {p_final:.4f} kg·m/s",
            f"Initial KE: {ke_initial:.4f} J",
            f"Final KE: {ke_final:.4f} J",
            "",
            "After collision:",
            f"  Object 1 velocity: {v1_final:.2f} m/s",
            f"  Object 2 velocity: {v2_final:.2f} m/s",
        ]

        if not elastic:
            energy_lost = ke_initial - ke_final
            results.append(f"  Energy lost: {energy_lost:.4f} J")

        return "\n".join(results)

    def _stress_analysis(self, params: dict, material_name: Optional[str]) -> str:
        """Perform stress analysis."""
        force = float(params.get("force", 1000))  # N
        area = float(params.get("area", 0.01))  # m²
        length = float(params.get("length", 1))  # m

        material = MATERIALS.get(material_name, MATERIALS["steel"]) if material_name else MATERIALS["steel"]

        # Stress and strain
        stress = force / area
        strain = stress / material.youngs_modulus
        deformation = strain * length

        # Safety factor
        safety_factor = material.yield_strength / stress if stress > 0 else float('inf')

        results = [
            "Stress Analysis",
            "=" * 40,
            f"Material: {material.name}",
            f"Applied force: {force:.2f} N",
            f"Cross-section area: {area*1e6:.2f} mm²",
            f"Length: {length:.2f} m",
            "",
            "Results:",
            f"  Stress: {stress/1e6:.2f} MPa",
            f"  Strain: {strain:.6f}",
            f"  Deformation: {deformation*1000:.4f} mm",
            f"  Yield strength: {material.yield_strength/1e6:.2f} MPa",
            f"  Safety factor: {safety_factor:.2f}",
        ]

        if safety_factor < 1:
            results.append("\n  ⚠️ WARNING: Safety factor < 1, material will yield!")
        elif safety_factor < 2:
            results.append("\n  ⚠️ Caution: Safety factor < 2, design may be unsafe")

        return "\n".join(results)

    def _beam_analysis(self, params: dict, material_name: Optional[str]) -> str:
        """Perform beam deflection analysis."""
        length = float(params.get("length", 2))  # m
        load = float(params.get("load", 1000))  # N (point load at center)
        width = float(params.get("width", 0.1))  # m
        height = float(params.get("height", 0.1))  # m
        load_type = params.get("load_type", "point")  # point or distributed

        material = MATERIALS.get(material_name, MATERIALS["steel"]) if material_name else MATERIALS["steel"]

        # Second moment of area (rectangular)
        I = (width * height ** 3) / 12

        # Deflection
        if load_type == "point":
            # Point load at center
            delta_max = (load * length ** 3) / (48 * material.youngs_modulus * I)
            max_moment = load * length / 4
        else:
            # Distributed load
            delta_max = (5 * load * length ** 4) / (384 * material.youngs_modulus * I)
            max_moment = load * length / 8

        # Stress
        max_stress = max_moment * (height / 2) / I

        results = [
            "Beam Analysis",
            "=" * 40,
            f"Material: {material.name}",
            f"Length: {length:.2f} m",
            f"Width: {width*100:.2f} cm",
            f"Height: {height*100:.2f} cm",
            f"Load type: {load_type}",
            f"Load: {load:.2f} N",
            "",
            "Results:",
            f"  Second moment of area: {I*1e12:.2f} mm⁴",
            f"  Maximum deflection: {delta_max*1000:.4f} mm",
            f"  Maximum bending stress: {max_stress/1e6:.2f} MPa",
            f"  Span/deflection ratio: {length/delta_max:.1f}",
        ]

        return "\n".join(results)

    def _thermal(self, params: dict, material_name: Optional[str]) -> str:
        """Perform thermal analysis."""
        mass = float(params.get("mass", 1))  # kg
        temp_initial = float(params.get("temp_initial", 20))  # °C
        temp_final = float(params.get("temp_final", 100))  # °C
        power = float(params.get("power", 100))  # W

        material = MATERIALS.get(material_name, MATERIALS["steel"]) if material_name else MATERIALS["steel"]

        # Heat required
        delta_t = temp_final - temp_initial
        heat = mass * material.specific_heat * delta_t

        # Time to heat
        time_seconds = heat / power if power > 0 else 0

        results = [
            "Thermal Analysis",
            "=" * 40,
            f"Material: {material.name}",
            f"Mass: {mass:.2f} kg",
            f"Temperature change: {temp_initial}°C → {temp_final}°C",
            f"Heating power: {power:.2f} W",
            "",
            "Results:",
            f"  Heat required: {heat/1000:.2f} kJ",
            f"  Time to heat: {time_seconds/60:.2f} minutes",
            f"  Specific heat: {material.specific_heat:.0f} J/(kg·K)",
        ]

        return "\n".join(results)

    def _fluid_dynamics(self, params: dict) -> str:
        """Perform fluid dynamics calculations."""
        fluid = params.get("fluid", "water")
        velocity = float(params.get("velocity", 1))  # m/s
        diameter = float(params.get("diameter", 0.1))  # m
        length = float(params.get("length", 10))  # m
        viscosity = float(params.get("viscosity", 0.001))  # Pa·s (water ~0.001)
        density = float(params.get("density", 1000))  # kg/m³ (water ~1000)

        # Calculate Reynolds number
        area = math.pi * (diameter / 2) ** 2
        flow_rate = velocity * area

        # Reynolds number
        Re = (density * velocity * diameter) / viscosity

        # Flow regime
        if Re < 2300:
            flow_regime = "Laminar"
            friction_factor = 64 / Re
        elif Re < 4000:
            flow_regime = "Transitional"
            friction_factor = 0.03  # Approximate
        else:
            flow_regime = "Turbulent"
            # Blasius formula for turbulent flow
            friction_factor = 0.316 / (Re ** 0.25)

        # Pressure drop (Darcy-Weisbach)
        pressure_drop = friction_factor * (length / diameter) * (0.5 * density * velocity ** 2)

        results = [
            "Fluid Dynamics Analysis",
            "=" * 40,
            f"Fluid: {fluid}",
            f"Velocity: {velocity:.2f} m/s",
            f"Pipe diameter: {diameter*1000:.2f} mm",
            f"Pipe length: {length:.2f} m",
            "",
            "Results:",
            f"  Flow rate: {flow_rate*1000:.4f} L/s",
            f"  Reynolds number: {Re:.0f}",
            f"  Flow regime: {flow_regime}",
            f"  Friction factor: {friction_factor:.4f}",
            f"  Pressure drop: {pressure_drop/1000:.2f} kPa",
        ]

        return "\n".join(results)


class DesignTool(Tool):
    """Tool for design calculations and recommendations."""

    name = "design_calc"
    description = "Perform design calculations and generate recommendations for engineering problems"
    parameters = {
        "type": "object",
        "properties": {
            "design_type": {
                "type": "string",
                "enum": ["beam", "shaft", "column", "pressure_vessel", "gear", "spring"],
                "description": "Type of design calculation",
            },
            "requirements": {
                "type": "object",
                "description": "Design requirements (load, span, safety_factor, etc.)",
            },
            "material": {
                "type": "string",
                "enum": ["steel", "aluminum", "wood", "concrete", "plastic", "glass"],
                "description": "Preferred material",
            },
        },
        "required": ["design_type"],
    }
    requires_permission = False

    async def execute(
        self,
        design_type: str = "",
        requirements: Optional[dict] = None,
        material: Optional[str] = None,
        **kwargs
    ) -> str:
        """Execute a design calculation."""
        requirements = requirements or {}

        try:
            if design_type == "beam":
                return self._design_beam(requirements, material)
            elif design_type == "shaft":
                return self._design_shaft(requirements, material)
            elif design_type == "column":
                return self._design_column(requirements, material)
            elif design_type == "pressure_vessel":
                return self._design_pressure_vessel(requirements, material)
            elif design_type == "gear":
                return self._design_gear(requirements)
            elif design_type == "spring":
                return self._design_spring(requirements, material)
            else:
                return f"Unknown design type: {design_type}"
        except Exception as e:
            return f"Design calculation error: {safe_error_message(e, 'Calculation error')}"

    def _design_beam(self, req: dict, material_name: Optional[str]) -> str:
        """Design a beam for given requirements."""
        span = float(req.get("span", 2))  # m
        load = float(req.get("load", 5000))  # N
        safety_factor = float(req.get("safety_factor", 2))
        max_deflection = float(req.get("max_deflection", span / 250))  # m

        material = MATERIALS.get(material_name, MATERIALS["steel"]) if material_name else MATERIALS["steel"]

        # Calculate required section modulus
        max_moment = load * span / 4  # Point load at center
        required_stress = material.yield_strength / safety_factor
        required_section_modulus = max_moment / required_stress

        # For rectangular beam
        # S = b*h²/6, assume b = h/2
        h = (6 * required_section_modulus * 2) ** (1/3)
        b = h / 2

        # Check deflection
        I = b * h ** 3 / 12
        E = material.youngs_modulus
        actual_deflection = (load * span ** 3) / (48 * E * I)

        results = [
            "Beam Design",
            "=" * 40,
            f"Material: {material.name}",
            f"Span: {span:.2f} m",
            f"Load: {load:.2f} N",
            f"Safety factor: {safety_factor}",
            "",
            "Recommended section:",
            f"  Height: {h*1000:.1f} mm",
            f"  Width: {b*1000:.1f} mm",
            f"  Section modulus: {required_section_modulus*1e6:.0f} mm³",
            "",
            "Verification:",
            f"  Max stress: {max_moment*h/(2*I)/1e6:.2f} MPa",
            f"  Max deflection: {actual_deflection*1000:.2f} mm",
            f"  Allowable: {max_deflection*1000:.2f} mm",
        ]

        if actual_deflection <= max_deflection:
            results.append("  ✓ Deflection OK")
        else:
            results.append("  ✗ Deflection exceeded, increase section size")

        return "\n".join(results)

    def _design_shaft(self, req: dict, material_name: Optional[str]) -> str:
        """Design a shaft for given requirements."""
        torque = float(req.get("torque", 500))  # N·m
        power = float(req.get("power", 5000))  # W
        rpm = float(req.get("rpm", 1000))
        safety_factor = float(req.get("safety_factor", 2))

        material = MATERIALS.get(material_name, MATERIALS["steel"]) if material_name else MATERIALS["steel"]

        # Calculate required diameter
        # τ = T*r/J = 16T/(π*d³)
        # d = (16*T/(π*τ_allowable))^(1/3)
        tau_allowable = material.yield_strength / (2 * safety_factor)  # Shear yield ~0.5 * tensile
        d = ((16 * torque) / (math.pi * tau_allowable)) ** (1/3)

        # Check torsional rigidity
        J = math.pi * d ** 4 / 32
        G = material.youngs_modulus / (2 * (1 + material.poisson_ratio))  # Shear modulus
        theta = torque / (G * J)  # Twist per unit length

        results = [
            "Shaft Design",
            "=" * 40,
            f"Material: {material.name}",
            f"Torque: {torque:.2f} N·m",
            f"Power: {power:.2f} W at {rpm:.0f} RPM",
            f"Safety factor: {safety_factor}",
            "",
            f"Recommended diameter: {d*1000:.1f} mm",
            "",
            "Verification:",
            f"  Shear stress: {16*torque/(math.pi*d**3)/1e6:.2f} MPa",
            f"  Allowable shear: {tau_allowable/1e6:.2f} MPa",
            f"  Twist rate: {theta:.6f} rad/m",
        ]

        return "\n".join(results)

    def _design_column(self, req: dict, material_name: Optional[str]) -> str:
        """Design a column for buckling."""
        length = float(req.get("length", 3))  # m
        load = float(req.get("load", 50000))  # N
        end_condition = req.get("end_condition", "pinned")  # pinned, fixed, free
        safety_factor = float(req.get("safety_factor", 2))

        material = MATERIALS.get(material_name, MATERIALS["steel"]) if material_name else MATERIALS["steel"]

        # Effective length factor
        k_factors = {"pinned": 1.0, "fixed": 0.5, "fixed_pinned": 0.7, "fixed_free": 2.0}
        k = k_factors.get(end_condition, 1.0)

        # Euler buckling load: P_cr = π²EI/(kL)²
        # Required I = P_cr*(kL)²/(π²E)
        P_critical = load * safety_factor
        I_required = P_critical * (k * length) ** 2 / (math.pi ** 2 * material.youngs_modulus)

        # For circular column: I = πd⁴/64
        d = (64 * I_required / math.pi) ** (1/4)
        A = math.pi * d ** 2 / 4

        # Check slenderness ratio
        r = d / 4  # Radius of gyration for circle
        slenderness = k * length / r

        results = [
            "Column Design",
            "=" * 40,
            f"Material: {material.name}",
            f"Length: {length:.2f} m",
            f"Load: {load:.2f} N",
            f"End condition: {end_condition} (k={k})",
            f"Safety factor: {safety_factor}",
            "",
            f"Recommended diameter: {d*1000:.1f} mm",
            f"Cross-section area: {A*1e6:.0f} mm²",
            "",
            "Verification:",
            f"  Slenderness ratio: {slenderness:.1f}",
            f"  Critical load: {math.pi**2*material.youngs_modulus*I_required/((k*length)**2)/1000:.1f} kN",
        ]

        return "\n".join(results)

    def _design_pressure_vessel(self, req: dict, material_name: Optional[str]) -> str:
        """Design a pressure vessel."""
        pressure = float(req.get("pressure", 1e6))  # Pa
        diameter = float(req.get("diameter", 0.5))  # m
        length = float(req.get("length", 2))  # m
        safety_factor = float(req.get("safety_factor", 3))

        material = MATERIALS.get(material_name, MATERIALS["steel"]) if material_name else MATERIALS["steel"]

        # Thin-walled pressure vessel
        # Hoop stress: σ = P*r/t
        # Required thickness: t = P*r/(σ_allowable)
        r = diameter / 2
        sigma_allowable = material.yield_strength / safety_factor
        t = pressure * r / sigma_allowable

        # Add corrosion allowance
        t_with_allowance = t + 0.002  # 2mm corrosion allowance

        results = [
            "Pressure Vessel Design",
            "=" * 40,
            f"Material: {material.name}",
            f"Design pressure: {pressure/1e6:.2f} MPa",
            f"Diameter: {diameter*1000:.0f} mm",
            f"Length: {length:.2f} m",
            f"Safety factor: {safety_factor}",
            "",
            "Required wall thickness:",
            f"  Calculated: {t*1000:.2f} mm",
            f"  With corrosion allowance: {t_with_allowance*1000:.2f} mm",
            "",
            "Verification:",
            f"  Hoop stress: {pressure*r/t/1e6:.2f} MPa",
            f"  Allowable stress: {sigma_allowable/1e6:.2f} MPa",
        ]

        return "\n".join(results)

    def _design_gear(self, req: dict) -> str:
        """Design a gear."""
        power = float(req.get("power", 5000))  # W
        rpm = float(req.get("rpm", 1000))
        ratio = float(req.get("ratio", 3))
        module = float(req.get("module", 3))  # mm

        # Calculate basic gear parameters
        torque = power / (2 * math.pi * rpm / 60)
        pinion_teeth = 18  # Minimum for no undercutting
        gear_teeth = int(pinion_teeth * ratio)

        # Pitch diameters
        d_pinion = module * pinion_teeth
        d_gear = module * gear_teeth

        # Center distance
        center_distance = (d_pinion + d_gear) / 2

        results = [
            "Gear Design",
            "=" * 40,
            f"Power: {power/1000:.2f} kW",
            f"Speed: {rpm:.0f} RPM",
            f"Gear ratio: {ratio:.1f}:1",
            f"Module: {module:.1f} mm",
            "",
            "Pinion:",
            f"  Teeth: {pinion_teeth}",
            f"  Pitch diameter: {d_pinion:.1f} mm",
            f"  Torque: {torque:.2f} N·m",
            "",
            "Gear:",
            f"  Teeth: {gear_teeth}",
            f"  Pitch diameter: {d_gear:.1f} mm",
            "",
            f"Center distance: {center_distance:.1f} mm",
        ]

        return "\n".join(results)

    def _design_spring(self, req: dict, material_name: Optional[str]) -> str:
        """Design a spring."""
        force = float(req.get("force", 100))  # N
        deflection = float(req.get("deflection", 0.05))  # m
        spring_type = req.get("type", "compression")

        # Spring rate
        k = force / deflection

        # Assume music wire properties
        G = 79.3e9  # Shear modulus Pa
        tau_allowable = 800e6  # Allowable shear stress Pa

        # Trial design (assume wire diameter)
        d = 0.003  # 3mm wire diameter
        D = 0.025  # 25mm mean coil diameter
        C = D / d  # Spring index

        # Number of active coils
        n = (G * d ** 4) / (8 * k * D ** 3)

        # Wahl correction factor
        K = (4 * C - 1) / (4 * C - 4) + 0.615 / C

        # Actual shear stress
        tau = K * (8 * force * D) / (math.pi * d ** 3)

        results = [
            "Spring Design",
            "=" * 40,
            f"Type: {spring_type}",
            f"Force: {force:.2f} N",
            f"Deflection: {deflection*1000:.1f} mm",
            f"Spring rate: {k:.2f} N/m",
            "",
            "Recommended dimensions:",
            f"  Wire diameter: {d*1000:.1f} mm",
            f"  Mean coil diameter: {D*1000:.1f} mm",
            f"  Spring index: {C:.1f}",
            f"  Active coils: {n:.1f}",
            "",
            "Verification:",
            f"  Shear stress: {tau/1e6:.1f} MPa",
            f"  Allowable: {tau_allowable/1e6:.0f} MPa",
        ]

        return "\n".join(results)


class SpatialTools:
    """Collection of spatial and physics tools."""

    def __init__(self):
        self.tools = [
            SpatialCalculationTool(),
            PhysicsSimulationTool(),
            DesignTool(),
        ]

    def get_tools(self) -> list[Tool]:
        """Get all spatial tools."""
        return self.tools
