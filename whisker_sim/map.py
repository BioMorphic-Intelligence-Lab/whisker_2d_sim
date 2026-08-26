import numpy as np

from .geometry import (
    point_in_polygon,
    point_segment_distance,
    ray_segment_intersection,
)
from .models import RaycastHit


class RoomMap:
    """Polygonal confined room with per-wall semantic texture labels."""

    def __init__(
        self,
        vertices,
        textures,
        name="custom_room",
        difficulty="custom",
        description="",
    ):
        self.vertices = np.asarray(vertices, dtype=float)

        if self.vertices.ndim != 2 or self.vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2).")

        if len(textures) != len(self.vertices):
            raise ValueError(
                "Provide exactly one texture label for each polygon edge."
            )

        self.textures = list(textures)
        self.name = str(name)
        self.difficulty = str(difficulty)
        self.description = str(description)

    @classmethod
    def default_rectangle(cls):
        """Backward-compatible default room."""
        return cls(
            vertices=[
                [-1.5, -1.0],
                [1.5, -1.0],
                [1.5, 1.0],
                [-1.5, 1.0],
            ],
            textures=[
                "foam",
                "glass",
                "wood",
                "plaster",
            ],
            name="easy_rectangle",
            difficulty="1 / 4 - easy",
            description="Simple rectangular room.",
        )

    def segment(self, wall_id):
        """Return the two endpoints of one polygon wall segment."""
        start = self.vertices[wall_id]
        end = self.vertices[(wall_id + 1) % len(self.vertices)]
        return start, end

    def sdf(self, point):
        """Analytical signed distance: inside negative, outside positive."""
        point = np.asarray(point, dtype=float)

        distance = min(
            point_segment_distance(
                point,
                *self.segment(i)
            )
            for i in range(len(self.vertices))
        )

        if point_in_polygon(point, self.vertices):
            return -distance

        return distance

    def plot_bounds(self, margin=0.25):
        """Return room bounds with a margin for visualization."""
        x_min = float(np.min(self.vertices[:, 0])) - margin
        x_max = float(np.max(self.vertices[:, 0])) + margin
        y_min = float(np.min(self.vertices[:, 1])) - margin
        y_max = float(np.max(self.vertices[:, 1])) + margin
        return x_min, x_max, y_min, y_max

    def raycast(self, origin, direction, max_range):
        """Return the nearest wall intersected by a finite whisker ray.

        This is an ideal geometric contact model. It asks which wall is
        encountered first when starting at the whisker base and moving along
        the whisker direction, up to the whisker length.
        """
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float)

        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise ValueError("Ray direction must be non-zero.")

        # Normalize so the intersection parameter is a metric distance [m].
        direction = direction / norm

        best_distance = None
        best_point = None
        best_wall_id = None

        for wall_id in range(len(self.vertices)):
            start, end = self.segment(wall_id)

            distance = ray_segment_intersection(
                origin,
                direction,
                start,
                end,
            )

            # No hit, or the wall lies beyond the whisker tip.
            if distance is None or distance > max_range:
                continue

            # If multiple wall segments intersect this ray, the physically
            # relevant contact is the nearest one.
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_point = origin + distance * direction
                best_wall_id = wall_id

        if best_distance is None:
            return RaycastHit(
                False,
                None,
                None,
                None,
                None,
            )

        return RaycastHit(
            True,
            best_distance,
            best_point,
            best_wall_id,
            self.textures[best_wall_id],
        )
