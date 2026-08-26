import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix(theta: float) -> np.ndarray:
    """Return a 2D rotation matrix."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar 2D cross product."""
    return float(a[0] * b[1] - a[1] * b[0])


def point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    """Euclidean distance from point to finite line segment."""
    segment = end - start
    denominator = float(np.dot(segment, segment))

    if denominator < 1e-12:
        return float(np.linalg.norm(point - start))

    t = float(np.dot(point - start, segment) / denominator)
    t = np.clip(t, 0.0, 1.0)
    closest = start + t * segment
    return float(np.linalg.norm(point - closest))


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray-crossing point-in-polygon test for a simple polygon."""
    x, y = point
    inside = False

    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]

        if (y1 > y) == (y2 > y):
            continue

        x_intersection = (
            (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
        )

        if x < x_intersection:
            inside = not inside

    return inside


def ray_segment_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    segment_start: np.ndarray,
    segment_end: np.ndarray,
):
    """Return ray distance t for a ray/segment intersection, otherwise None."""
    p = ray_origin
    r = ray_direction
    q = segment_start
    s = segment_end - segment_start

    r_cross_s = cross2(r, s)
    if abs(r_cross_s) < 1e-12:
        return None

    q_minus_p = q - p
    t = cross2(q_minus_p, s) / r_cross_s
    u = cross2(q_minus_p, r) / r_cross_s

    if t >= 0.0 and 0.0 <= u <= 1.0:
        return float(t)

    return None
