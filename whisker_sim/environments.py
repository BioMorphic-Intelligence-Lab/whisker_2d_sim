"""Environment presets ordered from easy to hard.

The progression is designed for localization experiments:
1. simple convex geometry;
2. oblique wall orientations;
3. concave geometry;
4. repeated local geometry / perceptual aliasing.
"""

from .map import RoomMap


ENVIRONMENT_NAMES = (
    "easy_rectangle",
    "angled_room",
    "concave_L",
    "repeated_bays",
)


def easy_rectangle():
    """Level 1: simple rectangular room."""
    return RoomMap(
        vertices=[
            [-1.50, -1.00],
            [ 1.50, -1.00],
            [ 1.50,  1.00],
            [-1.50,  1.00],
        ],
        textures=[
            "foam",
            "glass",
            "wood",
            "plaster",
        ],
        name="easy_rectangle",
        difficulty="1 / 4 - easy",
        description=(
            "Simple convex box for debugging contact detection, FSM behavior, "
            "and basic wall-relative localization."
        ),
    )


def angled_room():
    """Level 2: convex room with oblique walls."""
    return RoomMap(
        vertices=[
            [-1.60, -1.00],
            [ 1.35, -1.00],
            [ 1.70, -0.20],
            [ 1.30,  1.05],
            [-0.95,  1.20],
            [-1.60,  0.45],
        ],
        textures=[
            "foam",
            "glass",
            "glass",
            "wood",
            "plaster",
            "foam",
        ],
        name="angled_room",
        difficulty="2 / 4 - moderate",
        description=(
            "Convex but non-axis-aligned room. Slanted walls make yaw and "
            "surface-normal information more informative."
        ),
    )


def concave_L():
    """Level 3: non-convex L-shaped room."""
    return RoomMap(
        vertices=[
            [-1.50, -1.20],
            [ 1.50, -1.20],
            [ 1.50, -0.25],
            [ 0.65, -0.25],
            [ 0.65,  1.20],
            [-1.50,  1.20],
        ],
        textures=[
            "foam",
            "glass",
            "wood",
            "plaster",
            "glass",
            "foam",
        ],
        name="concave_L",
        difficulty="3 / 4 - hard",
        description=(
            "Non-convex L-shaped room with a strong concave corner and "
            "distinct local geometric contexts."
        ),
    )


def repeated_bays():
    """Level 4: repeated local geometries designed to create aliasing."""
    vertices = [
        [-2.00, -1.50],
        [ 2.00, -1.50],
        [ 2.00,  1.50],
        [ 1.40,  1.50],
        [ 1.40,  0.80],
        [ 1.00,  0.80],
        [ 1.00,  1.50],
        [ 0.40,  1.50],
        [ 0.40,  0.80],
        [ 0.00,  0.80],
        [ 0.00,  1.50],
        [-0.60,  1.50],
        [-0.60,  0.80],
        [-1.00,  0.80],
        [-1.00,  1.50],
        [-2.00,  1.50],
    ]

    # Three repeated U-shaped bays have similar geometry but different
    # semantic labels. This is useful later for geometry-vs-semantics studies.
    textures = [
        "concrete",
        "plaster",
        "glass",
        "foam",
        "foam",
        "foam",
        "plaster",
        "wood",
        "wood",
        "wood",
        "plaster",
        "glass",
        "glass",
        "glass",
        "plaster",
        "concrete",
    ]

    return RoomMap(
        vertices=vertices,
        textures=textures,
        name="repeated_bays",
        difficulty="4 / 4 - very hard",
        description=(
            "Non-convex room with three repeated U-shaped bays. Designed to "
            "create geometric perceptual aliasing; the bays deliberately use "
            "different texture labels for future semantic localization."
        ),
    )


_BUILDERS = {
    "easy_rectangle": easy_rectangle,
    "angled_room": angled_room,
    "concave_L": concave_L,
    "repeated_bays": repeated_bays,
}


def get_environment(name):
    """Construct an environment preset by name."""
    if name not in _BUILDERS:
        valid = ", ".join(ENVIRONMENT_NAMES)
        raise ValueError(
            "Unknown environment '{}'. Valid choices: {}".format(
                name,
                valid,
            )
        )

    return _BUILDERS[name]()


def environment_descriptions():
    """Return metadata for all available presets."""
    result = []

    for name in ENVIRONMENT_NAMES:
        room = get_environment(name)
        result.append(
            {
                "name": room.name,
                "difficulty": room.difficulty,
                "description": room.description,
            }
        )

    return result
