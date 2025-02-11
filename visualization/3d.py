from ursina import *
from textwrap import dedent
import random

app = Ursina()
window.fullscreen = False

# -------------------------------
# Scene setup (floor, sky, camera)
# -------------------------------
Entity(
    model='quad',
    scale=60,
    texture='white_cube',
    texture_scale=(60, 60),
    rotation_x=90,
    y=-5,
    color=color.light_gray
)
Entity(
    model='sphere',
    scale=100,
    texture='textures/sky0',
    double_sided=True
)
EditorCamera()
camera.world_position = (0, 0, -15)

# -------------------------------
# Global variables and state
# -------------------------------
cube_model = 'models/custom_cube'
CUBES = []
PARENT = Entity()  # used as a temporary parent for rotations
animation_time = 0.5
action_trigger = True
action_mode = True  # initially set to True (action mode), then toggled to start in view mode
message = Text(origin=(0, 19), color=color.black)


# -------------------------------
# INITIAL CONFIGURATION
# -------------------------------

def cube_string_to_config(cube_string):
    """
    Converts a cube definition string into the configuration dictionary format.

    The cube string is expected to have 54 characters, corresponding to:
      U1-U9, R1-R9, F1-F9, D1-D9, L1-L9, B1-B9.

    Returns:
        A dictionary with keys: 'LEFT', 'RIGHT', 'UP', 'DOWN', 'FACE', 'BACK',
        where each value is a list of 9 facelets (colors).
    """
    if len(cube_string) != 54:
        raise ValueError("Cube string must be exactly 54 characters long.")
    
    # Map each letter back to its corresponding color.
    letter_to_color = {
        'U': color.white,
        'R': color.red,
        'F': color.green,
        'D': color.yellow,
        'L': color.orange,
        'B': color.blue
    }
    
    # Build the configuration dictionary.
    config = {
        'UP':    [letter_to_color[c] for c in cube_string[0:9]],
        'RIGHT': [letter_to_color[c] for c in cube_string[9:18]],
        'FACE':  [letter_to_color[c] for c in cube_string[18:27]],
        'DOWN':  [letter_to_color[c] for c in cube_string[27:36]],
        'LEFT':  [letter_to_color[c] for c in cube_string[36:45]],
        'BACK':  [letter_to_color[c] for c in cube_string[45:54]]
    }
    return config

# initial_config = {
#     'LEFT':   [color.orange for _ in range(9)],
#     'RIGHT':  [color.red    for _ in range(9)],
#     'UP':    [color.white  for _ in range(9)],
#     'DOWN': [color.yellow for _ in range(9)],
#     'FACE':   [color.green  for _ in range(9)],
#     'BACK':   [color.blue   for _ in range(9)]
# }


initial_config_string = "DUUDUDUDUFRLLRLFLLLLFBFFBFFUUDUDUDDDLRBRLFRRBRFBBBBRBR"

initial_config = cube_string_to_config(initial_config_string)

def config_to_cube_string(config):
    # Map each color to the letter used in the cube string notation.
    color_to_letter = {
        color.white:  'U',
        color.red:    'R',
        color.green:  'F',
        color.yellow: 'D',
        color.orange: 'L',
        color.blue:   'B'
    }
    
    # The order of faces in the cube string is: Up, Right, Front, Down, Left, Back.
    face_order = ['UP', 'RIGHT', 'FACE', 'DOWN', 'LEFT', 'BACK']
    
    # Build the cube string by converting each face’s list of colors into letters.
    cube_string = ""
    for face in face_order:
        # Join the letters for each facelet in the current face.
        cube_string += "".join(color_to_letter[facelet] for facelet in config[face])
    return cube_string

# --- New: Moves from your Rubik's cube solver ---
# You can change this string to whatever your solver returns.
solver_moves = "R U D2 B2 R B D' L' U2 B2 U' F2 R2 D' F2 L2 F2 D'"
moves_list = solver_moves.split()

# (Optional) Show a little text with how many moves remain.
move_message = Text(text=f"Moves left: {len(moves_list)}", position=(-0.5,0.45), scale=2, color=color.black)

# -------------------------------
# Helper: Create positions for each cubie on the surface.
# -------------------------------
def create_cube_positions():
    LEFT   = {Vec3(-1, y, z) for y in range(-1, 2) for z in range(-1, 2)}
    DOWN = {Vec3(x, -1, z) for x in range(-1, 2) for z in range(-1, 2)}
    FACE   = {Vec3(x, y, -1) for x in range(-1, 2) for y in range(-1, 2)}
    BACK   = {Vec3(x, y, 1) for x in range(-1, 2) for y in range(-1, 2)}
    RIGHT  = {Vec3(1, y, z) for y in range(-1, 2) for z in range(-1, 2)}
    UP    = {Vec3(x, 1, z) for x in range(-1, 2) for z in range(-1, 2)}
    SIDE_POSITIONS = LEFT | DOWN | FACE | BACK | RIGHT | UP
    return LEFT, DOWN, FACE, BACK, RIGHT, UP, SIDE_POSITIONS

LEFT, DOWN, FACE, BACK, RIGHT, UP, SIDE_POSITIONS = create_cube_positions()

# For rotations we still need to know which cubies belong to each face.
cubes_side_positions = {
    'LEFT': LEFT,
    'DOWN': DOWN,
    'FACE': FACE,
    'BACK': BACK,
    'RIGHT': RIGHT,
    'UP': UP
}

# Mapping from side name to the rotation axis.
rotation_axes = {
    'LEFT': 'x',
    'RIGHT': 'x',
    'UP': 'y',
    'DOWN': 'y',
    'FACE': 'z',
    'BACK': 'z'
}

# -------------------------------
# Functions to place stickers on cubies.
# -------------------------------
def get_sticker_index(face, pos):
    if face == 'LEFT':  # face at x = -1; free: y and z.
        row = 1 - pos.y
        col = 1 - pos.z
        return (int(row), int(col))
    elif face == 'RIGHT':  # face at x = 1; free: y and z.
        row = 1 - pos.y
        col = pos.z + 1
        return (int(row), int(col))
    elif face == 'FACE':  # face at z = -1; free: x and y.
        row = 1 - pos.y
        col = pos.x + 1
        return (int(row), int(col))
    elif face == 'BACK':  # face at z = 1; free: x and y.
        row = 1 - pos.y
        col = 1 - pos.x
        return (int(row), int(col))
    elif face == 'UP':  # face at y = 1; free: x and z.
        row = 1 - pos.z
        col = pos.x + 1
        return (int(row), int(col))
    elif face == 'DOWN':  # face at y = -1; free: x and z.
        row = pos.z + 1
        col = pos.x + 1
        return (int(row), int(col))

face_to_data = {
    'LEFT':   {'normal': Vec3(-1, 0, 0), 'rotation': Vec3(0,90,0)},
    'RIGHT':  {'normal': Vec3(1, 0, 0),  'rotation': Vec3(0,-90,0)},
    'FACE':   {'normal': Vec3(0, 0, -1), 'rotation': Vec3(0,0,0)},
    'BACK':   {'normal': Vec3(0, 0, 1),  'rotation': Vec3(0,180,0)},
    'UP':    {'normal': Vec3(0, 1, 0),  'rotation': Vec3(-90,0,0)},
    'DOWN': {'normal': Vec3(0, -1, 0), 'rotation': Vec3(90,0,0)}
}

def create_sticker(cubie, face, sticker_color):
    data = face_to_data[face]
    normal = data['normal']
    rot = data['rotation']
    offset = normal * 0.501
    sticker = Entity(
        parent=cubie,
        model='quad',
        color=sticker_color,
        scale=0.9,
        position=offset,
        rotation=rot,
        double_sided=True,
    )
    return sticker

def apply_initial_config(config):
    for cube in CUBES:
        pos = cube.position
        if pos.x == -1:
            row, col = get_sticker_index('LEFT', pos)
            sticker_color = config['LEFT'][row*3 + col]
            create_sticker(cube, 'LEFT', sticker_color)
        if pos.x == 1:
            row, col = get_sticker_index('RIGHT', pos)
            sticker_color = config['RIGHT'][row*3 + col]
            create_sticker(cube, 'RIGHT', sticker_color)
        if pos.z == -1:
            row, col = get_sticker_index('FACE', pos)
            sticker_color = config['FACE'][row*3 + col]
            create_sticker(cube, 'FACE', sticker_color)
        if pos.z == 1:
            row, col = get_sticker_index('BACK', pos)
            sticker_color = config['BACK'][row*3 + col]
            create_sticker(cube, 'BACK', sticker_color)
        if pos.y == 1:
            row, col = get_sticker_index('UP', pos)
            sticker_color = config['UP'][row*3 + col]
            create_sticker(cube, 'UP', sticker_color)
        if pos.y == -1:
            row, col = get_sticker_index('DOWN', pos)
            sticker_color = config['DOWN'][row*3 + col]
            create_sticker(cube, 'DOWN', sticker_color)

# -------------------------------
# Create the individual cubies.
# -------------------------------
CUBES = [Entity(model=cube_model, color=color.dark_gray, position=pos) for pos in SIDE_POSITIONS]

# -------------------------------
# Create sensor entities to detect mouse clicks on each side.
# -------------------------------
def create_sensors():
    def create_sensor(name, pos, scale):
        return Entity(
            name=name,
            position=pos,
            model='cube',
            color=color.dark_gray,
            scale=scale,
            collider='box',
            visible=False
        )
    return {
        'LEFT':   create_sensor(name='LEFT', pos=(-0.99, 0, 0), scale=(1.01, 3.01, 3.01)),
        'FACE':   create_sensor(name='FACE', pos=(0, 0, -0.99), scale=(3.01, 3.01, 1.01)),
        'BACK':   create_sensor(name='BACK', pos=(0, 0, 0.99), scale=(3.01, 3.01, 1.01)),
        'RIGHT':  create_sensor(name='RIGHT', pos=(0.99, 0, 0), scale=(1.01, 3.01, 3.01)),
        'UP':    create_sensor(name='UP', pos=(0, 1, 0), scale=(3.01, 1.01, 3.01)),
        'DOWN': create_sensor(name='DOWN', pos=(0, -1, 0), scale=(3.01, 1.01, 3.01))
    }

sensors = create_sensors()

# -------------------------------
# Rotation functions (modified to accept an angle)
# -------------------------------
def reparent_to_scene():
    global CUBES, PARENT
    for cube in CUBES:
        if cube.parent == PARENT:
            world_pos = Vec3(
                round(cube.world_position.x, 1),
                round(cube.world_position.y, 1),
                round(cube.world_position.z, 1)
            )
            world_rot = cube.world_rotation
            cube.parent = scene
            cube.position = world_pos
            cube.rotation = world_rot
    PARENT.rotation = 0

def toggle_animation_trigger():
    global action_trigger
    action_trigger = not action_trigger

def rotate_side(side_name, angle=90):
    global action_trigger, animation_time
    action_trigger = False
    cube_positions = cubes_side_positions[side_name]
    rotation_axis = rotation_axes[side_name]
    reparent_to_scene()
    for cube in CUBES:
        if cube.position in cube_positions:
            cube.parent = PARENT
    if rotation_axis == 'x':
        PARENT.animate_rotation_x(angle, duration=animation_time)
    elif rotation_axis == 'y':
        PARENT.animate_rotation_y(angle, duration=animation_time)
    elif rotation_axis == 'z':
        PARENT.animate_rotation_z(angle, duration=animation_time)
    invoke(toggle_animation_trigger, delay=animation_time + 0.11)

def rotate_side_without_animation(side_name, angle=90):
    cube_positions = cubes_side_positions[side_name]
    rotation_axis = rotation_axes[side_name]
    reparent_to_scene()
    for cube in CUBES:
        if cube.position in cube_positions:
            cube.parent = PARENT
    if rotation_axis == 'x':
        PARENT.rotation_x = angle
    elif rotation_axis == 'y':
        PARENT.rotation_y = angle
    elif rotation_axis == 'z':
        PARENT.rotation_z = angle

# -------------------------------
# (Un)comment one of these to set the initial state:
# -------------------------------
# (1) Use a random scramble:
# def random_state(rotations=3):
#     for _ in range(rotations):
#         rotate_side_without_animation(random.choice(list(rotation_axes.keys())))
# random_state(rotations=3)

# (2) Or start with the specific initial configuration.
use_initial_config = True
if use_initial_config:
    apply_initial_config(initial_config)

# -------------------------------
# Toggle between action (cube manipulation) and view mode.
# -------------------------------
def toggle_game_mode():
    global action_mode, message
    action_mode = not action_mode
    mode_text = 'ACTION mode ON' if action_mode else 'VIEW mode ON'
    msg = dedent(f"{mode_text} (to switch - press middle mouse button)").strip()
    message.text = msg

# Start in view mode.
toggle_game_mode()

# -------------------------------
# Helper: Parse moves coming from your solver.
# -------------------------------
def parse_move(move):
    # The first character is the face letter.
    face_letter = move[0].upper()
    if face_letter == 'U':
        side_name = 'UP'
    elif face_letter == 'D':
        side_name = 'DOWN'
    elif face_letter == 'L':
        side_name = 'LEFT'
    elif face_letter == 'R':
        side_name = 'RIGHT'
    elif face_letter == 'F':
        side_name = 'FACE'
    elif face_letter == 'B':
        side_name = 'BACK'
    else:
        raise ValueError(f"Invalid move: {move}")
    # Determine the rotation angle.
    if len(move) == 1:
        angle = 90
    elif move[1] == '2':
        angle = 180
    elif move[1] == "'":
        angle = -90
    else:
        angle = 90

    if face_letter in ('B', 'D', 'L'):
        angle = -angle
    return side_name, angle

# -------------------------------
# Input handling
# -------------------------------
def input(key):
    global action_mode, action_trigger, moves_list, move_message
    print(key)
    if key in ('left mouse down', 'right mouse down') and action_mode and action_trigger:
        for hit in mouse.collisions:
            collider_name = hit.entity.name
            if (key == 'left mouse down' and collider_name in ['LEFT', 'RIGHT', 'FACE', 'BACK']) or \
               (key == 'right mouse down' and collider_name in ['UP', 'DOWN']):
                rotate_side(collider_name)
                break
    if key == 'space':
        toggle_game_mode()
    # --- NEW: Execute next solver move when N is pressed ---
    if key == 'n':
        print("this")
        if moves_list:
            move = moves_list.pop(0)
            side, angle = parse_move(move)
            print(f"Performing move: {move} -> side: {side}, angle: {angle}")
            rotate_side(side, angle)
            move_message.text = f"Moves left: {len(moves_list)}"
        else:
            print("No more moves in the sequence.")

app.run()
