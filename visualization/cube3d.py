from ursina import *
from textwrap import dedent
import random
import kociemba




def start(initial_config_string: str):
    app = Ursina(window_title='Rubik\'s Cube Solver',borderless=False, fullscreen=False, position=(35,35))
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
    # nonlocal variables and state
    # -------------------------------
    cube_model = 'models/custom_cube'
    CUBES = []
    PARENT = Entity()  # used as a temporary parent for rotations
    animation_time = 1.25
    action_trigger = True  # used to block new moves while an animation is running

    animation_curve = curve.in_out_expo

    # -------------------------------
    # INITIAL CONFIGURATION
    # -------------------------------
    def cube_string_to_config(cube_string):
        if len(cube_string) != 54:
            raise ValueError("Cube string must be exactly 54 characters long.")
        
        letter_to_color = {
            'U': color.white,
            'R': color.red,
            'F': color.green,
            'D': color.yellow,
            'L': color.orange,
            'B': color.blue
        }
        
        config = {
            'UP':    [letter_to_color[c] for c in cube_string[0:9]],
            'RIGHT': [letter_to_color[c] for c in cube_string[9:18]],
            'FACE':  [letter_to_color[c] for c in cube_string[18:27]],
            'DOWN':  [letter_to_color[c] for c in cube_string[27:36]],
            'LEFT':  [letter_to_color[c] for c in cube_string[36:45]],
            'BACK':  [letter_to_color[c] for c in cube_string[45:54]]
        }
        return config

    initial_config = cube_string_to_config(initial_config_string)
    solver_moves = kociemba.solve(initial_config_string)
    moves_list = solver_moves.split()
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

    cubes_side_positions = {
        'LEFT': LEFT,
        'DOWN': DOWN,
        'FACE': FACE,
        'BACK': BACK,
        'RIGHT': RIGHT,
        'UP': UP
    }

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
        if face == 'LEFT':
            row = 1 - pos.y
            col = 1 - pos.z
            return (int(row), int(col))
        elif face == 'RIGHT':
            row = 1 - pos.y
            col = pos.z + 1
            return (int(row), int(col))
        elif face == 'FACE':
            row = 1 - pos.y
            col = pos.x + 1
            return (int(row), int(col))
        elif face == 'BACK':
            row = 1 - pos.y
            col = 1 - pos.x
            return (int(row), int(col))
        elif face == 'UP':
            row = 1 - pos.z
            col = pos.x + 1
            return (int(row), int(col))
        elif face == 'DOWN':
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

    # Apply the initial cube configuration.
    apply_initial_config(initial_config)

    # -------------------------------
    # Rotation functions
    # -------------------------------
    def reparent_to_scene():
        nonlocal CUBES, PARENT
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
        nonlocal action_trigger
        action_trigger = not action_trigger

    def rotate_side(side_name, angle=90):
        nonlocal action_trigger, animation_time
        action_trigger = False
        cube_positions = cubes_side_positions[side_name]
        rotation_axis = rotation_axes[side_name]
        reparent_to_scene()
        for cube in CUBES:
            if cube.position in cube_positions:
                cube.parent = PARENT
        if rotation_axis == 'x':
            PARENT.animate_rotation_x(angle, duration=animation_time, curve=animation_curve)
        elif rotation_axis == 'y':
            PARENT.animate_rotation_y(angle, duration=animation_time, curve=animation_curve)
        elif rotation_axis == 'z':
            PARENT.animate_rotation_z(angle, duration=animation_time, curve=animation_curve)
        invoke(toggle_animation_trigger, delay=animation_time + 0.1)

    # -------------------------------
    # Parse moves from the solver.
    # -------------------------------
    def parse_move(move):
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
    class InputHandler(Entity):
        def input(self, key):
            nonlocal action_trigger, moves_list, move_message
            if key == 'n':
                # Only process the move if no animation is ongoing.
                if not action_trigger:
                    return
                if moves_list:
                    move = moves_list.pop(0)
                    side, angle = parse_move(move)
                    print(f"Performing move: {move} -> side: {side}, angle: {angle}")
                    rotate_side(side, angle)
                    move_message.text = f"Moves left: {len(moves_list)}"
                else:
                    print("No more moves in the sequence.")

    InputHandler()
    app.run()


if __name__ == '__main__':
    start("DDFUUBRFFLRRDRRDLRDUULFLLFLBRBDDFDDBLBFLLUBBUUFFUBBURR")