import pygame as pg
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import random

def create_shader(vertex_filepath: str, fragment_filepath: str) -> int:
    with open(vertex_filepath, 'r') as f:
        vertex_src = f.readlines()

    with open(fragment_filepath, 'r') as f:
        fragment_src = f.readlines()

    shader = compileProgram(
        compileShader(vertex_src, GL_VERTEX_SHADER),
        compileShader(fragment_src, GL_FRAGMENT_SHADER)
    )

    return shader

class Entity:
    def __init__(self, position: list[float], eulers: list[float]):
        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)
        self.is_rolling = False
        self.rotation_speed = 2 
        self.revolving_speed = 2
        self.roll_counter = 0
        self.roll_duration = 300
        self.target_side = random.randint(1, 6)
        self.stop_all_rotation = False

    def update(self) -> None:
        if self.is_rolling:
            # Increment rotation angles for all three axes
            self.eulers[0] += self.revolving_speed
            self.eulers[1] += self.rotation_speed
            self.eulers[2] += self.revolving_speed

            if self.eulers[0] > 360:
                self.eulers[0] -= 360
            if self.eulers[1] > 360:
                self.eulers[1] -= 360
            if self.eulers[2] > 360:
                self.eulers[2] -= 360

            # Update roll progress
            roll_progress = self.roll_counter / self.roll_duration

            # Calculate the angle needed to show each face during rolling
            target_rotation = (self.target_side - 1) * (360 / 6)
            

            # Interpolate between the current rotation and the target rotation
            self.eulers[1] = np.interp(roll_progress, [0, 1], [0, target_rotation + 360])

            self.roll_counter += 1

            if self.roll_counter >= self.roll_duration:
                # Stop rolling and rotations when the target side is reached
                self.is_rolling = False
                self.roll_counter = 0

                self.target_side = random.randint(1, 6)

        # Apply revolving transformation only if not stopped
        if not self.stop_all_rotation and self.is_rolling:
            self.eulers[0] += self.revolving_speed
            if self.eulers[0] > 360:
                self.eulers[0] -= 360

    def get_target_side(self) -> int:
        if not self.is_rolling:
            # Calculate the side based on the rolling rotation
            normalized_rotation = self.eulers[1] % 360
            if normalized_rotation < 0:
                normalized_rotation += 360

            # Map the rotation to a specific side
            side_angle = 360 / 6  # Assuming a six-sided die
            target_side = int(normalized_rotation / side_angle) + 1

            return target_side

    def get_model_transform(self) -> np.ndarray:
        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)

        if self.is_rolling:
            # Apply rolling transformation
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_axis_rotation(
                    axis=[0, 1, 1],
                    theta=np.radians(self.eulers[1]),
                    dtype=np.float32
                )
            )

        # Apply revolving transformation only if not stopped
        if not self.stop_all_rotation:
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_axis_rotation(
                    axis=[0, 0, 1],
                    theta=np.radians(self.eulers[0]),
                    dtype=np.float32
                )
            )

        # Orient the dice to stand vertically or horizontally when it stops
        if not self.is_rolling and not self.stop_all_rotation:
            if self.target_side % 2 == 0:  # If target side is even, stop horizontally
                model_transform = pyrr.matrix44.multiply(
                    m1=model_transform,
                    m2=pyrr.matrix44.create_from_axis_rotation(
                        axis=[1, 0, 0],
                        theta=np.radians(90),
                        dtype=np.float32
                    )
                )
            else:  # If target side is odd, stop vertically
                model_transform = pyrr.matrix44.multiply(
                    m1=model_transform,
                    m2=pyrr.matrix44.create_from_axis_rotation(
                        axis=[0, 1, 1],
                        theta=np.radians(90),
                        dtype=np.float32
                    )
                )

            # Determine the rotation needed to face the target side
            target_rotation = (self.target_side - 1) * (360 / 6)
            # Apply the rotation to face the target side
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_axis_rotation(
                    axis=[0, 1, 1],
                    theta=np.radians(target_rotation),
                    dtype=np.float32
                )
            )

        return pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec=np.array(self.position), dtype=np.float32
            )
        )

class Mesh:
    def __init__(self, filename: str):
        vertices = self.load_mesh(filename)
        self.vertex_count = len(vertices) // 8
        vertices = np.array(vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

    def load_mesh(self, filename: str) -> list[float]:
        v = []
        vt = []
        vn = []
        vertices = []

        with open(filename, "r") as file:
            line = file.readline()

            while line:
                words = line.split(" ")
                if words[0] == "v":
                    v.append(self.read_vertex_data(words))
                elif words[0] == "vt":
                    vt.append(self.read_texcoord_data(words))
                elif words[0] == "vn":
                    vn.append(self.read_normal_data(words))
                elif words[0] == "f":
                    self.read_face_data(words, v, vt, vn, vertices)

                line = file.readline()

        return vertices

    def read_vertex_data(self, words: list[str]) -> list[float]:
        return [float(words[1]), float(words[2]), float(words[3])]

    def read_texcoord_data(self, words: list[str]) -> list[float]:
        return [float(words[1]), float(words[2])]

    def read_normal_data(self, words: list[str]) -> list[float]:
        return [float(words[1]), float(words[2]), float(words[3])]

    def read_face_data(self, words: list[str], v: list[list[float]], vt: list[list[float]], vn: list[list[float]],
                       vertices: list[float]) -> None:
        triangle_count = len(words) - 3

        for i in range(triangle_count):
            self.make_corner(words[1], v, vt, vn, vertices)
            self.make_corner(words[2 + i], v, vt, vn, vertices)
            self.make_corner(words[3 + i], v, vt, vn, vertices)

    def make_corner(self, corner_description: str, v: list[list[float]], vt: list[list[float]], vn: list[list[float]],
                    vertices: list[float]) -> None:
        v_vt_vn = corner_description.split("/")

        for element in v[int(v_vt_vn[0]) - 1]:
            vertices.append(element)
        for element in vt[int(v_vt_vn[1]) - 1]:
            vertices.append(element)
        for element in vn[int(v_vt_vn[2]) - 1]:
            vertices.append(element)

    def arm_for_drawing(self) -> None:
        glBindVertexArray(self.vao)

    def draw(self) -> None:
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

    def destroy(self) -> None:
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))

class Material:
    def __init__(self, filepath: str):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        image = pg.image.load(filepath).convert()
        image_width, image_height = image.get_rect().size
        img_data = pg.image.tostring(image, 'RGBA')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self) -> None:
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self) -> None:
        glDeleteTextures(1, (self.texture,))

class App:
    def __init__(self):
        self._set_up_pygame()
        self._set_up_timer()
        self._set_up_opengl()
        self._create_assets()
        # Set view_matrix before calling _set_onetime_uniforms
        self.camera_position = np.array([0, 0, 0], dtype=np.float32)
        self.camera_target = np.array([0, 0, -1], dtype=np.float32)
        self.camera_up = np.array([0, 1, 0], dtype=np.float32)
        self.view_matrix = pyrr.matrix44.create_look_at(
            self.camera_position,
            self.camera_target,
            self.camera_up,
            dtype=np.float32
        )
        self._set_onetime_uniforms()
        self._get_uniform_locations()

    def _set_up_pygame(self) -> None:
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((640, 480), pg.OPENGL | pg.DOUBLEBUF)

    def _set_up_timer(self) -> None:
        self.clock = pg.time.Clock()

    def _set_up_opengl(self) -> None:
        glClearColor(0.1, 0.2, 0.2, 1)
        glEnable(GL_DEPTH_TEST)

    def _create_assets(self) -> None:
        self.cube = Entity(
            position=[0, 0, -9],
            eulers=[0, 0, 0]
        )
        self.cube_mesh = Mesh("models/real_dice.obj")
        self.wood_texture = Material("gfx/real.png")
        self.shader = create_shader(
            vertex_filepath="shaders/vertex.txt",
            fragment_filepath="shaders/fragment.txt"
        )

    def _set_onetime_uniforms(self) -> None:
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)

        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "view"),
            1, GL_FALSE, self.view_matrix
        )
        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=60,  # Adjust this value to a smaller angle
            aspect=640 / 480,
            near=0.1,
            far=10,
            dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )

    def _get_uniform_locations(self) -> None:
        glUseProgram(self.shader)
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")

    def _handle_input(self) -> None:
        keys = pg.key.get_pressed()
        if keys[pg.K_r]:
            self.cube.is_rolling = True
            # Randomize the target side when 'r' is pressed
            self.cube.target_side = random.randint(1, 6)

    def run(self) -> None:
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            self._handle_input()
            self.cube.update()

            if not self.cube.is_rolling:
                stopped_side = self.cube.get_target_side()
                print(f"The cube has stopped at side {stopped_side}")

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.shader)

            glUniformMatrix4fv(
                self.modelMatrixLocation, 1, GL_FALSE,
                self.cube.get_model_transform()
            )
            self.wood_texture.use()
            self.cube_mesh.arm_for_drawing()
            self.cube_mesh.draw()

            self.print_debug_info()

            pg.display.flip()

            self.clock.tick(60)

    def quit(self) -> None:
        self.cube_mesh.destroy()
        self.wood_texture.destroy()
        glDeleteProgram(self.shader)
        pg.quit()

    def print_debug_info(self) -> None:
        print(f"Is rolling: {self.cube.is_rolling}")
        print(f"Eulers: {self.cube.eulers}")
        target_side = self.cube.get_target_side()
        print(f"Target side: {target_side}")


if __name__ == "__main__":
    my_app = App()
    my_app.run()
    my_app.quit()
