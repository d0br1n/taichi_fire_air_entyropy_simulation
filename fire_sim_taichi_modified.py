import taichi as ti

ti.init(arch=ti.cuda)

scene = ti.ui.Scene()
window = ti.ui.Window("test", (700,700))
canvas = window.get_canvas()
camera = ti.ui.Camera()
camera.position(5,2,2)
v = ti.Vector.field(3, dtype= ti.f32, shape=3)
for i in range(3):
    v[i] = ti.Vector([i,i,i])
while window.running:

    camera.track_user_inputs(movement_speed = 0.05, hold_key = ti.ui.RMB, window=window)
    scene.set_camera(camera)
    scene.ambient_light((0.8,0.8,0.))
    scene.mesh(vertices = v, color = ti.Vector([0.0,1.0,1.0]), show_wireframe=True)
    canvas.render(scene)
    window.show()
