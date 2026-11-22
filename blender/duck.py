import bpy
import bmesh
import json
import math
import os
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

COLORS = {
    "body": (1.0, 0.82, 0.12),
    "beak": (1.0, 0.45, 0.08),
    "eye": (0.06, 0.05, 0.08),
}

bpy.ops.wm.read_factory_settings(use_empty=True)


def material(name):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*COLORS[name], 1.0)
    return mat


def sphere(name, part, location, scale, segments=14, rings=8):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material(part))
    bpy.ops.object.shade_smooth()
    return obj


# body: a squashed sphere with the back pulled up into a tail
body = sphere("body", "body", (0, 0, 0.42), (0.62, 0.48, 0.38), segments=16, rings=10)
for v in body.data.vertices:
    back = max(0.0, -v.co.x - 0.15)
    if v.co.z > -0.05:
        v.co.z += back * back * 1.1
        v.co.x -= back * 0.25
    # flatten the belly a little so it sits in the water
    if v.co.z < -0.2:
        v.co.z = -0.2 + (v.co.z + 0.2) * 0.6

head = sphere("head", "body", (0.3, 0, 0.9), (0.3, 0.3, 0.29))
beak = sphere("beak", "beak", (0.6, 0, 0.84), (0.16, 0.13, 0.05), segments=10, rings=6)
eye_l = sphere("eye_l", "eye", (0.52, 0.17, 0.97), (0.045, 0.045, 0.055), segments=8, rings=5)
eye_r = sphere("eye_r", "eye", (0.52, -0.17, 0.97), (0.045, 0.045, 0.055), segments=8, rings=5)
wing_l = sphere("wing_l", "body", (-0.05, 0.43, 0.5), (0.28, 0.07, 0.15), segments=10, rings=6)
wing_r = sphere("wing_r", "body", (-0.05, -0.43, 0.5), (0.28, 0.07, 0.15), segments=10, rings=6)
for wing in (wing_l, wing_r):
    wing.rotation_euler = (0, math.radians(-18), 0)

# join everything into one mesh
bpy.ops.object.select_all(action="DESELECT")
parts = [body, head, beak, eye_l, eye_r, wing_l, wing_r]
for p in parts:
    p.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
duck = bpy.context.active_object
duck.name = "duck"
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# triangulate and export per-corner data (smooth normals, flat per-part colour)
mesh = duck.data
bm = bmesh.new()
bm.from_mesh(mesh)
bmesh.ops.triangulate(bm, faces=bm.faces[:])
bm.to_mesh(mesh)
bm.free()
mesh.calc_loop_triangles()
if hasattr(mesh, "calc_normals_split"):
    mesh.calc_normals_split()

positions, normals, colors, indices = [], [], [], []
lookup = {}


def to_y_up(v):
    # blender is z-up; the app is y-up with the duck facing +x
    return (round(v[0], 5), round(v[2], 5), round(-v[1], 5))


for tri in mesh.loop_triangles:
    col = COLORS[mesh.materials[tri.material_index].name]
    for li in tri.loops:
        loop = mesh.loops[li]
        p = to_y_up(mesh.vertices[loop.vertex_index].co)
        n = to_y_up(loop.normal)
        key = (p, n, col)
        if key not in lookup:
            lookup[key] = len(positions) // 3
            positions.extend(p)
            normals.extend(n)
            colors.extend(round(c, 4) for c in col)
        indices.append(lookup[key])

out = {"positions": positions, "normals": normals, "colors": colors, "indices": indices}
os.makedirs(os.path.join(ROOT, "public"), exist_ok=True)
with open(os.path.join(ROOT, "public", "duck.json"), "w") as f:
    json.dump(out, f, separators=(",", ":"))

print(f"duck: {len(positions) // 3} verts, {len(indices) // 3} tris")
bb = [duck.matrix_world @ Vector(c) for c in duck.bound_box]
print("bounds min", [round(min(c[i] for c in bb), 3) for i in range(3)], "max", [round(max(c[i] for c in bb), 3) for i in range(3)])

# a quick preview render so the model can be checked without opening blender
for mat in mesh.materials:
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = mat.diffuse_color
cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.2
cam = bpy.data.objects.new("cam", cam_data)
bpy.context.scene.collection.objects.link(cam)
cam.location = (2.6, -2.6, 2.4)
cam.rotation_euler = (Vector((0, 0, 0.5)) - cam.location).to_track_quat("-Z", "Y").to_euler()
bpy.context.scene.camera = cam
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(30))
bpy.context.scene.collection.objects.link(sun)
world = bpy.data.worlds.new("w")
world.color = (0.35, 0.55, 0.7)
bpy.context.scene.world = world
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.display.shading.show_object_outline = True
scene.render.resolution_x = 480
scene.render.resolution_y = 480
scene.render.filepath = os.path.join(HERE, "duck-preview.png")
bpy.ops.render.render(write_still=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "duck.blend"))
