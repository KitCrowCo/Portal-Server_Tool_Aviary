"""poses.py - poses for pose-conditioned painting and for the 3D OpenPose editor (open-pose-editor by Yu Zhu, its Save Scene files): drawn as an OpenPose picture, made from joint positions an AI or a person gives, and described in words an AI can read and change.
Positions are the editor's world: centimetres, y up from the floor, x to the viewer's right, z toward the viewer (the editor's camera looks from +z). A figure facing the viewer has its left side at +x."""
import io, json, math, base64
from . import plan as P

# --- Drawing: keypoints as an OpenPose picture ---


COCO18 = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar"]   # the points pose-conditioned painting is trained on (OpenPose COCO order)
BODY25 = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"]
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]   # pairs of COCO18 indices, in OpenPose's drawing order
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

SCENE_TREE = {"five": "torso", "neck": "five", "nose": "neck", "right_eye": "nose", "right_ear": "right_eye", "left_eye": "nose", "left_ear": "left_eye", **{f"{s}_{p}": q if q == "five" else f"{s}_{q}" for s in ("right", "left") for p, q in (("shoulder_inner", "five"), ("shoulder", "shoulder_inner"), ("elbow", "shoulder"), ("wrist", "elbow"), ("hip_inner", "five"), ("hip", "hip_inner"), ("knee", "hip"), ("ankle", "knee"))}}   # each joint's parent in the 3D OpenPose editor's body (open-pose-editor src/body.ts, CreateTemplateBody)
SCENE_JOINTS = {"Nose": "nose", "Neck": "neck", "RShoulder": "right_shoulder", "RElbow": "right_elbow", "RWrist": "right_wrist", "LShoulder": "left_shoulder", "LElbow": "left_elbow", "LWrist": "left_wrist", "RHip": "right_hip", "RKnee": "right_knee", "RAnkle": "right_ankle", "LHip": "left_hip", "LKnee": "left_knee", "LAnkle": "left_ankle", "REye": "right_eye", "LEye": "left_eye", "REar": "right_ear", "LEar": "left_ear"}

def _m(a, b): return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def _local(d: dict) -> list:
    """A three.js object's local matrix from its saved position, rotation (Euler, any order) and scale: T * R * S."""
    (x, y, z), r, (sx, sy, sz) = d.get("position") or (0, 0, 0), d.get("rotation") or (0, 0, 0, "XYZ"), d.get("scale") or (1, 1, 1)
    c, s = [math.cos(v) for v in r[:3]], [math.sin(v) for v in r[:3]]
    axis = {"X": [[1, 0, 0, 0], [0, c[0], -s[0], 0], [0, s[0], c[0], 0], [0, 0, 0, 1]], "Y": [[c[1], 0, s[1], 0], [0, 1, 0, 0], [-s[1], 0, c[1], 0], [0, 0, 0, 1]], "Z": [[c[2], -s[2], 0, 0], [s[2], c[2], 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
    rot = _m(_m(axis[(r[3] if len(r) > 3 else "XYZ")[0]], axis[(r[3] if len(r) > 3 else "XYZ")[1]]), axis[(r[3] if len(r) > 3 else "XYZ")[2]])   # three.js: order "XYZ" is the matrix X * Y * Z
    return _m([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], _m(rot, [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]]))

def _inverse_rigid(m: list) -> list:
    """The inverse of a rotation + translation matrix (a camera's)."""
    r = [[m[j][i] for j in range(3)] for i in range(3)]
    return [[*r[i], -sum(r[i][k] * m[k][3] for k in range(3))] for i in range(3)] + [[0, 0, 0, 1]]

def _scene_world(b: dict) -> dict:
    """One body of an editor scene: each joint's world matrix (the editor's body tree, its saved local transforms)."""
    world = {"torso": _local(b)}
    def at(name):
        if name not in world: world[name] = _m(at(SCENE_TREE[name]), _local(b["child"].get(name) or {}))
        return world[name]
    for n in SCENE_JOINTS.values(): at(n)
    return world

def _scene_people(scene: dict, width: int, height: int) -> list:
    """Every body of a 3D OpenPose editor scene ("Openpose Editor by Yu Zhu", saved with Save Scene) seen through its saved camera (perspective, 60 degrees high, as the editor's), as COCO18 points in fractions of the picture."""
    cam = scene["object"]["camera"]
    view, half = _inverse_rigid(_local({"position": cam["position"], "rotation": cam["rotation"]})), math.tan(math.radians(60) / 2) / (cam.get("zoom") or 1)
    out = []
    for b in scene["object"]["bodies"]:
        w, fig = _scene_world(b), {"_unit": "fraction"}
        for k, n in SCENE_JOINTS.items():
            p = _m(view, w[n])
            x, y, z = p[0][3], p[1][3], p[2][3]
            if z < 0: fig[k] = (0.5 + x / (-z * half * width / height) / 2, 0.5 - y / (-z * half) / 2, 1.0)   # in front of the camera: it looks down its own -z
        out.append(fig)
    return out

def _people(pose, width: int = 768, height: int = 1024) -> list:
    """Every figure in a pose record as {COCO18 name: (x, y, confidence)}. Takes a bridge record ({"pose": ...}), a 3D OpenPose editor scene (Save Scene), named joints ({"joints": {"Nose": [x, y, c]}}), OpenPose JSON ({"people": [{"pose_keypoints_2d": [...]}]}, 18 or 25 points), or a flat list of triples. A figure with "_unit": "fraction" has every coordinate as a fraction of the picture."""
    pose = pose.get("pose", pose) if isinstance(pose, dict) and isinstance(pose.get("pose"), dict) else pose
    if isinstance(pose, dict) and str(pose.get("header", "")).startswith("Openpose Editor"): return _scene_people(pose, width, height)
    if isinstance(pose, dict) and "people" in pose: return [x for p in pose["people"] for x in _people(p.get("pose_keypoints_2d") or [])]
    if isinstance(pose, dict) and isinstance(pose.get("joints"), dict): return [{k: tuple((list(v) + [1.0])[:3]) for k, v in pose["joints"].items() if k in COCO18 and isinstance(v, (list, tuple)) and len(v) >= 2}]
    flat = pose.get("keypoints") if isinstance(pose, dict) else pose
    if isinstance(flat, list) and flat and not isinstance(flat[0], (list, tuple)): flat = [flat[i:i + 3] for i in range(0, len(flat) - 2, 3)]
    if not isinstance(flat, list) or len(flat) not in (18, 25): raise ValueError(f"pose_draw: not a pose it knows (an editor scene, named joints, OpenPose people, or 18 / 25 keypoints): {str(pose)[:200]}")
    names = COCO18 if len(flat) == 18 else BODY25
    return [{n: tuple((list(v) + [1.0])[:3]) for n, v in zip(names, flat) if n in COCO18}]

def pose_png(pose, width: int = 0, height: int = 0) -> bytes:
    """A pose as the picture pose-conditioned painting reads: coloured limbs and points on black, OpenPose's colours and order. Points with confidence 0 are left out; coordinates of at most 1 are fractions of the size. The size: width x height, else what the record says ("canvas" {width, height}), else 768 x 1024."""
    from PIL import Image, ImageDraw   # here: only this step needs it (the portal's image already has it)
    inner = pose.get("pose", pose) if isinstance(pose, dict) else {}
    canvas = (inner.get("canvas") if isinstance(inner, dict) else None) or {}
    w, h = int(width or canvas.get("width") or 768), int(height or canvas.get("height") or 1024)
    img = Image.new("RGB", (w, h), (0, 0, 0))
    d, stick = ImageDraw.Draw(img), max(2, round(min(w, h) / 128))
    for fig in _people(pose, w, h):
        frac = fig.pop("_unit", "") == "fraction"
        xy = {n: (x * w if frac or x <= 1 else x, y * h if frac or y <= 1 else y) for n, (x, y, c) in fig.items() if c and c > 0}
        for i, (a, b) in enumerate(LIMBS):
            if COCO18[a] in xy and COCO18[b] in xy: d.line([xy[COCO18[a]], xy[COCO18[b]]], fill=tuple(int(c * 0.6) for c in COLORS[i]), width=stick * 2)
        for i, n in enumerate(COCO18):
            if n in xy: d.ellipse([xy[n][0] - stick, xy[n][1] - stick, xy[n][0] + stick, xy[n][1] + stick], fill=COLORS[i])
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()

async def pose_draw(config: dict, inputs: dict, ctx) -> dict:
    """A pose record drawn as an OpenPose picture. config: {"pose": the record (a bridge's pose_read answer, an editor scene, OpenPose JSON, named joints, or {"png": base64} given as it is), "width", "height", "give": {"b64": name}}. Gives the PNG as base64."""
    c = P.render({k: v for k, v in config.items() if k != "give"}, inputs)
    if isinstance(c["pose"], dict) and c["pose"].get("png"): return {(config.get("give") or {}).get("b64", "b64"): c["pose"]["png"]}   # a pose picture the editor made (the bridge gives a .png as {"png": base64}): used as it is
    return {(config.get("give") or {}).get("b64", "b64"): base64.b64encode(pose_png(c["pose"], int(c.get("width") or 0), int(c.get("height") or 0))).decode()}


# --- Scenes: joint positions to an editor scene and back ---

TEMPLATE = {"position": [0, 115, 0], "child": {"five": [0, 23, 0], "nose": [0, 20, 14], "right_eye": [-3, 3, -3], "right_ear": [-4, -3, -8], "left_eye": [3, 3, -3], "left_ear": [4, -3, -8], "right_shoulder": [-17, 0, 0], "right_elbow": [0, -25, 0], "right_wrist": [0, -25, 0], "left_shoulder": [17, 0, 0], "left_elbow": [0, -25, 0], "left_wrist": [0, -25, 0], "right_hip": [-10, -46, 0], "right_knee": [0, -40, 0], "right_ankle": [0, -36, 0], "left_hip": [10, -46, 0], "left_knee": [0, -40, 0], "left_ankle": [0, -36, 0]}}   # the editor's new body (src/body.ts CreateTemplateBody): what it loads when a scene names no position
CAMERA = {"position": [0, 100, 200], "rotation": [0, 0, 0, "XYZ"], "target": [0, 100, 0], "near": 0.1, "far": 10000, "zoom": 1}   # the editor's own starting view
BONES = [("neck", "nose", "head"), *((f"{s}_{a}", f"{s}_{b}", f"{s} {w}") for s in ("right", "left") for a, b, w in (("shoulder", "elbow", "upper arm"), ("elbow", "wrist", "forearm"), ("hip", "knee", "thigh"), ("knee", "ankle", "shin")))]   # (joint turned, the joint it points at, what it is called), parents before children

def _v(a, b): return [b[i] - a[i] for i in range(3)]
def _n(v): return [x / (math.sqrt(sum(y * y for y in v)) or 1) for x in v]
def _x(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
def _dot(a, b): return sum(x * y for x, y in zip(a, b))
def _mv(r, v): return [sum(r[i][k] * v[k] for k in range(3)) for i in range(3)]
def _t(r): return [[r[j][i] for j in range(3)] for i in range(3)]
def _mm(a, b): return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def _align(a, b) -> list:
    """The smallest rotation taking direction a to direction b (Rodrigues: I + K + K^2 / (1 + cos)); no twist about the bone, which a keypoint cannot show."""
    a, b = _n(a), _n(b)
    c = _dot(a, b)
    if c < -0.999999: k = _n(_x(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])); return [[2 * k[i] * k[j] - (i == j) for j in range(3)] for i in range(3)]   # opposite: half a turn about any axis across it
    k = _x(a, b)
    K = [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]
    KK = _mm(K, K)
    return [[(i == j) + K[i][j] + KK[i][j] / (1 + c) for j in range(3)] for i in range(3)]

def _euler(r) -> list:
    """A rotation matrix as three.js Euler angles, order XYZ (Euler.setFromRotationMatrix)."""
    y = math.asin(max(-1.0, min(1.0, r[0][2])))
    return [math.atan2(-r[1][2], r[2][2]), y, math.atan2(-r[0][1], r[0][0]), "XYZ"] if abs(r[0][2]) < 0.9999999 else [math.atan2(r[2][1], r[1][1]), y, 0.0, "XYZ"]

def _joints_in(joints) -> list:
    """Joint positions as given (JSON text or an object), one body or several: {name: [x, y, z]}, {"joints": {...}}, a list of either, or {"bodies": [...]}; COCO18 names (RElbow) or the editor's (right_elbow), centimetres, z may be left out (0). Answers one {editor name: [x, y, z]} per body."""
    j = json.loads(joints) if isinstance(joints, str) else joints
    j = j.get("bodies", j) if isinstance(j, dict) else j
    out = [b.get("joints", b) if isinstance(b, dict) else b for b in (j if isinstance(j, list) else [j])]
    if not out or not all(isinstance(b, dict) for b in out): raise ValueError(f"pose_scene: joints are an object of name: [x, y, z], or a list of such objects (one per body) - got {str(joints)[:200]}")
    return [{SCENE_JOINTS.get(k, k): [float(x) for x in (list(v) + [0, 0])[:3]] for k, v in b.items() if (SCENE_JOINTS.get(k, k) in SCENE_TREE) and isinstance(v, (list, tuple)) and len(v) >= 2} for b in out]

def _posed(j: dict, body: dict) -> dict:
    """One body (a copy of a base body) with its bones pointed the way the joints say: only directions are taken, so it keeps its proportions; joints left out keep the body's own turn; it stands where the neck is given."""
    child, pos = body.setdefault("child", {}), lambda n: (child.get(n) or {}).get("position") or TEMPLATE["child"].get(n, [0, 0, 0])
    for g in ("five", "right_shoulder_inner", "left_shoulder_inner", "right_hip_inner", "left_hip_inner"): child.setdefault(g, {}).pop("rotation", None)   # groups between joints stay unturned: each turn below is of a joint
    rt = [row[:3] for row in _local({"rotation": body["rotation"]})[:3]] if body.get("rotation") else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if all(k in j for k in ("right_shoulder", "left_shoulder", "neck", "right_hip", "left_hip")):
        x = _n(_v(j["right_shoulder"], j["left_shoulder"]))
        y = _n(_v([(a + b) / 2 for a, b in zip(j["right_hip"], j["left_hip"])], j["neck"]))
        z = _n(_x(x, y))
        y = _x(z, x)
        rt = [[x[i], y[i], z[i]] for i in range(3)]   # columns: the body's own right-to-left, hips-to-neck, back-to-front
        body["rotation"] = _euler(rt)
    if "neck" in j: body["position"] = _v(_mv(rt, pos("five")), j["neck"])   # the torso's origin sits below the neck by "five"
    frame = {"torso": rt, "five": rt, "right_shoulder_inner": rt, "left_shoulder_inner": rt, "right_hip_inner": rt, "left_hip_inner": rt}
    for a, b, _ in BONES:
        parent = frame[SCENE_TREE[a]]
        if a in j and b in j: r = _align(pos(b), _mv(_t(parent), _v(j[a], j[b]))); child.setdefault(a, {})["rotation"] = _euler(r)
        else: r = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] if not (child.get(a) or {}).get("rotation") else [row[:3] for row in _local({"rotation": child[a]["rotation"]})[:3]]
        frame[a] = _mm(parent, r)
    return body

def scene_from_joints(joints, base: dict = None) -> dict:
    """An editor scene with one body per set of joints. Bone lengths, the camera and the links between joints come from base (an editor scene: its first body for the first set, its second for the second, its last for any more), else from the editor's new body; the editor loads it as its own."""
    fresh = {"position": TEMPLATE["position"], "child": {k: {"position": v} for k, v in TEMPLATE["child"].items()}}
    bases = (base or {}).get("object", {}).get("bodies") or [fresh]
    bodies = [_posed(j, json.loads(json.dumps(bases[min(i, len(bases) - 1)]))) for i, j in enumerate(_joints_in(joints))]
    return {"header": "Openpose Editor by Yu Zhu", "version": (base or {}).get("version", "0.1.23"), "object": {"bodies": bodies, "camera": (base or {}).get("object", {}).get("camera") or CAMERA}, "setting": {}}

def _words(d) -> str:
    """A direction in plain words, strongest part first: up / down, to the viewer's left / right, toward / away from the viewer."""
    parts = sorted(((abs(d[1]), "up" if d[1] > 0 else "down"), (abs(d[0]), "to the viewer's right" if d[0] > 0 else "to the viewer's left"), (abs(d[2]), "toward the viewer" if d[2] > 0 else "away from the viewer")), reverse=True)
    return ", ".join(f"{w} {round(100 * a)}%" for a, w in parts if a >= 0.15)

def describe(pose) -> dict:
    """A pose as an AI reads it: every joint's position (centimetres, the editor's world) and each bone's direction and each elbow's and knee's bend in words. Takes an editor scene (or a bridge record holding one)."""
    scene = pose.get("pose", pose) if isinstance(pose, dict) and isinstance(pose.get("pose"), dict) else pose
    if not (isinstance(scene, dict) and str(scene.get("header", "")).startswith("Openpose Editor")): raise ValueError(f"pose_describe: reads editor scenes (Save Scene files) - got {str(pose)[:200]}")
    out = []
    for i, b in enumerate(scene["object"]["bodies"]):
        w = _scene_world(b)
        at = {n: [round(w[n][r][3], 1) for r in range(3)] for n in SCENE_JOINTS.values()}
        rt = [row[:3] for row in _local(b)[:3]]
        lines = [f"""body {i + 1}: neck at {at["neck"]}, faces {_words(_mv(rt, [0, 0, 1]))}; its spine points {_words(_mv(rt, [0, 1, 0]))}"""]
        lines += [f"""{w_}: points {_words(_n(_v(at[a], at[c])))}""" for a, c, w_ in BONES]
        lines += [f"""{s} {j}: bent {round(math.degrees(math.acos(max(-1, min(1, _dot(_n(_v(at[f"{s}_{a}"], at[f"{s}_{j}"])), _n(_v(at[f"{s}_{j}"], at[f"{s}_{c}"]))))))))} degrees (0: straight)""" for s in ("right", "left") for a, j, c in (("shoulder", "elbow", "wrist"), ("hip", "knee", "ankle"))]
        out.append({"joints": {k: at[n] for k, n in SCENE_JOINTS.items()}, "text": "\n".join(lines)})
    return {"bodies": out, "text": "\n\n".join(o["text"] for o in out), "joints": out[0]["joints"] if out else {}}

async def pose_scene(config: dict, inputs: dict, ctx) -> dict:
    """An editor scene made from joint positions. config: {"joints": {name: [x, y, z]} or its JSON text (COCO18 names like RElbow, or the editor's like right_elbow), "base": an editor scene whose proportions and camera are kept (optional), "give": {"scene": name}}."""
    c = P.render({k: v for k, v in config.items() if k != "give"}, inputs)
    base = c.get("base") if isinstance(c.get("base"), dict) and c["base"].get("object") else None
    return {(config.get("give") or {}).get("scene", "scene"): scene_from_joints(c["joints"], base)}

async def pose_describe(config: dict, inputs: dict, ctx) -> dict:
    """A pose described for an AI to read and change. config: {"pose": an editor scene or a bridge record holding one, "give": {"text": name, "joints": name}}."""
    d, g = describe(P.render(config["pose"], inputs)), config.get("give") or {}
    return {g.get("text", "text"): d["text"], g.get("joints", "joints"): d["joints"]}

STEPS = (pose_draw, pose_scene, pose_describe)
