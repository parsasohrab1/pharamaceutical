"""PDB export and interactive 3D pocket viewer (FR-18)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def export_pocket_pdb(pocket: Dict, output_path: str) -> str:
    """Write pocket atoms to a minimal PDB file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "REMARK   HQCA synthetic pocket (PocketGen-compatible placeholder)",
        f"REMARK   sequence length {pocket.get('length', 0)}",
    ]
    for atom in pocket.get("atoms", []):
        idx = atom["index"]
        res = atom.get("residue", "ALA")[:3].upper()
        x, y, z = atom["x"], atom["y"], atom["z"]
        lines.append(
            f"ATOM  {idx:5d}  CA  {res:>3} A{idx:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def generate_pocket_viewer_html(
    pocket: Dict,
    binding_score: float,
    confidence_pct: float,
    output_path: str,
    title: str = "HQCA Pocket Viewer",
) -> str:
    """Standalone Three.js viewer for interactive 3D pocket display."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = pocket.get("atoms", [])
    center = pocket.get("center", [0.0, 0.0, 0.0])
    payload = json.dumps(
        {
            "atoms": atoms,
            "center": list(center),
            "binding_score": binding_score,
            "confidence_pct": confidence_pct,
            "sequence": pocket.get("sequence", ""),
        },
        ensure_ascii=False,
    )
    html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }}
    #info {{ position: absolute; top: 12px; right: 12px; background: rgba(15,23,42,.85);
             padding: 12px 16px; border-radius: 8px; font-size: 14px; z-index: 10; }}
    #canvas-wrap {{ width: 100vw; height: 100vh; }}
    .score {{ font-size: 1.4rem; font-weight: 700; color: #38bdf8; }}
  </style>
</head>
<body>
  <div id="info">
    <div>نمره اتصال: <span class="score">{binding_score:.1f}</span> / 100</div>
    <div>اطمینان: {confidence_pct:.1f}%</div>
    <div style="margin-top:8px;font-size:12px;opacity:.8">ماوس: چرخش | اسکرول: زوم</div>
  </div>
  <div id="canvas-wrap"></div>
  <script type="importmap">
    {{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
  </script>
  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
    const DATA = {payload};
    const wrap = document.getElementById('canvas-wrap');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f172a);
    const camera = new THREE.PerspectiveCamera(55, innerWidth/innerHeight, 0.1, 2000);
    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(innerWidth, innerHeight);
    wrap.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    const cx = DATA.center[0], cy = DATA.center[1], cz = DATA.center[2];
    const geom = new THREE.SphereGeometry(0.6, 16, 16);
    const mat = new THREE.MeshPhongMaterial({{ color: 0x38bdf8 }});
    DATA.atoms.forEach((a, i) => {{
      const m = new THREE.Mesh(geom, mat.clone());
      m.material.color.setHSL((i / Math.max(DATA.atoms.length, 1)) * 0.3 + 0.5, 0.7, 0.55);
      m.position.set(a.x - cx, a.y - cy, a.z - cz);
      scene.add(m);
    }});
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(10, 20, 15);
    scene.add(light, new THREE.AmbientLight(0x404060, 0.6));
    camera.position.set(30, 25, 35);
    controls.target.set(0, 0, 0);
  function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
  animate();
  addEventListener('resize', () => {{
    camera.aspect = innerWidth/innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  }});
  </script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")
    return str(path)
