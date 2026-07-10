"""
Dynamic Recoil Compensation Matrix (for Cronus Zen integration).
Per-weapon vertical/horizontal anti-recoil tables.
Can be used to generate profiles or ZPU-compatible data.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List
import json
from pathlib import Path

@dataclass
class WeaponRecoil:
    name: str = "Default"
    vertical: List[int] = field(default_factory=lambda: [0] * 30)   # per-shot or per-ms values
    horizontal: List[int] = field(default_factory=lambda: [0] * 30)
    notes: str = ""

class RecoilMatrix:
    def __init__(self, profile_path: str = "recoil_profiles.json"):
        self.path = Path(profile_path)
        self.weapons: Dict[str, WeaponRecoil] = {}
        self.load()

    def add_or_update(self, weapon: WeaponRecoil):
        self.weapons[weapon.name] = weapon
        self.save()

    def get(self, name: str) -> WeaponRecoil:
        return self.weapons.get(name, WeaponRecoil(name=name))

    def list_weapons(self) -> List[str]:
        return list(self.weapons.keys())

    def save(self):
        data = {k: asdict(v) for k, v in self.weapons.items()}
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.weapons = {k: WeaponRecoil(**v) for k, v in data.items()}
            except:
                self.weapons = {}
        else:
            # Seed some examples
            self.weapons["AR-15"] = WeaponRecoil("AR-15", vertical=[2,3,2,1]*8, horizontal=[0,1,-1,0]*8)
            self.weapons["SMG"] = WeaponRecoil("SMG", vertical=[1,2,1]*10, horizontal=[-1,0,1]*10)
            self.save()

    def generate_cronus_note(self, weapon_name: str) -> str:
        w = self.get(weapon_name)
        return f"// ScriptedElite Recoil for {w.name}\n// Vertical: {w.vertical[:8]}...\n// Horizontal: {w.horizontal[:8]}...\n// Load into your ZPU / MKZEN profile."
