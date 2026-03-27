from models.pinn import PINN


_REGISTRY = {
    "pinn": PINN,
}


def get_model(name: str):
    key = (name or "pinn").lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    return _REGISTRY[key]
