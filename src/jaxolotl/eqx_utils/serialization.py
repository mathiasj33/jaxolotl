"""Serialization utilities for PyTrees with optional metadata."""

import base64
import json
import pickle
from pathlib import Path

import equinox as eqx
import jax
from jaxtyping import PyTree


def save(path: Path | str, model: PyTree, metadata: dict | None = None):
    """Serialize a PyTree along with optional metadata to a file.

    Args:
        path (Path): The path to the file where the PyTree will be saved.
        model (PyTree): The PyTree to serialize.
        metadata (dict): Optional metadata to include with the serialized PyTree (must be JSON-serializable).
    """
    with open(path, "wb") as f:
        if not metadata:
            metadata = {}
        f.write(json.dumps(metadata, indent=None).encode("utf-8"))
        f.write(b"\n")
        eqx.tree_serialise_leaves(f, model)


def save_with_template(path: Path | str, model: PyTree, metadata: dict | None = None):
    """Serialize a PyTree along with a lightweight template and optional metadata.

    Args:
        path (Path): The path to the file where the PyTree will be saved.
        model (PyTree): The PyTree to serialize.
        metadata (dict): Optional metadata to include with the serialized PyTree (must be JSON-serializable).
    """
    template = jax.tree.map(
        lambda x: (
            jax.ShapeDtypeStruct(x.shape, x.dtype) if isinstance(x, jax.Array) else x
        ),
        model,
    )
    template_bytes = pickle.dumps(template)
    template_b64 = base64.b64encode(template_bytes).decode("ascii")

    metadata = {} if metadata is None else metadata.copy()
    metadata["template_b64"] = template_b64

    with open(path, "wb") as f:
        # Write JSON header followed by a newline
        header = json.dumps(metadata).encode("utf-8")
        f.write(header)
        f.write(b"\n")

        # Write the leaves (weights/arrays) using Equinox
        eqx.tree_serialise_leaves(f, model)


def load_metadata(path: Path | str) -> dict:
    """Load metadata from a file.

    Args:
        path (Path): The path to the file from which to load the metadata.

    Returns:
        dict: The loaded metadata.
    """
    with open(path, "rb") as f:
        metadata = json.loads(f.readline().decode("utf-8"))
    return metadata


def load(path: Path | str, template: PyTree) -> PyTree:
    """Load a PyTree from a file.

    Args:
        path (Path): The path to the file from which to load the PyTree.
        template (PyTree): A template PyTree with the same structure as the one being loaded.

    Returns:
        PyTree: The loaded PyTree.
    """
    with open(path, "rb") as f:
        f.readline()  # Discard metadata line
        model = eqx.tree_deserialise_leaves(f, template)
    return model


def load_from_template(path: Path | str) -> PyTree:
    """Load a PyTree using the lightweight template stored in the file.

    Args:
        path (Path | str): The path to the file from which to load the PyTree.

    Returns:
        PyTree: The loaded PyTree.
    """
    with open(path, "rb") as f:
        header_line = f.readline().strip()
        metadata = json.loads(header_line)

        template_b64 = metadata["template_b64"]
        template_bytes = base64.b64decode(template_b64)
        template = pickle.loads(template_bytes)

        return eqx.tree_deserialise_leaves(f, template)
