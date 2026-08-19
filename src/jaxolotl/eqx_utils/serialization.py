"""Serialization utilities for PyTrees with optional metadata."""

import base64
import dataclasses
import json
import pickle
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
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


def save_with_treedef(path: Path | str, model: PyTree, metadata: dict | None = None):
    """Serialize a PyTree along with treedef and optional metadata to a file.

    Args:
        path (Path): The path to the file where the PyTree will be saved.
        model (PyTree): The PyTree to serialize.
        metadata (dict): Optional metadata to include with the serialized PyTree (must be JSON-serializable).
    """
    # 1. Get the structure
    treedef = jax.tree.structure(model)

    # 2. Serialize the treedef using pickle, then base64 encode it to store in JSON
    # PyTreeDefs are C++ objects and cannot be purely JSON serialized.
    treedef_bytes = pickle.dumps(treedef)
    treedef_b64 = base64.b64encode(treedef_bytes).decode("ascii")

    if not metadata:
        metadata = {}
    metadata["treedef_b64"] = treedef_b64

    # 3. Write the file
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


def load_from_treedef(path: Path | str) -> PyTree:
    """Load a PyTree from a file from the stored treedef.

    Args:
        path (Path | str): The path to the file from which to load the PyTree.

    Returns:
        PyTree: The loaded PyTree.
    """
    with open(path, "rb") as f:
        # 1. Read the header
        header_line = f.readline().strip()
        metadata = json.loads(header_line)

        # 2. Decode the treedef
        treedef_b64 = metadata["treedef_b64"]
        treedef_bytes = base64.b64decode(treedef_b64)
        treedef = pickle.loads(treedef_bytes)

        # 3. Load the leaves
        # Since we don't have a template, we cannot use eqx.tree_deserialise_leaves.
        # Instead, we know how many leaves the treedef expects, and we know
        # eqx saves them as concatenated .npy files.
        leaves = []
        for _ in range(treedef.num_leaves):
            # jnp.load can read directly from the open file handle
            leaves.append(jnp.load(f))

        # 4. Reconstruct the model
        model = jax.tree.unflatten(treedef, leaves)
    return _reinstantiate(model)


def _reinstantiate(tree: PyTree) -> PyTree:
    """Recursively reinstantiate Equinox modules in a PyTree to ensure the pickled
    class matches the current class definition.

    Args:
        tree (PyTree): The input PyTree.

    Returns:
        PyTree: The re-instantiated PyTree."""
    if isinstance(tree, eqx.Module):
        cls = type(tree)
        init_kwargs = {}
        for field in dataclasses.fields(tree):
            if field.init:
                value = getattr(tree, field.name)
                init_kwargs[field.name] = _reinstantiate(value)
        return cls(**init_kwargs)
    elif _is_named_tuple_instance(tree):
        cls = type(tree)
        init_kwargs = {}
        for field in tree._fields:
            value = getattr(tree, field)
            init_kwargs[field] = _reinstantiate(value)
        return cls(**init_kwargs)
    elif isinstance(tree, list | tuple):
        return type(tree)(_reinstantiate(x) for x in tree)
    elif isinstance(tree, dict):
        return {k: _reinstantiate(v) for k, v in tree.items()}
    else:
        return tree


def _is_named_tuple_instance(x):
    """Check if x is an instance of a namedtuple."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] is not tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) is str for n in f)
