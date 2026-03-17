# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Context path descriptors for type-safe observation/reward bindings.

This module provides FieldPath and NestedField descriptors that enable dual access:
- Class access: Returns path information for configuration
- Instance access: Returns actual tensor values

Usage in view classes:
    class CurrentStateView:
        rigid_body_pos: Tensor = FieldPath()
        
        def __init__(self, state):
            self.rigid_body_pos = state.rigid_body_pos

Usage in experiment configs:
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    
    observation_components = {
        "max_coords_obs": MdpComponent(
            compute_func=compute_humanoid_max_coords_observations,
            dynamic_vars={
                "body_pos": EnvContext.current.rigid_body_pos,  # FieldPath object
            },
        ),
    }
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar, Optional, TYPE_CHECKING


T = TypeVar('T')


class FieldPath(Generic[T]):
    """Descriptor that provides dual path/value access.
    
    On class access, returns self (a FieldPath with .path property).
    On instance access, returns the stored value.
    
    This enables:
    - EnvContext.current.rigid_body_pos -> FieldPath("current.rigid_body_pos")
    - ctx.current.rigid_body_pos -> Tensor (actual value)
    
    Attributes:
        name: Field name (set by __set_name__)
        _parent_path: Dot-separated parent path for nested fields
    """
    
    __slots__ = ('name', '_parent_path')
    
    def __init__(self, parent_path: str = ""):
        """Initialize FieldPath descriptor.
        
        Args:
            parent_path: Dot-separated parent path (e.g., "current" for nested fields)
        """
        self.name: str = ""
        self._parent_path = parent_path
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute."""
        self.name = name
    
    @property
    def path(self) -> str:
        """Full dot-separated path to this field."""
        if self._parent_path:
            return f"{self._parent_path}.{self.name}"
        return self.name
    
    def __get__(self, obj, objtype=None):
        """Get descriptor value.
        
        Args:
            obj: Instance (None for class access)
            objtype: Owner class
        
        Returns:
            - If obj is None (class access): Returns self (FieldPath with .path)
            - If obj is not None (instance access): Returns stored value
        """
        if obj is None:
            # Class access -> return FieldPath descriptor
            return self
        # Instance access -> return stored value
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        """Set field value on instance."""
        obj.__dict__[self.name] = value
    
    def __str__(self) -> str:
        """String representation (returns path)."""
        return self.path
    
    def __repr__(self) -> str:
        """Repr representation."""
        return f"FieldPath('{self.path}')"
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison."""
        if isinstance(other, FieldPath):
            return self.path == other.path
        if isinstance(other, str):
            return self.path == other
        return False
    
    def __hash__(self) -> int:
        """Hash for use in dicts/sets."""
        return hash(self.path)


class NestedField(Generic[T]):
    """Descriptor for nested view objects that propagates parent path.
    
    On class access, returns a path proxy with FieldPaths that include parent path.
    On instance access, returns the nested object instance.
    
    This enables:
    - EnvContext.current -> PathProxy with FieldPaths like "current.rigid_body_pos"
    - ctx.current -> CurrentStateView instance
    
    Attributes:
        nested_class: The class of the nested view
        name: Field name (set by __set_name__)
        _parent_path: Dot-separated parent path
        _path_proxy: Lazy-created proxy class for path access
    """
    
    __slots__ = ('nested_class', 'name', '_parent_path', '_path_proxy')
    
    def __init__(self, nested_class: type, parent_path: str = ""):
        """Initialize NestedField descriptor.
        
        Args:
            nested_class: The class of the nested view (e.g., CurrentStateView)
            parent_path: Dot-separated parent path
        """
        self.nested_class = nested_class
        self.name: str = ""
        self._parent_path = parent_path
        self._path_proxy: Optional[type] = None
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute."""
        self.name = name
    
    @property
    def path(self) -> str:
        """Full dot-separated path to this nested field."""
        if self._parent_path:
            return f"{self._parent_path}.{self.name}"
        return self.name
    
    def _create_path_proxy(self) -> type:
        """Create a proxy class that returns FieldPaths with correct parent path.
        
        This dynamically creates a class where all FieldPath descriptors from the
        nested class are recreated with the updated parent path.
        
        Returns:
            Proxy class with path-aware FieldPath descriptors
        """
        full_path = self.path
        nested_class = self.nested_class
        
        # Create proxy class
        class PathProxy:
            """Proxy class for path access to nested fields."""
            pass
        
        # Copy all FieldPath and NestedField descriptors with updated parent path.
        # We iterate __dict__ directly (not dir + getattr) to avoid triggering
        # the descriptor protocol, which would return PathProxy classes instead
        # of the raw NestedField instances we need to inspect.
        for cls in nested_class.__mro__:
            for attr_name, attr in cls.__dict__.items():
                if attr_name.startswith('_'):
                    continue
                
                if isinstance(attr, FieldPath):
                    # Create new FieldPath with updated parent
                    new_field = FieldPath(parent_path=full_path)
                    new_field.name = attr_name
                    setattr(PathProxy, attr_name, new_field)
                elif isinstance(attr, NestedField):
                    # Recursively handle nested fields
                    new_nested = NestedField(attr.nested_class, parent_path=full_path)
                    new_nested.name = attr_name
                    setattr(PathProxy, attr_name, new_nested)
        
        return PathProxy
    
    def __get__(self, obj, objtype=None):
        """Get descriptor value.
        
        Args:
            obj: Instance (None for class access)
            objtype: Owner class
        
        Returns:
            - If obj is None (class access): Returns path proxy
            - If obj is not None (instance access): Returns nested object
        """
        if obj is None:
            # Class access -> return path proxy
            if self._path_proxy is None:
                self._path_proxy = self._create_path_proxy()
            return self._path_proxy
        # Instance access -> return nested object
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        """Set nested object on instance."""
        obj.__dict__[self.name] = value
    
    def __str__(self) -> str:
        """String representation (returns path)."""
        return self.path
    
    def __repr__(self) -> str:
        """Repr representation."""
        return f"NestedField('{self.path}')"


def resolve_path(ctx: Any, path: str) -> Any:
    """Resolve a dot-separated path to a value from a context instance.
    
    Args:
        ctx: The context instance (e.g., EnvContext)
        path: Dot-separated path string (e.g., "current.rigid_body_pos")
    
    Returns:
        The value at the specified path in the context.
    
    Raises:
        AttributeError: If the path doesn't exist in the context.
    
    Example:
        >>> resolve_path(ctx, "current.rigid_body_pos")
        Tensor([...])
    """
    value = ctx
    for part in path.split("."):
        value = getattr(value, part)
    return value


# Exports
__all__ = [
    "FieldPath",
    "NestedField",
    "resolve_path",
]
