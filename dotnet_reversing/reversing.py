#
# Fair warning, this file is a mess on the part of .NET interop. Order matters here for imports.
#

import sys
import typing as t
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import dreadnode as dn
import rigging as rg
from loguru import logger
from pythonnet import load  # type: ignore [import-untyped]

load("coreclr")

import clr  # type: ignore [import-untyped] # noqa: E402

lib_dir = Path(__file__).parent / "lib"
sys.path.append(str(lib_dir))

clr.AddReference("ICSharpCode.Decompiler")
clr.AddReference("Mono.Cecil")

from ICSharpCode.Decompiler import (  # type: ignore [import-not-found] # noqa: E402
    DecompilerSettings,
)
from ICSharpCode.Decompiler.CSharp import (  # type: ignore [import-not-found] # noqa: E402
    CSharpDecompiler,
)
from ICSharpCode.Decompiler.Metadata import (  # type: ignore [import-not-found] # noqa: E402
    MetadataTokenHelpers,
)
from Mono.Cecil import AssemblyDefinition  # type: ignore [import-not-found] # noqa: E402

# Helpers


def _shorten_dotnet_name(name: str) -> str:
    if " " not in name or "(" not in name:
        return name
    return name.split(" ")[-1].split("(")[0]


def _get_decompiler(path: Path | str) -> CSharpDecompiler:
    settings = DecompilerSettings()
    settings.ThrowOnAssemblyResolveErrors = False
    return CSharpDecompiler(str(path), settings)


def _decompile_token(path: Path | str, token: int) -> str:
    entity_handle = MetadataTokenHelpers.TryAsEntityHandle(token.ToUInt32())  # type: ignore [attr-defined]
    return _get_decompiler(path).DecompileAsString(entity_handle)  # type: ignore [no-any-return]


def _find_references(assembly: AssemblyDefinition, search: str) -> list[str]:
    flexible_search_strings = [
        search.lower(),
        search.lower().replace(".", "::"),
        search.lower().replace("::", "."),
    ]

    using_methods: set[str] = set()
    for module in assembly.Modules:
        methods = []
        for module_type in module.Types:
            for method in module_type.Methods:
                methods.append(method)

        for method in methods:
            if not method.HasBody:
                continue

            for instruction in method.Body.Instructions:
                intruction_str = str(instruction.Operand).lower()

                for _search in flexible_search_strings:
                    if _search in intruction_str:
                        using_methods.add(method.FullName)

    return list(using_methods)


def _extract_unique_call_paths(
    tree: dict[str, t.Any],
    current_path: list[str] | None = None,
) -> list[list[str]]:
    if current_path is None:
        current_path = []

    if not tree:  # Leaf node
        return [current_path] if current_path else []

    paths = []
    for method, subtree in tree.items():
        new_path = [method, *current_path]
        paths.extend(_extract_unique_call_paths(subtree, new_path))

    return paths


# Tools

DEFAULT_EXCLUDE = [
    "mscorlib.dll",
]


@dataclass
class DotnetReversing:
    base_path: Path
    binaries: list[str]

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        pattern: str = "**/*",
        exclude: list[str] = DEFAULT_EXCLUDE,
    ) -> "DotnetReversing":
        base_path = Path(path)
        if not base_path.exists():
            raise ValueError(f"Base path does not exist: {base_path}")

        binaries: list[str] = []
        for file_path in base_path.rglob(pattern):
            rel_path = file_path.relative_to(base_path)
            if not any(ex in str(rel_path) for ex in exclude):
                binaries.append(str(rel_path))

        if not binaries:
            raise ValueError(
                f"No binaries found in {base_path} ({pattern})",
            )

        return cls(base_path=base_path, binaries=binaries)

    @cached_property
    def tools(self) -> list[t.Callable[..., t.Any]]:
        return [
            rg.tool(catch=True)(dn.task()(func))
            for func in (
                self.decompile_module,
                self.decompile_type,
                self.decompile_methods,
                self.list_types,
                self.list_methods,
                self.search_for_references,
                self.get_call_flows_to_method,
            )
        ]

    def _resolve_path(self, path: str) -> str:
        rel_path = Path(path)
        full_path = self.base_path / path

        # If we are already in the base path, make it relative
        # before we check anything (can occur with repeated calls)
        if rel_path.is_relative_to(self.base_path):
            rel_path = rel_path.relative_to(self.base_path)

        if str(rel_path) not in self.binaries or not full_path.exists():
            raise ValueError(f"{path} is not available.")

        return str(full_path)

    def decompile_module(self, path: t.Annotated[str, "The binary file path"]) -> str:
        """
        Decompile the entire module and return the decompiled code as a string.
        """
        logger.info(f"decompile_module({path})")
        path = self._resolve_path(path)
        return _get_decompiler(path).DecompileWholeModuleAsString()  # type: ignore [no-any-return]

    def decompile_type(
        self,
        path: t.Annotated[str, "The binary file path"],
        type_name: t.Annotated[str, "The specific type to decompile"],
    ) -> str:
        """
        Decompile a specific type and return the decompiled code as a string.
        """
        logger.info(f"decompile_type({path}, {type_name})")
        path = self._resolve_path(path)
        return _get_decompiler(path).DecompileTypeAsString(type_name)  # type: ignore [no-any-return]

    def decompile_methods(
        self,
        path: t.Annotated[str, "The binary file path"],
        method_names: t.Annotated[list[str], "List of methods to decompile"],
    ) -> dict[str, str]:
        """
        Decompile specific methods and return a dictionary with method names as keys and decompiled code as values.
        """
        logger.info(f"decompile_methods({path}, {method_names})")
        flexible_method_names = [_shorten_dotnet_name(name) for name in method_names]
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)
        methods: dict[str, str] = {}
        for module in assembly.Modules:
            for module_type in module.Types:
                for method in module_type.Methods:
                    if _shorten_dotnet_name(method.FullName) in flexible_method_names:
                        methods[method.FullName] = _decompile_token(path, method.MetadataToken)
        return methods

    def list_types(self, path: t.Annotated[str, "The binary file path"]) -> list[str]:
        """
        List all types in the assembly and return their full names.
        """
        logger.info(f"list_types({path})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)
        return [module_type.FullName for module in assembly.Modules for module_type in module.Types]

    def list_methods(self, path: t.Annotated[str, "The binary file path"]) -> list[str]:
        """
        List all methods in the assembly and return their full names.
        """
        logger.info(f"list_methods({path})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)
        methods: list[str] = []
        for module in assembly.Modules:
            for module_type in module.Types:
                methods.extend([method.FullName for method in module_type.Methods])
        return methods

    def search_for_references(
        self,
        path: t.Annotated[str, "The binary file path"],
        search: t.Annotated[str, "A flexible search string used to check called function names"],
    ) -> list[str]:
        """
        Locate all methods inside the assembly that reference the search string.

        This can be used to locate uses of a specific function or method anywhere in the assembly.
        """
        logger.info(f"search_for_references({path}, {search})")
        path = self._resolve_path(path)
        assembly = AssemblyDefinition.ReadAssembly(path)
        return _find_references(assembly, search)

    def get_call_flows_to_method(
        self,
        paths: t.Annotated[
            list[str],
            "Paths of all .NET assemblies to consider as part of the search",
        ],
        method_name: t.Annotated[str, "Target method name"],
        *,
        max_depth: int = 10,
    ) -> list[list[str]]:
        """
        Find all unique call flows to the target method inside provided assemblies and
        return a nested list of method names representing the call paths.
        """
        logger.info(f"get_call_flows_to_method({paths}, {method_name})")
        assemblies = [AssemblyDefinition.ReadAssembly(self._resolve_path(path)) for path in paths]
        short_target_name = _shorten_dotnet_name(method_name)

        def build_tree(
            method_name: str,
            current_depth: int = 0,
            visited: set[str] | None = None,
        ) -> dict[str, t.Any]:
            visited = visited or set()
            if method_name in visited or current_depth > max_depth:
                return {}

            visited.add(method_name)
            tree = {}

            for assembly in assemblies:
                for caller in _find_references(assembly, method_name):
                    if caller not in visited:
                        tree[caller] = build_tree(
                            _shorten_dotnet_name(caller),
                            current_depth + 1,
                            visited.copy(),
                        )

            return tree

        call_tree = build_tree(short_target_name)
        return _extract_unique_call_paths(call_tree)
