import ast
import sys
from collections import defaultdict

def analyze_file(file_path):
    """
    Parses a Python file and builds a call graph and import map.
    Returns (call_graph, local_functions, import_map).
    """
    with open(file_path, "r") as f:
        tree = ast.parse(f.read(), filename=file_path)

    call_graph = defaultdict(list)
    local_functions = set()
    import_map = {}
    
    # First pass: find all function definitions and imports
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            local_functions.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_map[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_map[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    # Second pass: build the call graph
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            current_function = node.name
            for call_node in ast.walk(node):
                if isinstance(call_node, ast.Call):
                    # Reconstruct the called function's name (e.g., 'np.array')
                    func_path = []
                    curr = call_node.func
                    while isinstance(curr, ast.Attribute):
                        func_path.insert(0, curr.attr)
                        curr = curr.value
                    if isinstance(curr, ast.Name):
                        func_path.insert(0, curr.id)
                    
                    call_name = ".".join(func_path)
                    if call_name not in call_graph[current_function]:
                        call_graph[current_function].append(call_name)

    return call_graph, local_functions, import_map

def resolve_dependencies_dfs(root, call_graph, resolved_deps):
    """
    Recursively resolves dependencies using DFS.
    """
    if root in resolved_deps:
        return
    resolved_deps.append(root)
    for dep in call_graph.get(root, []):
        resolve_dependencies_dfs(dep, call_graph, resolved_deps)

def print_help():
    """Prints CLI usage information."""
    print("Usage: python deps.py <file.py> <function_name> [flags]")
    print("\nFlags:")
    print("  --no-libs    Hide library function dependencies.")
    print("  --no-locals  Hide local function dependencies.")
    print("  -r, --reverse Invert the order of local functions.")
    print("  --leaf       Highlight leaf functions (no local calls).")
    print("  -h, --help   Show this help message.")

def main():
    """Main entry point for the script."""
    args = [arg for arg in sys.argv[1:] if not arg.startswith('-')]
    flags = {arg for arg in sys.argv[1:] if arg.startswith('-')}

    if not args or '-h' in flags or '--help' in flags:
        print_help()
        sys.exit(0)

    if len(args) < 2:
        print("Error: Missing file path or function name.")
        print_help()
        sys.exit(1)

    file_path, root_function = args[0], args[1]

    try:
        call_graph, local_defs, import_map = analyze_file(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        sys.exit(1)

    if root_function not in local_defs:
        print(f"Error: Function '{root_function}' not defined in '{file_path}'.")
        sys.exit(1)

    ordered_deps = []
    resolve_dependencies_dfs(root_function, call_graph, ordered_deps)
    
    # Filter out the root function itself and duplicates
    dependencies = list(dict.fromkeys(ordered_deps[1:])) # Keep order and remove duplicates

    local_deps = sorted([d for d in dependencies if d in local_defs], key=dependencies.index)
    
    # Identify leaf functions (functions that don't call other local functions)
    leaf_functions = {
        f for f in local_deps 
        if not any(called in local_defs for called in call_graph.get(f, []))
    } if '--leaf' in flags else set()

    lib_deps = defaultdict(list)
    
    for dep in dependencies:
        if dep not in local_defs:
            parts = dep.split('.')
            base = parts[0]
            if base in import_map:
                module_name = import_map[base].split('.')[0]
                lib_deps[module_name].append(dep)

    print(f"{root_function} dependencies:")

    if '--no-locals' not in flags and local_deps:
        print(" Local functions:")
        
        deps_to_print = reversed(local_deps) if '-r' in flags or '--reverse' in flags else local_deps
        
        # ANSI color codes
        GREEN = '\033[92m'
        ENDC = '\033[0m'

        for dep in deps_to_print:
            if dep in leaf_functions:
                print(f"  - {GREEN}{dep}{ENDC}")
            else:
                print(f"  - {dep}")

    if '--no-libs' not in flags and lib_deps:
        print(" Library functions:")
        for lib, funcs in sorted(lib_deps.items()):
            print(f"  {lib}:")
            for func in sorted(funcs):
                print(f"   - {func}")

if __name__ == "__main__":
    main()