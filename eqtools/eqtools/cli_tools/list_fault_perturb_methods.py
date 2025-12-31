import sys
import os
import argparse
import inspect

# -----------------------------------------------------------------------------
# 1. Path Setup
# Adjust this to point to your project root so 'csiExtend' can be imported
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Assuming tools/ is inside root
sys.path.append(project_root)

def load_registry():
    """
    Imports fault classes to trigger decorators and populate the registry.
    """
    try:
        # Import the Registry
        from ..csiExtend.bayesian_perturbation_base import PerturbationRegistry
        
        # Import ALL fault classes you want to inspect.
        # This execution triggers the @track_mesh_update decorators.
        from ..csiExtend.BayesianAdaptiveTriangularPatches import BayesianAdaptiveTriangularPatches
        from ..csiExtend.AdaptiveLayeredDipTriangularPatches import AdaptiveLayeredDipTriangularPatches
        # import ..csiExtend.YourFutureFaultClass  <-- Add new classes here
        
        return PerturbationRegistry.get_help()
    except ImportError as e:
        print(f"Error: Failed to import project modules.\n{e}")
        print(f"PYTHONPATH: {sys.path}")
        sys.exit(1)

def format_text(registry):
    """Generates a human-readable text table for the terminal."""
    lines = []
    lines.append("="*100)
    lines.append(f"{'AVAILABLE PERTURBATION METHODS':^100}")
    lines.append("="*100)
    
    for cls_name, methods in registry.items():
        if not methods: continue
        
        lines.append(f"\n[ CLASS: {cls_name} ]")
        lines.append("-" * 100)
        lines.append(f"{'Method Name':<45} | {'Description'}")
        lines.append("-" * 45 + "-+-" + "-" * 50)
        
        for name, meta in methods.items():
            desc = meta.get('description', 'N/A')
            params = meta.get('params')
            
            lines.append(f"{name:<45} | {desc}")
            if params:
                lines.append(f"{' ':<45} | Params: {params}")
        lines.append("")
    
    return "\n".join(lines)

def format_markdown(registry):
    """
    Generates a categorized Markdown document.
    """
    lines = []
    lines.append("# Available Perturbation Methods\n")
    lines.append("> Auto-generated from source code. categorized for easier navigation.\n")

    # Define Categories and Keywords
    categories = {
        "🧬 Dutta Methods (Shape)": ["Dutta"],
        "📉 Dip Angle Adjustments": ["dip", "Dip"],
        "🔄 Rotation & Rigid Motion": ["Rotate", "Rotation", "rotation"],
        "⚓ Endpoints & Midpoints": ["Endpoints", "Midpoint"],
        "🧩 Complex/Composite": ["RotateTrans", "multiLayer"],
        "📏 Translation & Shifts": ["Translation", "translation", "Trans", "FixedDir", "fixed_direction"]
    }
    
    # Helper to clean params string
    def clean_params(p_dict):
        if not p_dict: return "N/A"
        # Convert dict to string and escape pipes for markdown table
        return str(p_dict).replace("|", "\|").replace("{", "").replace("}", "").replace("'", "")

    for cls_name, methods in registry.items():
        if not methods: continue
        
        lines.append(f"## Class: `{cls_name}`\n")
        
        # Bucket for categorized methods
        buckets = {k: [] for k in categories.keys()}
        buckets["⚙️ Other / General"] = [] # Fallback category

        for name, meta in methods.items():
            matched = False
            for cat_name, keywords in categories.items():
                if any(k in name for k in keywords):
                    buckets[cat_name].append((name, meta))
                    matched = True
                    break # Assign to first matching category
            if not matched:
                buckets["⚙️ Other / General"].append((name, meta))

        # Print Categories
        for cat_name, items in buckets.items():
            if not items: continue
            
            lines.append(f"### {cat_name}")
            lines.append("| Method Name | Description | Parameters |")
            lines.append("|---|---|---|")
            
            for name, meta in items:
                desc = meta.get('description', 'N/A')
                params = clean_params(meta.get('params'))
                lines.append(f"| `{name}` | {desc} | `{params}` |")
            lines.append("\n")
            
    return "\n".join(lines)

def format_yaml_comment(registry):
    """Generates text suitable for copy-pasting into config.yml comments."""
    lines = []
    lines.append("# --- Reference: Available Methods ---")
    
    for cls_name, methods in registry.items():
        lines.append(f"# Class: {cls_name}")
        for name, meta in methods.items():
            desc = meta.get('description', '')
            lines.append(f"#  - method: {name}  # {desc}")
        lines.append("#")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="List available Bayesian perturbation methods.")
    
    parser.add_argument(
        '-f', '--format', 
        choices=['text', 'markdown', 'yaml'], 
        default='text',
        help="Output format (default: text)"
    )
    
    parser.add_argument(
        '-o', '--out', 
        type=str,
        help="Output file path. If not specified, prints to console."
    )

    args = parser.parse_args()
    
    # 1. Load Data
    registry = load_registry()
    
    # 2. Format Data
    if args.format == 'markdown':
        output = format_markdown(registry)
    elif args.format == 'yaml':
        output = format_yaml_comment(registry)
    else:
        output = format_text(registry)
    
    # 3. Output
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Successfully saved {args.format} list to: {args.out}")
    else:
        print(output)

if __name__ == "__main__":
    main()