#!/usr/bin/env python3
"""Convert Jupyter notebooks to Python scripts.

This utility helps convert Jupyter notebooks into standalone Python scripts
for batch processing, removing interactive elements and cell magic commands.

Usage:
    python scripts/convert_notebook_to_script.py notebooks/topic-modeling/Topic_Modeling.ipynb \\
        --output scripts/my_script.py --clean

    # Convert all notebooks in a directory
    python scripts/convert_notebook_to_script.py notebooks/ \\
        --output-dir scripts/ --clean --recursive
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def extract_code_cells(notebook_path: Path) -> List[str]:
    """Extract code cells from a Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code)

    return code_cells


def clean_code(code: str, remove_magic: bool = True, remove_colab: bool = True) -> str:
    """Clean code by removing magic commands, Colab-specific code, etc."""
    lines = code.split('\n')
    cleaned_lines = []

    skip_block = False

    for line in lines:
        # Skip empty lines at the start
        if not cleaned_lines and not line.strip():
            continue

        # Remove magic commands
        if remove_magic:
            if line.strip().startswith('%') or line.strip().startswith('!'):
                # Keep important installations but comment them out
                if 'pip install' in line or 'python -m' in line:
                    cleaned_lines.append('# ' + line)
                continue

        # Remove Colab-specific code
        if remove_colab:
            if 'google.colab' in line:
                skip_block = True
                continue
            if skip_block and (line.strip().startswith('except') or 'except' in line):
                skip_block = False
                continue
            if skip_block:
                continue

        # Remove display commands
        if any(cmd in line for cmd in ['display(', 'pyLDAvis.display(', 'get_ipython()']):
            continue

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def convert_notebook_to_script(
    notebook_path: Path,
    output_path: Optional[Path] = None,
    clean: bool = True,
    add_header: bool = True,
    add_main: bool = True
) -> str:
    """Convert a Jupyter notebook to a Python script.

    Args:
        notebook_path: Path to the notebook file
        output_path: Optional output path for the script
        clean: Whether to clean the code (remove magic commands, etc.)
        add_header: Whether to add a docstring header
        add_main: Whether to wrap code in if __name__ == '__main__'

    Returns:
        The generated Python script as a string
    """
    print(f"Converting {notebook_path}...")

    # Extract code cells
    code_cells = extract_code_cells(notebook_path)

    if not code_cells:
        print(f"Warning: No code cells found in {notebook_path}")
        return ""

    # Process each cell
    processed_cells = []
    for cell_code in code_cells:
        if clean:
            cell_code = clean_code(cell_code)

        if cell_code.strip():
            processed_cells.append(cell_code)

    # Combine cells
    script = '\n\n'.join(processed_cells)

    # Add header
    if add_header:
        header = f'''"""
Script converted from Jupyter notebook: {notebook_path.name}

This script was automatically converted from a Jupyter notebook.
You may need to modify it for your specific use case.

Original notebook: {notebook_path}
"""

'''
        script = header + script

    # Add main block
    if add_main and 'if __name__' not in script:
        # Find import section
        lines = script.split('\n')
        import_end = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('import') and not line.strip().startswith('from'):
                import_end = i
                break

        if import_end > 0:
            imports = '\n'.join(lines[:import_end])
            main_code = '\n'.join(lines[import_end:])
            script = f"{imports}\n\ndef main():\n"
            # Indent main code
            main_lines = main_code.split('\n')
            indented_lines = ['    ' + line if line.strip() else '' for line in main_lines]
            script += '\n'.join(indented_lines)
            script += "\n\nif __name__ == '__main__':\n    main()\n"

    # Save to file if output path is provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script)
        print(f"Script saved to {output_path}")

    return script


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to Python scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', type=Path, help='Input notebook file or directory')
    parser.add_argument('--output', '-o', type=Path, help='Output Python script file')
    parser.add_argument('--output-dir', type=Path, help='Output directory (for batch conversion)')
    parser.add_argument('--clean', action='store_true', help='Clean code (remove magic commands, Colab code)')
    parser.add_argument('--no-header', action='store_true', help='Do not add docstring header')
    parser.add_argument('--no-main', action='store_true', help='Do not wrap in main() function')
    parser.add_argument('--recursive', '-r', action='store_true', help='Process notebooks recursively')

    args = parser.parse_args(argv)

    input_path = args.input

    if not input_path.exists():
        print(f"Error: {input_path} does not exist")
        sys.exit(1)

    # Single file conversion
    if input_path.is_file():
        if not input_path.suffix == '.ipynb':
            print(f"Error: {input_path} is not a Jupyter notebook (.ipynb)")
            sys.exit(1)

        if args.output:
            output_path = args.output
        elif args.output_dir:
            output_path = args.output_dir / input_path.with_suffix('.py').name
        else:
            output_path = input_path.with_suffix('.py')

        convert_notebook_to_script(
            input_path,
            output_path,
            clean=args.clean,
            add_header=not args.no_header,
            add_main=not args.no_main
        )

    # Directory conversion
    elif input_path.is_dir():
        if not args.output_dir:
            print("Error: --output-dir required for directory conversion")
            sys.exit(1)

        pattern = '**/*.ipynb' if args.recursive else '*.ipynb'
        notebooks = list(input_path.glob(pattern))

        if not notebooks:
            print(f"No notebooks found in {input_path}")
            sys.exit(0)

        print(f"Found {len(notebooks)} notebooks")

        for notebook in notebooks:
            # Preserve directory structure
            rel_path = notebook.relative_to(input_path)
            output_path = args.output_dir / rel_path.with_suffix('.py')

            try:
                convert_notebook_to_script(
                    notebook,
                    output_path,
                    clean=args.clean,
                    add_header=not args.no_header,
                    add_main=not args.no_main
                )
            except Exception as e:
                print(f"Error converting {notebook}: {e}")

        print(f"\nConversion complete! Scripts saved to {args.output_dir}")

    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
