from pathlib import Path
import sys

if len(sys.argv) != 2:
    print('Usage: python combine.py "folder"')
    sys.exit(1)

folder = Path(sys.argv[1]).resolve()

if not folder.is_dir():
    print(f"Folder does not exist: {folder}")
    sys.exit(1)

output = folder / "combined.txt"

print(f"Reading ONLY:")
print(f"  {folder}")
print()

count = 0

with open(output, "w", encoding="utf-8") as out:
    for file in folder.rglob("*"):

        if not file.is_file():
            continue

        if file == output:
            continue

        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            print(f"Skipped: {file}")
            continue

        relative_path = file.relative_to(folder)

        out.write("=" * 70)
        out.write(f"\nFILE: {relative_path}\n")
        out.write("=" * 70)
        out.write("\n\n")
        out.write(text)
        out.write("\n\n")

        count += 1

print(f"Combined {count} files.")
print(f"Created: {output}")