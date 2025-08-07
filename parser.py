# parser.py
def parse_ssae(source):
    lines = source.splitlines()
    program = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("//"): continue

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid syntax: {line}")

        instruction = {
            "verb": parts[0],
            "target": parts[1],
            "qualifier": parts[2],
            "operands": parts[3:] if len(parts) > 3 else []
        }
        program.append(instruction)

    return program
