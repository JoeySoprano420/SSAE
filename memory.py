# memory.py
class Memory:
    def __init__(self):
        self.registers = {f"R{i}": 0 for i in range(16)}
        self.stack = []

    def reg(self, name):
        return name.upper()

    def get(self, reg):
        return self.registers[reg]

    def set(self, reg, val):
        self.registers[reg] = val

    def resolve(self, val):
        if val.startswith("R"):
            return self.get(val)
        elif val.startswith("'") and val.endswith("'"):
            return val[1:-1]
        elif val.startswith("#"):
            return int(val[1:])
        else:
            raise ValueError(f"Unknown operand: {val}")
