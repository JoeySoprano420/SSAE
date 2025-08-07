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

    def push(self, val):
        if isinstance(val, str) and val.startswith("R"):
            val = self.get(val)
        self.stack.append(val)

    def pop(self):
        if not self.stack:
            raise IndexError("Stack underflow")
        return self.stack.pop()

    def clear(self, reg):
        if reg.startswith("R"):
            self.set(reg, 0)
        else:
            raise ValueError(f"Unknown register: {reg}")

    def store(self, reg, label):
        if label.startswith("@"):
            self.registers[label[1:]] = self.get(reg)
        else:
            raise ValueError(f"Invalid label: {label}")

    def load(self, reg, label):
        if label.startswith("@"):
            self.set(reg, self.registers[label[1:]])
        else:
            raise ValueError(f"Invalid label: {label}")

    def pull(self, reg):
        if not self.stack:
            raise IndexError("Stack underflow")
        self.set(reg, self.stack.pop())

    def push_stack(self, reg):
        if reg.startswith("R"):
            self.push(self.get(reg))
        else:
            raise ValueError(f"Unknown register: {reg}")

    def pull_stack(self, reg):
        if not self.stack:
            raise IndexError("Stack underflow")
        if reg.startswith("R"):
            self.set(reg, self.stack.pop())
        else:
            raise ValueError(f"Unknown register: {reg}")

    def __repr__(self):
        return f"Memory({self.registers}, Stack: {self.stack})"

    def __str__(self):
        return f"Memory Registers: {self.registers}, Stack: {self.stack}"

