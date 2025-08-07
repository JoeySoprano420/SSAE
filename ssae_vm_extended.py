# ssae_vm_extended.py
# Full SSAE VM Interpreter with extensions: Push, Pull, Store, Clear, Test, Echo Brightest, Trace, Dump, and RAM

from parser import parse_ssae
from instructions import execute_program
import sys

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        print("Usage: ssae run <filename.ssae> [--trace|--dump|--verbose]")
        return

    filename = sys.argv[2]
    flags = sys.argv[3:]

    with open(filename, "r") as f:
        source = f.read()

    program = parse_ssae(source)
    execute_program(program, flags)

if __name__ == "__main__":
    main()

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

# memory.py
class Memory:
    def __init__(self):
        self.registers = {f"R{i}": 0 for i in range(16)}
        self.stack = []
        self.ram = {}  # for @Label support

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
        elif val.startswith("@"):
            return self.ram.get(val[1:], 0)
        else:
            raise ValueError(f"Unknown operand: {val}")

    def store(self, label, val):
        self.ram[label] = val

    def dump(self):
        return {
            "registers": self.registers,
            "stack": self.stack,
            "ram": self.ram
        }

# instructions.py
from memory import Memory
from termcolor import cprint

def execute_program(program, flags):
    mem = Memory()
    ip = 0
    trace = "--trace" in flags
    dump = "--dump" in flags
    verbose = "--verbose" in flags

    labels = {line["target"].strip(":"): i for i, line in enumerate(program) if ":" in line["target"]}

    while ip < len(program):
        line = program[ip]
        verb = line["verb"].lower()
        target = line["target"]
        qualifier = line["qualifier"].lower()
        operands = line["operands"]

        if ":" in target:
            ip += 1
            continue

        if trace:
            print(f"[{ip}] {verb.upper()} {target} {qualifier.upper()} {operands}")

        if verb == "load":
            reg = mem.reg(target)
            val = mem.resolve(operands[0])
            mem.set(reg, val)

        elif verb == "store":
            val = mem.resolve(target)
            mem.store(operands[0][1:], val)

        elif verb == "push":
            val = mem.resolve(target)
            mem.stack.append(val)

        elif verb == "pull":
            if mem.stack:
                mem.set(mem.reg(target), mem.stack.pop())

        elif verb == "clear":
            mem.set(mem.reg(target), 0)

        elif verb == "test":
            val1 = mem.resolve(target)
            val2 = mem.resolve(operands[0])
            if verbose:
                print(f"TEST {val1} == {val2} => {val1 == val2}")

        elif verb == "calc":
            reg = mem.reg(target)
            op = operands[0]
            rhs = mem.resolve(operands[1])
            if op == '+': mem.set(reg, mem.get(reg) + rhs)
            elif op == '-': mem.set(reg, mem.get(reg) - rhs)

        elif verb == "nudge":
            reg = mem.reg(target)
            delta = -1 if operands[0] == "-1" else 1
            mem.set(reg, mem.get(reg) + delta)

        elif verb == "echo":
            ch = mem.resolve(operands[0])
            if qualifier == "brightest":
                cprint(str(ch), "cyan", end='')
            else:
                print(str(ch), end='', flush=True)

        elif verb == "jump":
            label = operands[0]
            condition = operands[1].lower() if len(operands) > 1 else None
            arg1 = mem.resolve(operands[2]) if len(operands) > 2 else None
            arg2 = mem.resolve(operands[3]) if len(operands) > 3 else None

            jump = False
            if condition == "ifequal" and arg1 == arg2:
                jump = True
            elif condition == "ifabove" and arg1 > arg2:
                jump = True

            if jump:
                ip = labels[label]
                continue

        ip += 1

    if dump:
        import json
        print("\n--MEMORY DUMP--")
        print(json.dumps(mem.dump(), indent=2))
