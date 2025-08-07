# ssae_vm_extended.py
# Full SSAE VM Interpreter with extensions: Push, Pull, Store, Clear, Test, Echo Brightest, Trace, Dump, RAM, and NASM Assembly Translator

import sys
import os
from termcolor import cprint

# ---------------------- Parser ----------------------
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

# ---------------------- Memory ----------------------
class Memory:
    def __init__(self):
        self.registers = {f"R{i}": 0 for i in range(16)}
        self.stack = []
        self.ram = {}

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

# ---------------------- Instructions ----------------------
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

# ---------------------- Translator ----------------------
def translate_to_nasm(program):
    lines = [
        "section .data",
        "msg db 'Output:', 0Ah, 0",
        "section .text",
        "global main",
        "main:",
        "    push rbp",
        "    mov rbp, rsp"
    ]
    reg_map = {
        "R1": "r8",
        "R2": "r9",
        "R3": "r10",
        "R4": "r11"
    }
    for instr in program:
        verb = instr["verb"].lower()
        target = instr["target"]
        operands = instr["operands"]
        if ":" in target:
            label = target.strip(":")
            lines.append(f"{label}:")
            continue
        if verb == "load":
            reg = reg_map.get(target, "r8")
            val = operands[0]
            if val.startswith("#"):
                lines.append(f"    mov {reg}, {val[1:]}")
        elif verb == "nudge":
            reg = reg_map.get(target, "r8")
            delta = "1" if operands[0] == "+1" else "-1"
            lines.append(f"    add {reg}, {delta}")
        elif verb == "calc":
            reg = reg_map.get(target, "r8")
            op = operands[0]
            rhs = operands[1]
            rhs_val = rhs[1:] if rhs.startswith("#") else rhs
            if op == "+":
                lines.append(f"    add {reg}, {rhs_val}")
            elif op == "-":
                lines.append(f"    sub {reg}, {rhs_val}")
        elif verb == "echo":
            ch = operands[0]
            if ch.startswith("'"):
                lines.append(f"    mov rax, 0x2000004")
                lines.append(f"    mov rdi, 1")
                lines.append(f"    mov rsi, msg")
                lines.append(f"    mov rdx, 8")
                lines.append(f"    syscall")
        elif verb == "jump":
            label = operands[0]
            condition = operands[1].lower()
            arg1 = operands[2]
            arg2 = operands[3]
            if condition == "ifequal":
                lines.append(f"    cmp {reg_map.get(arg1, arg1)}, {arg2[1:]}")
                lines.append(f"    je {label}")
    lines.append("    mov rsp, rbp")
    lines.append("    pop rbp")
    lines.append("    ret")
    return "\n".join(lines)

# ---------------------- Entry ----------------------
def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  ssae run <file.ssae> [--trace|--dump|--verbose]")
        print("  ssae build <file.ssae> [-o output.asm] [--exe]")
        return

    mode = sys.argv[1]
    filename = sys.argv[2]
    flags = sys.argv[3:]

    with open(filename, "r") as f:
        source = f.read()

    program = parse_ssae(source)

    if mode == "run":
        execute_program(program, flags)
    elif mode == "build":
        asm_out = "output.asm"
        if "-o" in flags:
            idx = flags.index("-o")
            asm_out = flags[idx + 1]
        asm_code = translate_to_nasm(program)
        with open(asm_out, "w") as f:
            f.write(asm_code)
        print(f"Assembly written to {asm_out}")
        if "--exe" in flags:
            bin_out = asm_out.replace(".asm", ".exe")
            os.system(f"nasm -f win64 {asm_out} -o temp.obj")
            os.system(f"clang temp.obj -o {bin_out}")
            os.remove("temp.obj")
            print(f"Executable built: {bin_out}")

if __name__ == "__main__":
    main()
