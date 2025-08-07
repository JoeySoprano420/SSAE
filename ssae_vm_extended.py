# ssae_vm_extended.py
# Full SSAE VM Interpreter with extensions: Push, Pull, Store, Clear, Test, Echo Brightest, Trace, Dump, RAM, Metadata, Opcode Optimization, NASM Translator, and VSIX IDE hooks

import sys
import os
import json
from termcolor import cprint

# ---------------------- Capsule Metadata ----------------------
def extract_metadata(source):
    metadata = {}
    for line in source.splitlines():
        if line.startswith("@meta"):
            keyval = line[6:].split(":", 1)
            if len(keyval) == 2:
                key, val = keyval[0].strip(), keyval[1].strip()
                metadata[key] = val
        if line.strip() == "":
            break
    return metadata

# ---------------------- Parser ----------------------
def parse_ssae(source):
    lines = source.splitlines()
    program = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("@meta"): continue
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

# ---------------------- Optimizer ----------------------
def optimize_program(program):
    optimized = []
    last = None
    for instr in program:
        if last and last["verb"] == "nudge" and instr["verb"] == "nudge" and last["target"] == instr["target"]:
            delta1 = int(last["operands"][0])
            delta2 = int(instr["operands"][0])
            last["operands"] = [str(delta1 + delta2)]
        else:
            if last:
                optimized.append(last)
            last = instr
    if last:
        optimized.append(last)
    return optimized

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
            delta = int(operands[0])
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
    reg_map = {"R1": "r8", "R2": "r9", "R3": "r10", "R4": "r11"}
    for instr in program:
        verb = instr["verb"].lower()
        target = instr["target"]
        operands = instr["operands"]
        if ":" in target:
            lines.append(f"{target.strip(':')}:")
            continue
        if verb == "load":
            reg = reg_map.get(target, "r8")
            val = operands[0]
            if val.startswith("#"):
                lines.append(f"    mov {reg}, {val[1:]}")
        elif verb == "nudge":
            reg = reg_map.get(target, "r8")
            delta = operands[0]
            lines.append(f"    add {reg}, {delta}")
        elif verb == "calc":
            reg = reg_map.get(target, "r8")
            op = operands[0]
            rhs_val = operands[1][1:] if operands[1].startswith("#") else operands[1]
            if op == "+":
                lines.append(f"    add {reg}, {rhs_val}")
            elif op == "-":
                lines.append(f"    sub {reg}, {rhs_val}")
        elif verb == "echo":
            lines.extend([
                f"    mov rax, 0x2000004",
                f"    mov rdi, 1",
                f"    mov rsi, msg",
                f"    mov rdx, 8",
                f"    syscall"
            ])
        elif verb == "jump":
            label = operands[0]
            condition = operands[1].lower()
            arg1 = reg_map.get(operands[2], operands[2])
            arg2 = operands[3][1:] if operands[3].startswith("#") else operands[3]
            if condition == "ifequal":
                lines.append(f"    cmp {arg1}, {arg2}")
                lines.append(f"    je {label}")
    lines.extend(["    mov rsp, rbp", "    pop rbp", "    ret"])
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
    metadata = extract_metadata(source)
    program = parse_ssae(source)
    program = optimize_program(program)
    if mode == "run":
        print(f"[Capsule: {metadata.get('name', 'Unnamed')}] {metadata.get('description', '')}")
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

# ---------------------- VSIX Hook Stub ----------------------
# This section is reserved for a future VSCode extension integration
# It will include syntax, hover, and metadata doc via LSP (language server protocol)
# Placeholder stub for compatibility:
# def provide_hover_info(token): return f"Docs for {token}: built-in SSAE opcode."
