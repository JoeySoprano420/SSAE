# ssae_vm_extended.py
# Full SSAE VM Interpreter with extensions: Push, Pull, Store, Clear, Test, Echo Brightest, Trace, Dump, RAM, Metadata, Opcode Optimization, NASM Translator, and VSIX IDE hooks

import sys
import os
import json
try:
    from termcolor import cprint
except ImportError:
    def cprint(text, color=None, **kwargs):
        print(text)


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

# ---------------------- Autocompletion Engine ----------------------

ssae_keywords = [
    "Load", "Push", "Pull", "Store", "Clear", "Test", "Calc", "Echo", "Jump", "Nudge",
    "Loop", "Blink", "Zap"
]

ssae_registers = [f"R{i}" for i in range(16)]
ssae_qualifiers = [
    "Fastest", "Tightest", "Hardest", "Loudest", "Softest", "Brightest", "Deepest", "Cleanest"
]
ssae_conditions = [
    "IfEqual", "IfNotEqual", "IfAbove", "IfBelow", "IfZero", "IfNonZero", "IfCarry"
]

def provide_autocompletions(prefix):
    pool = ssae_keywords + ssae_registers + ssae_qualifiers + ssae_conditions
    return [word for word in pool if word.lower().startswith(prefix.lower())]

# ---------------------- CapsuleNet: Peer Execution Module ----------------------
import socket
import threading

CAPSULENET_PORT = 60606
CAPSULENET_HEADER = b"[CAPSULENET_PACKET]"

# Peer Capsule Registry
capsule_registry = {}

def start_capsulenet_server(vm_callback):
    def handler(conn, addr):
        data = conn.recv(65536)
        if data.startswith(CAPSULENET_HEADER):
            capsule_data = data[len(CAPSULENET_HEADER):].decode()
            capsule_id = f"{addr[0]}:{addr[1]}"
            capsule_registry[capsule_id] = capsule_data
            print(f"[🛰️ Capsule received from {capsule_id}]")
            vm_callback(capsule_data, capsule_id)
        conn.close()

    def server():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('', CAPSULENET_PORT))
        s.listen(5)
        print(f"[🌐 CapsuleNet Listening on port {CAPSULENET_PORT}]")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handler, args=(conn, addr)).start()

    threading.Thread(target=server, daemon=True).start()

def send_capsule_to_peer(peer_ip, capsule_text):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((peer_ip, CAPSULENET_PORT))
        s.sendall(CAPSULENET_HEADER + capsule_text.encode())
        print(f"[📡 Capsule sent to {peer_ip}]")
    except Exception as e:
        print(f"[❌ CapsuleNet Error]: {e}")
    finally:
        s.close()

# ssae_vm_extended.py
# Full SSAE VM Interpreter with DGs, SAN, DLMT, SSAE AOT/JIT Capsule Compilation,
# LLVM IR Transpiler, Metadata-Aware Optimizations, and Python-Bridge utilities

import sys
import os
import json
import keyword
try:
    from termcolor import cprint
except ImportError:
    def cprint(text, color=None, **kwargs):
        print(text)


# ---------------------- Capsule Metadata ----------------------
def extract_metadata(source: str) -> dict:
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
def parse_ssae(source: str) -> list:
    lines = source.splitlines()
    program = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("@meta"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid syntax: {line}")
        instruction = {
            "verb": parts[0],
            "target": parts[1],
            "qualifier": parts[2],
            "operands": parts[3:] if len(parts) > 3 else [],
        }
        program.append(instruction)
    return program

# ---------------------- Memory ----------------------
class Memory:
    def __init__(self):
        self.registers = {f"R{i}": 0 for i in range(16)}
        self.stack = []
        self.ram = {}
        self.mutations = []  # For DG tracing / debug

    def reg(self, name):
        return name.upper()

    def get(self, reg):
        return self.registers[reg]

    def set(self, reg, val):
        self.mutations.append((reg, val))
        self.registers[reg] = val

    def resolve(self, val):
        if isinstance(val, (int, float)):
            return val
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
            "registers": dict(self.registers),
            "stack": list(self.stack),
            "ram": dict(self.ram),
            "mutations": list(self.mutations),
        }

# ---------------------- Optimizer (DLMT-ish) ----------------------
def optimize_program(program: list) -> list:
    optimized = []
    last = None
    for instr in program:
        if (
            last
            and last["verb"].lower() == "nudge"
            and instr["verb"].lower() == "nudge"
            and last["target"] == instr["target"]
        ):
            # Combine consecutive nudges on same register
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

# ---------------------- Translator (NASM) ----------------------
def translate_to_nasm(program: list) -> str:
    lines = [
        "section .data",
        "msg db 'Output:', 0Ah, 0",
        "section .text",
        "global main",
        "main:",
        "    push rbp",
        "    mov rbp, rsp",
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
            if isinstance(val, str) and val.startswith("#"):
                lines.append(f"    mov {reg}, {val[1:]}")
        elif verb == "nudge":
            reg = reg_map.get(target, "r8")
            delta = operands[0]
            lines.append(f"    add {reg}, {delta}")
        elif verb == "calc":
            reg = reg_map.get(target, "r8")
            op = operands[0]
            rhs_val = operands[1]
            if isinstance(rhs_val, str) and rhs_val.startswith("#"):
                rhs_val = rhs_val[1:]
            if op == "+":
                lines.append(f"    add {reg}, {rhs_val}")
            elif op == "-":
                lines.append(f"    sub {reg}, {rhs_val}")
        elif verb == "echo":
            lines.extend(
                [
                    "    mov rax, 0x2000004",
                    "    mov rdi, 1",
                    "    mov rsi, msg",
                    "    mov rdx, 8",
                    "    syscall",
                ]
            )
        elif verb == "jump":
            label = operands[0]
            condition = operands[1].lower() if len(operands) > 1 else None
            arg1 = operands[2] if len(operands) > 2 else None
            arg2 = operands[3] if len(operands) > 3 else None
            if isinstance(arg1, str):
                arg1 = reg_map.get(arg1, arg1)
            if isinstance(arg2, str) and arg2.startswith("#"):
                arg2 = arg2[1:]
            if condition == "ifequal" and arg1 is not None and arg2 is not None:
                lines.append(f"    cmp {arg1}, {arg2}")
                lines.append(f"    je {label}")
    lines.extend(["    mov rsp, rbp", "    pop rbp", "    ret"])
    return "\n".join(lines)

# ---------------------- Execute ----------------------
def execute_program(program: list, flags: list):
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
            if op == "+":
                mem.set(reg, mem.get(reg) + rhs)
            elif op == "-":
                mem.set(reg, mem.get(reg) - rhs)
        elif verb == "nudge":
            reg = mem.reg(target)
            delta = int(operands[0])
            mem.set(reg, mem.get(reg) + delta)
        elif verb == "zap":
            # ZAP: zero out a register or RAM label
            if target.startswith("R"):
                mem.set(mem.reg(target), 0)
            elif target.startswith("@"):
                mem.store(target[1:], 0)
            else:
                raise ValueError(f"ZAP: Unknown target {target}")
        elif verb == "zap":
            # ZAP: zero out a register or RAM label
            if target.startswith("R"):
                mem.set(mem.reg(target), 0)
            elif target.startswith("@"):
                mem.store(target[1:], 0)
            else:
                raise ValueError(f"ZAP: Unknown target {target}")
        elif verb == "zap":
            # ZAP: zero out a register or RAM label
            if target.startswith("R"):
                mem.set(mem.reg(target), 0)
            elif target.startswith("@"):
                mem.store(target[1:], 0)
            else:
                raise ValueError(f"ZAP: Unknown target {target}")
        elif verb == "zap":
            # ZAP: zero out a register or RAM label
            if target.startswith("R"):
                mem.set(mem.reg(target), 0)
            elif target.startswith("@"):
                mem.store(target[1:], 0)
            else:
                raise ValueError(f"ZAP: Unknown target {target}")
        elif verb == "echo":
            ch = mem.resolve(operands[0])
            if qualifier == "brightest":
                cprint(str(ch), "cyan", attrs=["bold", "underline"], end="")
            else:
                print(str(ch), end="", flush=True)
        elif verb == "jump":
            label = operands[0]
            condition = operands[1].lower() if len(operands) > 1 else None
            arg1 = mem.resolve(operands[2]) if len(operands) > 2 else None
            arg2 = mem.resolve(operands[3]) if len(operands) > 3 else None
            jump = False
            if condition == "ifequal" and arg1 == arg2:
                jump = True
            elif condition == "ifabove" and (arg1 is not None and arg2 is not None) and arg1 > arg2:
                jump = True
            if jump:
                ip = labels[label]
                continue
        ip += 1

    if dump:
        print("\n--MEMORY DUMP--")
        print(json.dumps(mem.dump(), indent=2))

# ---------------------- LLVM IR Transpiler ----------------------
def transpile_to_llvm(program: list) -> str:
    ir = ["define i32 @main() {"]
    reg_map = {f"R{i}": f"%r{i}" for i in range(4)}
    for instr in program:
        verb, target, operands = instr["verb"].lower(), instr["target"], instr["operands"]
        if ":" in target:
            continue
        if verb == "load":
            ir.append(f"  {reg_map.get(target, '%r1')} = add i32 0, {operands[0][1:]}")
        elif verb == "nudge":
            ir.append(f"  {reg_map.get(target, '%r1')} = add i32 {reg_map.get(target, '%r1')}, {operands[0]}")
        elif verb == "calc":
            op = operands[0]
            if op == "+":
                ir.append(
                    f"  {reg_map.get(target, '%r1')} = add i32 {reg_map.get(target, '%r1')}, {operands[1][1:]}"
                )
            elif op == "-":
                ir.append(
                    f"  {reg_map.get(target, '%r1')} = sub i32 {reg_map.get(target, '%r1')}, {operands[1][1:]}"
                )
    ir.append("  ret i32 0\n}")
    return "\n".join(ir)

# ---------------------- Python Keyword Integration & Bridge ----------------------
PY_KEYWORDS = keyword.kwlist  # e.g., ['False','None','True','and','as','assert', ...]

ssae_keywords = [
    "Load",
    "Push",
    "Pull",
    "Store",
    "Clear",
    "Test",
    "Calc",
    "Echo",
    "Jump",
    "Nudge",
    "Loop",
    "Blink",
    "Zap",
]

ssae_registers = [f"R{i}" for i in range(16)]
ssae_qualifiers = [
    "Fastest",
    "Tightest",
    "Hardest",
    "Loudest",
    "Softest",
    "Brightest",
    "Deepest",
    "Cleanest",
]
ssae_conditions = [
    "IfEqual",
    "IfNotEqual",
    "IfAbove",
    "IfBelow",
    "IfZero",
    "IfNonZero",
    "IfCarry",
]

def provide_autocompletions_with_python(prefix: str) -> list:
    pool = ssae_keywords + ssae_registers + ssae_qualifiers + ssae_conditions + PY_KEYWORDS
    return [w for w in pool if w.lower().startswith(prefix.lower())]

# Inline-Python capture: lines beginning with `py:` are collected and can be executed or emitted.

def parse_ssae_with_python(source: str) -> tuple:
    program = []
    py_snippets = []
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@meta") or line.startswith("//"):
            continue
        if line.startswith("py:"):
            # Store raw Python code after the prefix (preserve original spacing after 'py:')
            py_snippets.append(raw.split("py:", 1)[1])
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(
                f"Invalid SSAE syntax (need verb target qualifier): {line}"
            )
        program.append(
            {
                "verb": parts[0],
                "target": parts[1],
                "qualifier": parts[2],
                "operands": parts[3:] if len(parts) > 3 else [],
            }
        )
    return program, py_snippets

# Execute collected Python snippets in a simple (non-sandboxed) scope.

def execute_python_snippets(snippets: list, env=None, echo: bool = True):
    scope = {} if env is None else env
    results = []
    for i, code in enumerate(snippets, 1):
        try:
            try:
                val = eval(code, {}, scope)
                results.append((i, "eval", val))
                if echo:
                    print(f"[py#{i} eval] => {val}")
            except SyntaxError:
                exec(code, {}, scope)
                results.append((i, "exec", None))
                if echo:
                    print(f"[py#{i} exec] ok")
        except Exception as e:
            results.append((i, "error", str(e)))
            if echo:
                print(f"[py#{i} error] {e}")
    return scope, results

# Bridge: emit a minimal Python module from SSAE registers/RAM for interop/testing.

def emit_python_bridge(mem_dump: dict) -> str:
    regs = mem_dump.get("registers", {})
    ram = mem_dump.get("ram", {})
    lines = [
        "# Auto-generated from SSAE VM",
        "registers = {}",
        "ram = {}",
    ]
    for k, v in regs.items():
        lines.append(f"registers['{k}'] = {repr(v)}")
    for k, v in ram.items():
        lines.append(f"ram['{k}'] = {repr(v)}")
    lines.append(
        """
# Example helper

def read(name):
    return registers.get(name, 0)

def read_mem(label):
    return ram.get(label, 0)
"""
    )
    return "\n".join(lines)

# ---------------------- Entry (classic) ----------------------
def main():
    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "  ssae run <file.ssae> [--trace|--dump|--verbose]\n"
            "  ssae build <file.ssae> [-o output.asm] [--exe] [--llvm]"
        )
        return

    mode = sys.argv[1]
    filename = sys.argv[2]
    flags = sys.argv[3:]

    with open(filename, "r") as f:
        source = f.read()

    metadata = extract_metadata(source)
    program = optimize_program(parse_ssae(source))

    if mode == "run":
        print(f"[Capsule: {metadata.get('name', 'Unnamed')}] {metadata.get('description', '')}")
        execute_program(program, flags)
    elif mode == "build":
        if "--llvm" in flags:
            llvm_code = transpile_to_llvm(program)
            out = filename.replace(".ssae", ".ll")
            with open(out, "w") as f:
                f.write(llvm_code)
            print(f"LLVM IR written to {out}")
            return

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
    elif mode == "selftest":
        run_self_tests()
    else:
        print(f"Unknown mode: {mode}")

# ---------------------- Entry (python-extended) ----------------------
def main_with_python():
    if len(sys.argv) < 3:
        print(
            """Usage:
  ssae run <file.ssae> [--trace|--dump|--verbose|--py|--emit-py bridge.py]
  ssae build <file.ssae> [-o output.asm] [--exe] [--llvm]
  ssae selftest
"""
        )
        return

    mode = sys.argv[1]
    filename = sys.argv[2]
    flags = sys.argv[3:]

    with open(filename, "r") as f:
        source = f.read()

    metadata = extract_metadata(source)
    program, py_snips = parse_ssae_with_python(source)
    program = optimize_program(program)

    if mode == "run":
        print(f"[Capsule: {metadata.get('name', 'Unnamed')}] {metadata.get('description', '')}")
        execute_program(program, flags)
        if "--py" in flags and py_snips:
            print("[Python Bridge] Executing inline python snippets...")
            execute_python_snippets(py_snips)
        if "--emit-py" in flags:
            try:
                out_idx = flags.index("--emit-py") + 1
                out_path = flags[out_idx]
            except Exception:
                out_path = filename.replace(".ssae", "_bridge.py")
            # Export current (empty) memory bridge; can be replaced with real post-run mem if desired
            bridge = emit_python_bridge(Memory().dump())
            with open(out_path, "w") as f:
                f.write(bridge)
            print(f"[Python Bridge] Written to {out_path}")
    elif mode == "build":
        if "--llvm" in flags:
            llvm_code = transpile_to_llvm(program)
            out = filename.replace(".ssae", ".ll")
            with open(out, "w") as f:
                f.write(llvm_code)
            print(f"LLVM IR written to {out}")
            return
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
    elif mode == "selftest":
        run_self_tests()
    else:
        print(f"Unknown mode: {mode}")

# ---------------------- Self Tests ----------------------
def run_self_tests():
    print("[SELFTEST] Starting...")

    # Test 1: Basic SSAE parse/execute
    src1 = (
        "@meta name: SelfTest1\n"
        "Load R1 Fastest #2\n"
        "Nudge R1 Fastest +3\n"
        "Calc R1 Fastest + #5\n"
        "Echo Char Loudest 'X'\n"
    )
    prog1 = optimize_program(parse_ssae(src1))
    execute_program(prog1, ["--trace"])  # Should print an X and show trace

    # Test 2: NASM translation minimal
    asm = translate_to_nasm(prog1)
    assert "global main" in asm and "add r8" in asm, "NASM translation missing expected instructions"
    print("[SELFTEST] NASM translation ok")

    # Test 3: LLVM IR transpile
    ll = transpile_to_llvm(prog1)
    assert "define i32 @main()" in ll, "LLVM IR header missing"
    print("[SELFTEST] LLVM IR generation ok")

    # Test 4: Python inline parse & exec
    src2 = (
        "@meta name: PyTest\n"
        "py: 1+1\n"
        "Load R1 Fastest #1\n"
    )
    p2, py2 = parse_ssae_with_python(src2)
    _, results = execute_python_snippets(py2, echo=False)
    assert results and results[0][1] in ("eval", "exec"), "Inline Python did not execute"
    print("[SELFTEST] Python inline execution ok")

    # Test 5: Bridge emission string validity
    mem = Memory()
    mem.set("R1", 42)
    mem.store("Answer", 42)
    bridge_text = emit_python_bridge(mem.dump())
    compile(bridge_text, "<bridge>", "exec")  # Should not raise
    print("[SELFTEST] Python bridge emission ok")

    print("[SELFTEST] All tests passed ✅")

if __name__ == "__main__":
    main()
elif mode == "ide":
    print("[🔧 IDE Mode Activated: Streaming AOT Compilation]")
    stream_compile_to_exe(filename, flags)

import time

def stream_compile_to_exe(filename, flags):
    last_frame = ""
    asm_out = filename.replace(".ssae", ".stream.asm")
    bin_out = asm_out.replace(".asm", ".exe")
    compiled_frames = []

    while True:
        try:
            with open(filename, "r") as f:
                source = f.read()
            if source != last_frame:
                last_frame = source
                metadata = extract_metadata(source)
                program = optimize_program(parse_ssae(source))
                asm_code = translate_to_nasm(program)

                compiled_frames.append(asm_code)
                with open(asm_out, "w") as f:
                    f.write("\n\n".join(compiled_frames))
                print(f"[🧩 Frame compiled: {len(program)} instructions]")

                if "--exe" in flags:
                    os.system(f"nasm -f win64 {asm_out} -o temp.obj")
                    os.system(f"clang temp.obj -o {bin_out}")
                    os.remove("temp.obj")
                    print(f"[🚀 Executable updated: {bin_out}]")

            time.sleep(1)  # Polling interval
        except KeyboardInterrupt:
            print("[🛑 IDE Mode Terminated]")
            break


# ---------------------- VSIX Hook Stub ----------------------
# This section is reserved for a future VSCode extension integration
# It will include syntax, hover, and metadata doc via LSP (language server protocol)
# Placeholder stub for compatibility:
# def provide_hover_info(token): return f"Docs for {token}: built-in SSAE opcode."
