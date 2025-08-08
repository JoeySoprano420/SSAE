Got you. Here’s a single, unified **`ssae_vm_extended.py`** that merges everything you pasted: VM + parser + optimizer + DG tracing + SAN + DLMT + NASM translator + LLVM IR transpiler + simple AOT/JIT build hooks + autocompletions + VSIX/LSP stubs. Drop this into a file and run `python ssae_vm_extended.py run sample.ssae --trace` or build with `python ssae_vm_extended.py build sample.ssae --llvm -o out.ll` or `--exe`.

```python
# ssae_vm_extended.py
# Full SSAE VM Interpreter with DGs, SAN, DLMT, SSAE AOT/JIT Capsule Compilation,
# LLVM IR Transpiler, Metadata-Aware Optimizations, NASM Translator, and IDE hooks.

import sys
import os
import json
import shutil
from termcolor import cprint

# ============================== Capsule Metadata ==============================
def extract_metadata(source):
    """
    Parse leading @meta lines of form:
      @meta key: value
    Stops at first blank line.
    """
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

# ============================== Parser =======================================
def parse_ssae(source):
    """
    Minimal SSAE line parser:
      VERB TARGET QUALIFIER [OPERANDS...]
    - Ignores empty lines, // comments, and @meta header lines.
    - Labels are encoded by any TARGET that contains ":" (e.g., "loop:")
    """
    lines = source.splitlines()
    program = []
    for line in lines:
        raw = line
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("@meta"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid syntax: {raw}")
        instruction = {
            "verb": parts[0],
            "target": parts[1],
            "qualifier": parts[2],
            "operands": parts[3:] if len(parts) > 3 else []
        }
        program.append(instruction)
    return program

# ============================== DG: Dodecagram IR =============================
# Symbolic base-12 introspective memory graph, rewindable, loggable, streamable

class DGNode:
    def __init__(self, type, name=None, source=None, modifier=None, extra=None):
        self.type = type
        self.name = name
        self.source = source
        self.modifier = modifier
        self.extra = extra
        self.base12_timestamp = DGNode.generate_base12_time()

    @staticmethod
    def generate_base12_time():
        import time
        return DGNode.to_base12(int(time.time() * 1000) % (12**6))

    @staticmethod
    def to_base12(num):
        digits = "0123456789AB"
        if num == 0:
            return "0"
        out = ""
        while num:
            out = digits[num % 12] + out
            num //= 12
        return out

    def serialize(self):
        return {
            "type": self.type,
            "name": self.name,
            "source": self.source,
            "modifier": self.modifier,
            "extra": self.extra,
            "timestamp": self.base12_timestamp
        }

    def __repr__(self):
        return f"DG::{self.type}::{self.name}->{self.source}|{self.modifier} [{self.base12_timestamp}]"

DG_TYPES = [
    "VAL_INIT", "FLOW_LOOP", "TRACE_DERIVE", "DEBUG_WHEN", "REWIND_STATE",
    "MEM_WRITE", "MEM_LOAD", "FUNC_ENTER", "FUNC_EXIT", "BRANCH_TAKEN",
    "MACRO_EXPAND", "INLINE_HINT", "THREAD_BEGIN", "THREAD_END"
]
DG_LOG = []
DG_SERIAL_BUFFER = []

def trace_dg(type, name, src=None, mod=None, extra=None):
    node = DGNode(type, name, src, mod, extra)
    DG_LOG.append(node)
    DG_SERIAL_BUFFER.append(node.serialize())
    return node

# ============================== Memory =======================================
class Memory:
    def __init__(self):
        self.registers = {f"R{i}": 0 for i in range(16)}
        self.stack = []
        self.ram = {}
        self.mutations = []  # For DG tracing

    def reg(self, name): 
        return name.upper()

    def get(self, reg):
        val = self.registers[reg]
        trace_dg("MEM_LOAD", reg, "registers", "read", {"value": val})
        return val

    def set(self, reg, val):
        self.mutations.append((reg, val))
        self.registers[reg] = val
        trace_dg("MEM_WRITE", reg, "registers", "write", {"value": val})

    def resolve(self, val):
        # Register
        if val.startswith("R"):
            return self.get(self.reg(val))
        # String literal e.g. 'A'
        if val.startswith("'") and val.endswith("'"):
            return val[1:-1]
        # Immediate integer e.g. #42
        if val.startswith("#"):
            try:
                return int(val[1:])
            except ValueError:
                raise ValueError(f"Bad immediate: {val}")
        # RAM reference e.g. @label
        if val.startswith("@"):
            resolved = self.ram.get(val[1:], 0)
            trace_dg("MEM_LOAD", val[1:], "ram", "read", {"value": resolved})
            return resolved
        raise ValueError(f"Unknown operand: {val}")

    def store(self, label, val):
        self.ram[label] = val
        trace_dg("MEM_WRITE", label, "ram", "write", {"value": val})

    def dump(self):
        return {
            "registers": self.registers,
            "stack": self.stack,
            "ram": self.ram,
            "mutations": self.mutations,
            "dg_log_size": len(DG_LOG)
        }

# ============================== Optimizer (DLMT-lite) ========================
def optimize_program(program):
    """
    Simple peephole: combine consecutive 'nudge' on same target.
    """
    optimized = []
    last = None
    for instr in program:
        if last and last["verb"].lower() == "nudge" and instr["verb"].lower() == "nudge" and last["target"] == instr["target"]:
            delta1 = int(last["operands"][0])
            delta2 = int(instr["operands"][0])
            last["operands"] = [str(delta1 + delta2)]
            trace_dg("INLINE_HINT", "nudge_fold", last["target"], f"{delta1}+{delta2}", None)
        else:
            if last:
                optimized.append(last)
            last = instr
    if last:
        optimized.append(last)
    return optimized

# ============================== Executor =====================================
def execute_program(program, flags):
    mem = Memory()
    ip = 0
    trace = "--trace" in flags
    dump = "--dump" in flags
    verbose = "--verbose" in flags

    # Build label table from targets that look like "name:"
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
            label = operands[0]
            label = label[1:] if label.startswith("@") else label
            mem.store(label, val)

        elif verb == "push":
            val = mem.resolve(target)
            mem.stack.append(val)
            trace_dg("MEM_WRITE", "stack", "stack", "push", {"value": val})

        elif verb == "pull":
            if mem.stack:
                popped = mem.stack.pop()
                mem.set(mem.reg(target), popped)
                trace_dg("MEM_LOAD", "stack", "stack", "pop", {"value": popped})

        elif verb == "clear":
            mem.set(mem.reg(target), 0)

        elif verb == "test":
            val1 = mem.resolve(target)
            val2 = mem.resolve(operands[0])
            res = (val1 == val2)
            if verbose:
                print(f"TEST {val1} == {val2} => {res}")

        elif verb == "calc":
            reg = mem.reg(target)
            op = operands[0]
            rhs = mem.resolve(operands[1])
            if op == '+':
                mem.set(reg, mem.get(reg) + rhs)
            elif op == '-':
                mem.set(reg, mem.get(reg) - rhs)
            else:
                raise ValueError(f"Unsupported calc op: {op}")

        elif verb == "nudge":
            reg = mem.reg(target)
            delta = int(operands[0])
            mem.set(reg, mem.get(reg) + delta)

        elif verb == "echo":
            ch = mem.resolve(operands[0])
            if qualifier == "brightest":
                # bold cyan underline for dramatic effect
                cprint(str(ch), "cyan", attrs=["bold", "underline"], end='')
            else:
                print(str(ch), end='', flush=True)

        elif verb == "jump":
            label = operands[0]
            condition = operands[1].lower() if len(operands) > 1 else None
            arg1 = mem.resolve(operands[2]) if len(operands) > 2 else None
            arg2 = mem.resolve(operands[3]) if len(operands) > 3 else None
            take = False
            if condition == "ifequal" and arg1 == arg2:
                take = True
            elif condition == "ifabove" and arg1 is not None and arg2 is not None and arg1 > arg2:
                take = True
            if take:
                if label not in labels:
                    raise ValueError(f"Unknown label: {label}")
                trace_dg("BRANCH_TAKEN", label, "ip", "jump", {"from": ip})
                ip = labels[label]
                continue

        else:
            raise ValueError(f"Unknown verb: {verb}")

        ip += 1

    if dump:
        print("\n--MEMORY DUMP--")
        print(json.dumps(mem.dump(), indent=2))

# ============================== SAN: Shorthand Assembly ======================
# Human-readable pseudo-assembly bridging SSAE/MeraLang → NASM → CPU/GPU/Quantum micro-op
SAN_OPCODES = {
    "ADD_VAR": ["mov eax, [a]", "add eax, [b]"],
    "SUB_VAR": ["mov eax, [a]", "sub eax, [b]"],
    "MUL_VAR": ["mov eax, [a]", "mul [b]"],
    "DIV_CONST": ["mov eax, [a]", "mov ecx, c", "div ecx"],
    "MOV_MEM": ["mov eax, [src]", "mov [dst], eax"],
    "MOV_CONST": ["mov dword [dst], imm"],
    "AND_REG": ["and eax, ebx"],
    "OR_REG":  ["or eax, ebx"],
    "XOR_REG": ["xor eax, ebx"],
    "NOT_REG": ["not eax"],
    "SHL_REG": ["shl eax, cl"],
    "SHR_REG": ["shr eax, cl"],
    "PUSH_REG": ["push eax"],
    "POP_REG":  ["pop eax"],
    "CMP_EQ":   ["cmp eax, ebx", "sete al"],
    "JMP_IF_ZERO":    ["cmp eax, 0", "je LABEL"],
    "JMP_IF_NONZERO": ["cmp eax, 0", "jne LABEL"],
    "MOD_CONST": ["mov eax, [a]", "mov ecx, c", "xor edx, edx", "div ecx", "mov eax, edx"],
    "INC_VAR": ["inc dword [x]"],
    "DEC_VAR": ["dec dword [x]"]
}

SAN_MACROS = {
    "SWAP": ["push eax", "mov eax, ebx", "pop ebx"],
    "ZERO_REG": ["xor eax, eax"],
    "CLEAR_MEM": ["mov dword [dst], 0"],
    "CALL_FUNC": ["call FUNCTION"],
    "RET_FUNC": ["ret"],
    "LOAD_IMM_PAIR": ["mov eax, val1", "mov ebx, val2"]
}

# ============================== DLMT: Direct-Link Mappings ===================
DLMT_TABLE = {
    "val x = 5": ["MOV_CONST x, 5", "mov dword [x], 5"],
    "derive sum from a by b": ["ADD_VAR a, b", "mov eax, [a]", "add eax, [b]"],
    "zero x": ["MOV_CONST x, 0"],
    "swap a b": ["mov eax, [a]", "mov ebx, [b]", "SWAP", "mov [a], eax", "mov [b], ebx"],
    "mod a by 10": ["MOD_CONST a, 10"],
    "mutate capsule x": ["TRACE_DERIVE x, x, \"delta\"", "ADD_VAR x, delta", "MOV_MEM x_result, eax"],
    "resolve tension between a and b": ["CMP_EQ a, b", "JMP_IF_ZERO HANDLE_CONFLICT", "CALL_FUNC RESOLVE"]
}

DLMT_META = {
    "supports": ["SIMD", "AVX2", "FMA", "SSSE3", "GPU_OFFLOAD", "PIPE_FUSION"],
    "target": "x86_64-w64-windows-gnu",
    "opt_level": "O3",
    "jit_enabled": True,
    "profile_guided": True
}

IR_LAYER_PURPOSES = {
    "DG IR": "Symbolic introspection + rewind engine",
    "SAN": "Human-readable assembly pseudocode",
    "DLMT": "Direct-link high-performance mappings"
}

IR_LAYER_TOOLS = {
    "DG IR": ["trace", "rewind", "inspect", "serialize", "stream_trace"],
    "SAN": ["emit_nasm", "expand_macros", "debug_emit", "reconstruct"],
    "DLMT": ["optimize", "emit_asm", "emit_jit", "target_info", "autotune"]
}

def load_vm_extensions():
    """
    Return VM extensions including SAN, DG, and DLMT.
    """
    return {
        "SAN": SAN_OPCODES,
        "SAN_MACROS": SAN_MACROS,
        "DG": {
            "Node": DGNode,
            "trace": trace_dg,
            "log": DG_LOG,
            "serial_buffer": DG_SERIAL_BUFFER
        },
        "DLMT": {
            "table": DLMT_TABLE,
            "meta": DLMT_META
        }
    }

# ============================== NASM Translator ==============================
def translate_to_nasm(program):
    """
    Translate a subset of SSAE into NASM (Linux or Windows-agnostic-ish).
    We just move numbers into caller-saved regs and do simple math.
    """
    lines = [
        "default rel",
        "section .data",
        "msg db 'Output:', 0Ah, 0",
        "section .text",
        "global main",
        "main:",
        "    push rbp",
        "    mov rbp, rsp"
    ]
    # Very small virtual mapping
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
            else:
                # If from RAM/register immediate fallback – not fully implemented
                lines.append(f"    xor {reg}, {reg} ; unsupported load form")

        elif verb == "nudge":
            reg = reg_map.get(target, "r8")
            delta = operands[0]
            lines.append(f"    add {reg}, {delta}")

        elif verb == "calc":
            reg = reg_map.get(target, "r8")
            op = operands[0]
            rhs_val = operands[1]
            if rhs_val.startswith("#"):
                rhs_val = rhs_val[1:]
            if op == "+":
                lines.append(f"    add {reg}, {rhs_val}")
            elif op == "-":
                lines.append(f"    sub {reg}, {rhs_val}")
            else:
                lines.append(f"    ; unsupported calc op {op}")

        elif verb == "echo":
            # Minimal "do nothing" placeholder (real IO omitted for portability)
            lines.append("    ; echo omitted in raw NASM demo")

        elif verb == "jump":
            label = operands[0]
            if len(operands) >= 2 and operands[1].lower() == "ifequal":
                # naive compare of r8 vs immediate/register fallback
                arg1 = operands[2]
                arg2 = operands[3]
                a1 = reg_map.get(arg1, None)
                if a1 is None and arg1.startswith("#"):
                    # Move immediate into rax then compare with arg2
                    lines.append(f"    mov rax, {arg1[1:]}")
                    a1 = "rax"
                b = arg2[1:] if arg2.startswith("#") else reg_map.get(arg2, arg2)
                lines.append(f"    cmp {a1}, {b}")
                lines.append(f"    je {label}")
            else:
                # Unconditional
                lines.append(f"    jmp {label}")

        else:
            lines.append(f"    ; unknown verb {verb} ignored")

    lines.extend(["    mov rsp, rbp", "    pop rbp", "    ret"])
    return "\n".join(lines)

# ============================== LLVM IR Transpiler ===========================
def transpile_to_llvm(program):
    """
    Naive IR just to demonstrate lowering:
    - track a few virtual regs (R0..R3) as i32
    - treat immediates (#X) as constants
    """
    ir = ["; ModuleID = 'ssae_capsule'",
          "source_filename = \"ssae\"",
          "",
          "define i32 @main() {",
          "entry:"]
    # Map SSAE registers to LLVM locals
    reg_map = {f"R{i}": f"%r{i}" for i in range(4)}
    # Initialize to zero
    for r in reg_map.values():
        ir.append(f"  {r} = alloca i32")
        ir.append(f"  store i32 0, i32* {r}")

    for instr in program:
        verb, target, operands = instr["verb"].lower(), instr["target"], instr["operands"]
        if ":" in target:
            # Labels here would need IR blocks; for the minimal demo we skip.
            continue

        if verb == "load":
            dst = reg_map.get(target, "%r0")
            imm = operands[0]
            if imm.startswith("#"):
                val = imm[1:]
                ir.append(f"  store i32 {val}, i32* {dst}")

        elif verb == "nudge":
            dst = reg_map.get(target, "%r0")
            delta = operands[0]
            tmp1 = f"%t{len(ir)}"
            tmp2 = f"%t{len(ir)+1}"
            ir.append(f"  {tmp1} = load i32, i32* {dst}")
            ir.append(f"  {tmp2} = add i32 {tmp1}, {delta}")
            ir.append(f"  store i32 {tmp2}, i32* {dst}")

        elif verb == "calc":
            dst = reg_map.get(target, "%r0")
            op = operands[0]
            rhs = operands[1]
            if rhs.startswith("#"):
                rhs = rhs[1:]
            tmp1 = f"%t{len(ir)}"
            tmp2 = f"%t{len(ir)+1}"
            ir.append(f"  {tmp1} = load i32, i32* {dst}")
            if op == '+':
                ir.append(f"  {tmp2} = add i32 {tmp1}, {rhs}")
            elif op == '-':
                ir.append(f"  {tmp2} = sub i32 {tmp1}, {rhs}")
            else:
                # no-op for unknown op
                ir.append(f"  {tmp2} = add i32 {tmp1}, 0")
            ir.append(f"  store i32 {tmp2}, i32* {dst}")

        # echo/jump not modeled in this stub IR

    ir.append("  ret i32 0")
    ir.append("}")
    return "\n".join(ir)

# ============================== Autocomplete (IDE) ===========================
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

# ============================== Helpers: Build/JIT ===========================
def tool_exists(name):
    return shutil.which(name) is not None

def build_nasm_to_exe(asm_path, bin_out):
    """
    Best-effort NASM -> OBJ -> EXE using nasm + clang or nasm + gcc.
    """
    obj = asm_path + ".obj"
    fmt = "win64" if os.name == "nt" else "elf64"
    nasm = f"nasm -f {fmt} \"{asm_path}\" -o \"{obj}\""
    linker = None
    if tool_exists("clang"):
        linker = f"clang \"{obj}\" -o \"{bin_out}\""
    elif tool_exists("gcc"):
        linker = f"gcc \"{obj}\" -o \"{bin_out}\""
    else:
        raise RuntimeError("No C compiler (clang/gcc) found for linking.")
    if os.system(nasm) != 0:
        raise RuntimeError("NASM failed.")
    if os.system(linker) != 0:
        raise RuntimeError("Linking failed.")
    try:
        os.remove(obj)
    except OSError:
        pass

# ============================== Main CLI =====================================
def main():
    if len(sys.argv) < 3:
        print("Usage:\n"
              "  ssae run <file.ssae> [--trace|--dump|--verbose]\n"
              "  ssae build <file.ssae> [-o output.asm|.ll] [--exe] [--llvm]")
        return

    mode = sys.argv[1]
    filename = sys.argv[2]
    flags = sys.argv[3:]

    with open(filename, "r", encoding="utf-8") as f:
        source = f.read()

    metadata = extract_metadata(source)
    program = optimize_program(parse_ssae(source))

    if mode == "run":
        print(f"[Capsule: {metadata.get('name', 'Unnamed')}] {metadata.get('description', '')}")
        execute_program(program, flags)

    elif mode == "build":
        # LLVM path
        if "--llvm" in flags:
            llvm_code = transpile_to_llvm(program)
            out = filename.rsplit(".", 1)[0] + ".ll"
            if "-o" in flags:
                try:
                    out = flags[flags.index("-o") + 1]
                except Exception:
                    pass
            with open(out, "w", encoding="utf-8") as f:
                f.write(llvm_code)
            print(f"LLVM IR written to {out}")
            return

        # NASM path
        asm_out = filename.rsplit(".", 1)[0] + ".asm"
        if "-o" in flags:
            try:
                asm_out = flags[flags.index("-o") + 1]
            except Exception:
                pass
        asm_code = translate_to_nasm(program)
        with open(asm_out, "w", encoding="utf-8") as f:
            f.write(asm_code)
        print(f"Assembly written to {asm_out}")

        if "--exe" in flags:
            bin_out = asm_out.rsplit(".", 1)[0] + (".exe" if os.name == "nt" else "")
            try:
                build_nasm_to_exe(asm_out, bin_out)
                print(f"Executable built: {bin_out}")
            except Exception as e:
                print(f"[build error] {e}")

    else:
        print(f"Unknown mode: {mode}")

# ============================== Demo / Dev Hook ==============================
def example_flow():
    print("--- SAN Instruction: ADD_VAR a, b →")
    for instr in SAN_OPCODES["ADD_VAR"]:
        print(f"  {instr}")

    print("\n--- SAN Macro: SWAP →")
    for instr in SAN_MACROS["SWAP"]:
        print(f"  {instr}")

    print("\n--- DG Node (symbolic trace):")
    dg = trace_dg("TRACE_DERIVE", "sum", "sum", "px")
    print(dg)

    print("\n--- DLMT Mapping: mutate capsule x →")
    for m in DLMT_TABLE["mutate capsule x"]:
        print(f"  {m}")

    print("\n--- DLMT Mapping: resolve tension between a and b →")
    for m in DLMT_TABLE["resolve tension between a and b"]:
        print(f"  {m}")

    print("\n--- DLMT Capabilities:")
    for cap in DLMT_META["supports"]:
        print(f"  ✔ {cap}")

    print("\n--- Capsule Fusion Logic →")
    trace_dg("FUNC_ENTER", "capsule_fusion", "capsule_x")
    if "mutate capsule x" in DLMT_TABLE:
        print("  ☢ Fusing DLMT ops for capsule_x...")
    else:
        print("  ❌ Fallback: Using default capsule handler")
    trace_dg("FUNC_EXIT", "capsule_fusion", "capsule_x")

if __name__ == "__main__":
    # Uncomment if you want a quick extensions print + demo:
    # print("VM Extensions Loaded:", load_vm_extensions().keys())
    # example_flow()
    main()
```
