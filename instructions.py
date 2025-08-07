# instructions.py
from memory import Memory

def execute_program(program):
    mem = Memory()
    ip = 0  # instruction pointer

    labels = {line["target"].strip(":"): i for i, line in enumerate(program) if ":" in line["target"]}

    while ip < len(program):
        line = program[ip]
        verb = line["verb"].lower()
        target = line["target"]
        qualifier = line["qualifier"].lower()
        operands = line["operands"]

        if ":" in target:  # label
            ip += 1
            continue

        if verb == "load":
            reg = mem.reg(target)
            val = mem.resolve(operands[0])
            mem.set(reg, val)

        elif verb == "calc":
            reg = mem.reg(target)
            op = operands[0]
            rhs = mem.resolve(operands[1])
            if op == '+': mem.set(reg, mem.get(reg) + rhs)
            if op == '-': mem.set(reg, mem.get(reg) - rhs)

        elif verb == "nudge":
            reg = mem.reg(target)
            delta = 1 if operands[0] == "+1" else -1
            mem.set(reg, mem.get(reg) + delta)

        elif verb == "echo":
            if target.lower() == "char":
                print(mem.resolve(operands[0]), end='', flush=True)

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
        return mem  # Return memory state after execution for inspection
    def execute_instruction(instruction, mem):
        verb = instruction["verb"].lower()
        target = instruction["target"]
        qualifier = instruction["qualifier"].lower()
        operands = instruction["operands"]
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
            return val1 == val2
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
            print(str(ch), end='', flush=True)
        return None
    def execute_program(program):
        mem = Memory()
        ip = 0
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
            if verb == "load":
                reg = mem.reg(target)
                val = mem.resolve(operands[0])
                mem.set(reg, val)
mem.set(mem.reg(target), 0)

