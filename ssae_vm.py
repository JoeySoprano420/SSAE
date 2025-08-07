# ssae_vm.py
from parser import parse_ssae
from instructions import execute_program

import sys

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        print("Usage: ssae run <filename.ssae>")
        return

    with open(sys.argv[2], "r") as f:
        source_code = f.read()

    program = parse_ssae(source_code)
    execute_program(program)

if __name__ == "__main__":
    main()
