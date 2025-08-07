# ---------------------- SSAE Standard Library ----------------------
# These capsules can be imported or copied into user .ssae scripts for reuse

ssae_stdlib = {
    "std_countdown": '''
@meta name: StdCountdown
@meta description: Counts down from a given number and prints stars.

Load R1 Fastest #5
Loop Deepest COUNT:
  Echo Char Loudest '*'
  Nudge R1 Fastest -1
  Jump COUNT Hardest IfAbove R1 #0
Echo Char Brightest '\n'
''',

    "std_compare": '''
@meta name: StdCompare
@meta description: Compares two values and prints = or ≠.

Load R1 Fastest #7
Load R2 Fastest #7
Test R1 Softest R2
Echo Char Loudest '='
''',

    "std_swap": '''
@meta name: StdSwap
@meta description: Swaps values between R1 and R2.

Push Stack Fastest R1
Push Stack Fastest R2
Pull R1 Fastest
Pull R2 Fastest
''',

    "std_clear_all": '''
@meta name: StdClearAll
@meta description: Clears R0 through R7.

Clear R0 Cleanest
Clear R1 Cleanest
Clear R2 Cleanest
Clear R3 Cleanest
Clear R4 Cleanest
Clear R5 Cleanest
Clear R6 Cleanest
Clear R7 Cleanest
''',

    "std_increment": '''
@meta name: StdIncrement
@meta description: Increments R1 by 1 and echoes it.

Nudge R1 Fastest +1
Echo Char Loudest R1
''',

    "std_store_and_reload": '''
@meta name: StdStoreReload
@meta description: Stores and reloads values using RAM @Label.

Load R1 Fastest #42
Store R1 Fastest @SavedVal
Load R2 Fastest @SavedVal
Echo Char Brightest R2
'''
}

# To inject or export capsules from the standard library:
def get_stdlib_capsule(name):
    return ssae_stdlib.get(name.lower())
def list_stdlib_capsules():
    return list(ssae_stdlib.keys())
def add_stdlib_capsule(name, code):
    ssae_stdlib[name.lower()] = code
    def remove_stdlib_capsule(name):
        name = name.lower()
        if name in ssae_stdlib:
            del ssae_stdlib[name]
        else:
            raise KeyError(f"Capsule '{name}' not found in standard library.")
        def update_stdlib_capsule(name, code):
            name = name.lower()
            if name in ssae_stdlib:
                ssae_stdlib[name] = code
            else:
                raise KeyError(f"Capsule '{name}' not found in standard library.")
            return get_stdlib_capsule, list_stdlib_capsules, add_stdlib_capsule, remove_stdlib_capsule, update_stdlib_capsule
        # Return the functions for external use
        return get_stdlib_capsule, list_stdlib_capsules, add_stdlib_capsule, remove_stdlib_capsule, update_stdlib_capsule
    

