# ---------------------- Term Color Module (Extended) ----------------------
# Full terminal coloring system with styles, RGB, and block formatting

class TermColor:
    COLORS = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "underline": "\033[4m",
        "blink": "\033[5m",
        "reverse": "\033[7m",
        "invisible": "\033[8m"
    }

    BG_COLORS = {
        "bg_black": "\033[40m",
        "bg_red": "\033[41m",
        "bg_green": "\033[42m",
        "bg_yellow": "\033[43m",
        "bg_blue": "\033[44m",
        "bg_magenta": "\033[45m",
        "bg_cyan": "\033[46m",
        "bg_white": "\033[47m"
    }

    SUPERLATIVE_STYLES = {
        "brightest": ["bold", "underline"],
        "loudest": ["bold", "blink"],
        "softest": ["dim"],
        "cleanest": ["bold"],
        "deepest": ["reverse"],
    }

    @staticmethod
    def print(text, color="reset", style=None, bg=None, end="\n"):
        style_sequence = ""
        if style in TermColor.SUPERLATIVE_STYLES:
            for s in TermColor.SUPERLATIVE_STYLES[style]:
                style_sequence += TermColor.COLORS.get(s, "")
        else:
            style_sequence = TermColor.COLORS.get(style.lower(), "") if style else ""
        fg_code = TermColor.COLORS.get(color.lower(), "")
        bg_code = TermColor.BG_COLORS.get(bg.lower(), "") if bg else ""
        print(f"{style_sequence}{fg_code}{bg_code}{text}{TermColor.COLORS['reset']}", end=end)

    @staticmethod
    def cprint(text, color="reset", style=None, bg=None, end="\n"):
        TermColor.print(text, color=color, style=style, bg=bg, end=end)

    @staticmethod
    def rgb(r, g, b):
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def bg_rgb(r, g, b):
        return f"\033[48;2;{r};{g};{b}m"

    @staticmethod
    def block(text, fg_rgb=(255,255,255), bg_rgb=(0,0,0), style=None, pad=1):
        fg = TermColor.rgb(*fg_rgb)
        bg = TermColor.bg_rgb(*bg_rgb)
        style_code = ""
        if style in TermColor.SUPERLATIVE_STYLES:
            for s in TermColor.SUPERLATIVE_STYLES[style]:
                style_code += TermColor.COLORS.get(s, "")
        else:
            style_code = TermColor.COLORS.get(style, "") if style else ""
        padding = " " * pad
        print(f"{bg}{fg}{style_code}{padding}{text}{padding}{TermColor.COLORS['reset']}")

    @staticmethod
    def banner(text, fg="cyan", bg="bg_black", width=60):
        edge = "=" * width
        TermColor.print(edge, color=fg, bg=bg)
        TermColor.print(text.center(width), color=fg, style="bold", bg=bg)
        TermColor.print(edge, color=fg, bg=bg)

    @staticmethod
    def highlight_capsule_state(state):
        if state.lower() == "volatile":
            TermColor.block("⚠️ Volatile Capsule", fg_rgb=(255, 255, 0), bg_rgb=(80, 0, 0), style="brightest")
        elif state.lower() == "stable":
            TermColor.block("✅ Stable Capsule", fg_rgb=(0, 255, 0), bg_rgb=(0, 40, 0), style="bold")
        elif state.lower() == "corrupt":
            TermColor.block("🛑 Corrupt Capsule", fg_rgb=(255, 255, 255), bg_rgb=(200, 0, 0), style="reverse")

    @staticmethod
    def banner_phase(label):
        TermColor.banner(f"🔍 {label} Phase")

    @staticmethod
    def demo_all():
        for color in TermColor.COLORS:
            TermColor.print(f"{color:<10}", color=color)
        for bg in TermColor.BG_COLORS:
            TermColor.print(f"{bg:<10}", bg=bg)
        TermColor.block("RGB Demo", (255, 255, 0), (0, 0, 255))
        TermColor.banner("🎨 TermColor Full Demo")
        TermColor.highlight_capsule_state("volatile")
        TermColor.highlight_capsule_state("stable")
        TermColor.highlight_capsule_state("corrupt")
        TermColor.banner_phase("Introspection")
