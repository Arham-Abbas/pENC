import win32com.client

def resolve_shortcut(shortcut_path):
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(shortcut_path)
    return shortcut.Targetpath

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python resolve_shortcut.py <shortcut_path>")
    else:
        shortcut_path = sys.argv[1]
        resolved_path = resolve_shortcut(shortcut_path)
        print(f"Resolved path: {resolved_path}")
