import win32com.client
import argparse

def resolve_shortcut(shortcut_path):
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortcut(shortcut_path)
    return shortcut.Targetpath

def main():
    parser = argparse.ArgumentParser(description='Resolve a Windows shortcut to its target path.')
    parser.add_argument('shortcut_path', help='The path to the shortcut (.lnk) file.')
    args = parser.parse_args()
    
    resolved_path = resolve_shortcut(args.shortcut_path)
    print(f"Resolved path: {resolved_path}")

if __name__ == "__main__":
    main()
