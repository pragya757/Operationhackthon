from pathlib import Path
main_py = Path("backend/main.py").resolve()
contacts = main_py.parent / "data" / "contacts.json"
print("main.py  :", main_py)
print("contacts :", contacts)
print("exists   :", contacts.exists())
if contacts.exists():
    print("content  :", contacts.read_text())
