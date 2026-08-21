"""Quick offline test for contacts whitelist matching logic."""
import re, json, os

contacts_path = os.path.join(os.path.dirname(__file__), "backend", "data", "contacts.json")
contacts = json.load(open(contacts_path))

def check_saved_contact(caller_num):
    if not caller_num:
        return None
    clean = re.sub(r'[\s\-()]+', '', caller_num)
    for c in contacts:
        clean_c = re.sub(r'[\s\-()]+', '', c.get("phone", ""))
        if clean == clean_c or (len(clean) >= 7 and clean_c.endswith(clean)) or (len(clean_c) >= 7 and clean.endswith(clean_c)):
            return c
    return None

tests = [
    ("+91 98765 43210", "Dad"),
    ("+919876543210",   "Dad"),        # no spaces/dashes variant
    ("+91 98765 00000", "Mom"),
    ("+1 555-0199",     "Alex (Boss)"),
    ("+44 7911 123456", "John (Colleague)"),
    ("+1 999-999-9999", None),          # unknown number
    ("",                None),          # empty
]

print("\n=== Contacts Whitelist Test ===\n")
all_pass = True
for num, expected_name in tests:
    result = check_saved_contact(num)
    actual = result["name"] if result else None
    passed = actual == expected_name
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] '{num}' => {actual!r}  (expected {expected_name!r})")
    if not passed:
        all_pass = False

print()
if all_pass:
    print("ALL TESTS PASSED ✅")
else:
    print("SOME TESTS FAILED ❌")
