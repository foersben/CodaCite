import keyring
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_keyring():
    print("Keyring backends available:")
    from keyring.backend import get_all_keyring
    for backend in get_all_keyring():
        print(f"  - {backend}")
    
    print("\nCurrent backend:")
    print(f"  {keyring.get_keyring()}")
    
    try:
        print("\nAttempting to retrieve password for Service='Gemini_API', User='gemini_user'...")
        key = keyring.get_password("Gemini_API", "gemini_user")
        if key:
            print(f"SUCCESS: Key found (length: {len(key)})")
        else:
            print("FAILURE: Key not found (None returned)")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_keyring()
