import os
import shutil

def remove_python_caches(start_path='.', names_to_delete=None):
    if names_to_delete is None:
        names_to_delete = {'__pycache__', '.mypy_cache', '.pytest_cache'}

    deleted_dirs = 0
    deleted_files = 0

    for root, dirs, files in os.walk(start_path):
        # Remove matching directories
        for dir_name in dirs:
            if dir_name in names_to_delete:
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted directory: {dir_path}")
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Failed to delete {dir_path}: {e}")
        
        # Remove .pyc files
        for file_name in files:
            if file_name.endswith('.pyc'):
                file_path = os.path.join(root, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    print(f"\nDone. Deleted {deleted_dirs} cache directories and {deleted_files} .pyc files.")

if __name__ == '__main__':
    remove_python_caches('.')
