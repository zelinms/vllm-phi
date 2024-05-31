import pkg_resources
import shutil
import os

def replace_triton_cuda():

    def get_package_path(package_name):
        return pkg_resources.get_distribution(package_name).location

    def get_package_version(package_name):
        return pkg_resources.get_distribution(package_name).version
    
    assert get_package_version('triton') == "2.3.0" or get_package_version('triton') == "2.2.0"

    cur_folder_cuda_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'triton_cuda.py')
    target_folder_cuda_py = os.path.join(get_package_path('triton'), 'triton', 'language', 'extra', 'cuda.py')
    shutil.copyfile(cur_folder_cuda_py, target_folder_cuda_py)

if __name__ == "__main__":
    replace_triton_cuda()