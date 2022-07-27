import subprocess
from pyz3_utils.common import GlobalConfig


def get_git_revision_short_hash():
    hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return hash.decode()[:-1]


def setup_logger_file(test_func_frame, suffix="", log_dir='logs'):
    this_function_name = test_func_frame.f_code.co_name
    commit_id = get_git_revision_short_hash()
    GlobalConfig().log_to_file(
        "{}-{}-{}.log".format(this_function_name, commit_id, suffix), log_dir)
