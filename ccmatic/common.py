

# From https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    GENERATOR = OKCYAN
    VERIFIER = WARNING
    CANDIDATESOLUTION = BOLD + OKBLUE
    PROVEDSOLUTION = BOLD + OKGREEN

    @staticmethod
    def generator(s: str):
        return bcolors.GENERATOR + s + bcolors.ENDC

    @staticmethod
    def verifier(s: str):
        return bcolors.VERIFIER + s + bcolors.ENDC

    @staticmethod
    def candidate(s: str):
        return bcolors.CANDIDATESOLUTION + s + bcolors.ENDC

    @staticmethod
    def proved(s: str):
        return bcolors.PROVEDSOLUTION + s + bcolors.ENDC