import configparser

parser = configparser.ConfigParser()
parser.read("pyproject.toml")


def get_version() -> str:
    return parser["tool.poetry"]["version"].replace('"', "")
