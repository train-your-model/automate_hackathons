# ------------------
# IMPORTS
# ------------------
import argparse
import sys

from utils import supplementary as sp


# ------------------
# PARSING
# ------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="Setting-up Directories, Pipelines for automating the Hackathon as much as possible",
        description="Provides functionality for the user to set-up directories and subsequent steps needed"
                    "for completion of a hackathon based ML project"
    )

    subparsers = p.add_subparsers(
        dest="command",
        required=True
    )

    # Dumping File Paths
    path_parser = subparsers.add_parser(
        'path_setup', help="Dumps the extracted paths for the yaml and json files required later for referencing"
    )

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "path_setup":
        sp.PathDump().paths_dump()
        sp.create_config_json()
        sp.yaml_path_dump()


# ------------------
# INITIALIZATION
# ------------------
if __name__ == "__main__":
    main()