# -------
# IMPORTS
# -------
import argparse
import sys

from utils import set_ups as st
from utils import supplementary as sp
from utils import operations as op

# --------
# PARSING
# --------
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

    # Normal Operation
    ops = subparsers.add_parser(
        'norm_ops', help="Deals with the Normal Operation "
    )
    ops.add_argument("site_abv", type=str,
                     help="Abbreviated Site Name")
    ops.add_argument("--proj_name", type=str,
                     help="Name of the Project-specific directory")
    ops.add_argument("--api", action="store_true")

    return p


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "path_setup":
        st.PathDump().run()
        sp.create_root_dir()
        print("Required Initial Set-Up has been completed")

    if args.command == "norm_ops":
        if args.api:
            site_url = input("Enter the URL of the site for the data to be downloaded: ")
            op.NormOps(site_abv=args.site_abv).run(proj_name=args.proj_name, url=site_url)

        else:
            op.NormOps(site_abv=args.site_abv).run(proj_name=args.proj_name)


# ------------------
# INITIALIZATION
# ------------------
if __name__ == "__main__":
    main()